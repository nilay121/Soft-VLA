#!/usr/bin/env python3
"""
Behaviour-clone the tracking demonstrations into the VLA.

    python train.py                            # the default 'both'
    python train.py --conditioning future_only
    python train.py --conditioning past_only

Reads dataset/ (generate_demos.py), writes checkpoints/track_vla_<cond>.pt.

The network is the pick-and-place one, unchanged (`vla_model.PickPlaceVLA`,
ImageNet-pretrained ResNet18): two 112x112 views x N frames -> 3*2*N
channels, plus the 7 pressures it commanded last step, out to a chunk of 10
pressure CHANGES of which 5 are executed.

WHAT IS DIFFERENT FROM PICK-AND-PLACE
-------------------------------------
Only which frames go into the stack. There, all N frames came from BEFORE
the current step. Here some come from AFTER it, and those are the goal:
pictures of the arm in the configurations it is supposed to reach next.
`conditioning.py` explains why the task is unanswerable without them, and
defines the three variants compared in `compare_conditioning.py`. All three
use four frames, so the architecture and parameter count never change - only
the meaning of the channels does.

Everything else is inherited from the pick-and-place recipe because it was
measured there: predict the CHANGE rather than the absolute pressure (the
absolute map is nearly the identity and deploys as a frozen arm), predict a
chunk rather than a single step, add noise to the previous-pressure input so
the policy learns to correct its own drift, and augment photometrically
only - never geometrically, because with fixed cameras the pixel position IS
the position signal.

The split is by EPISODE: consecutive frames are near-duplicates, so a
frame-level split would leak an episode into both sides.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
SIMULATOR = HERE.parent.parent / "simulator"
for path in (str(SIMULATOR), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

import conditioning as C  # noqa: E402
from augment import DEFAULTS, photometric_jitter, view_channel_groups  # noqa: E402
from vla_model import (  # noqa: E402
    CHUNK_STEPS,
    NUM_VIEWS,
    PRESSURE_LABELS,
    VISUAL_ENCODERS,
    PickPlaceVLA,
    count_parameters,
    save_checkpoint,
    stack_frames,
)

# dataset/, goals/, checkpoints/, results/ live in this folder, or in
# TRACK_OUT (e.g. seeds/, the 3-seed study; relative = to this folder)
OUT_DIR = HERE / os.environ.get("TRACK_OUT", ".")
DATA_DIR = OUT_DIR / "dataset"
CKPT_DIR = OUT_DIR / "checkpoints"


class TrackDataset(Dataset):
    """Frames at `offsets` around t, plus the previous command."""

    def __init__(self, episodes, index, offsets, augment=False,
                 state_noise=0.0, strength=None, seed=0):
        self.episodes = episodes
        self.index = index
        self.offsets = list(offsets)
        self.augment = augment
        self.state_noise = state_noise
        self.strength = strength
        self.groups = view_channel_groups(NUM_VIEWS, len(self.offsets))
        self.seed = seed
        self._rng = None

    @property
    def rng(self):
        # per worker AND per epoch: DataLoader redraws the worker seed each
        # epoch, so the jitter never replays
        if self._rng is None:
            info = torch.utils.data.get_worker_info()
            worker = 0 if info is None else info.id
            self._rng = np.random.default_rng(
                (int(torch.initial_seed()) + 7919 * (worker + 1)
                 + self.seed) % (2 ** 63))
        return self._rng

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        ep, t = self.index[i]
        frames = self.episodes[ep]["frames"]
        pressure = self.episodes[ep]["pressure"]
        n = len(pressure)

        # Clamped at both ends: before the start the arm was where it
        # started, and past the end the reference has stopped moving.
        picks = [int(np.clip(t + o, 0, n - 1)) for o in self.offsets]
        images = stack_frames([frames[p] for p in picks])

        if self.augment:
            images = photometric_jitter(images, self.rng, self.groups,
                                        self.strength)

        previous = pressure[t - 1].copy()
        future = np.concatenate([pressure[t:t + CHUNK_STEPS],
                                 np.repeat(pressure[-1:], CHUNK_STEPS, axis=0)]
                                )[:CHUNK_STEPS]
        delta = np.diff(np.vstack([previous, future]),
                        axis=0).astype(np.float32)
        if self.state_noise:
            previous = previous + self.rng.normal(
                0.0, self.state_noise, previous.shape).astype(np.float32)

        return (torch.from_numpy(np.ascontiguousarray(images)),
                torch.from_numpy(previous), torch.from_numpy(delta))


def load_episodes():
    files = sorted(DATA_DIR.glob("episode_*.npz"))
    if not files:
        raise SystemExit(f"no demonstrations in {DATA_DIR} - "
                         "run generate_demos.py first")
    episodes = []
    for f in files:
        d = np.load(f)
        episodes.append({"frames": np.load(f.with_suffix(".npy"),
                                           mmap_mode="r"),
                         "pressure": d["pressure"], "name": f.stem})
    return episodes


def evaluate(model, loader, device):
    model.eval()
    errors, targets = [], []
    with torch.no_grad():
        for images, state, target in loader:
            out = model(images.to(device), state.to(device))
            errors.append((out.cpu() - target).numpy()
                          .reshape(-1, out.shape[-1]))
            targets.append(target.numpy().reshape(-1, out.shape[-1]))
    err, tgt = np.concatenate(errors), np.concatenate(targets)
    return {"mae_bar": float(np.abs(err).mean()),
            "rmse_bar": float(np.sqrt((err ** 2).mean())),
            "mae_per_line_bar": np.abs(err).mean(0).tolist(),
            "baseline_mae_bar": float(np.abs(tgt).mean()),
            "fraction_of_baseline": float(np.abs(err).mean()
                                          / max(np.abs(tgt).mean(), 1e-9))}


def main(args):
    n_past, n_future, label = C.resolve(args.conditioning)
    offsets = C.frame_offsets(n_past, n_future)

    print()
    print("=" * 76)
    print(f"TRAJECTORY-TRACKING VLA — behaviour cloning [{args.conditioning}]")
    print("=" * 76)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    episodes = load_episodes()
    meta = json.loads((DATA_DIR / "metadata.json").read_text())

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(episodes))
    n_val = max(1, int(0.1 * len(episodes)))
    splits = {"val": order[:n_val], "test": order[n_val:2 * n_val],
              "train": order[2 * n_val:]}
    empty = [k for k, v in splits.items() if len(v) == 0]
    if empty:
        raise SystemExit(
            f"only {len(episodes)} demonstration(s) in {DATA_DIR}: the "
            f"{', '.join(empty)} split(s) came out empty - at least 3 are "
            "needed, realistically a few hundred.")

    strength = None
    if args.augment and args.aug_scale != 1.0:
        strength = {k: v * args.aug_scale for k, v in DEFAULTS.items()}

    loaders, sizes = {}, {}
    for name, eps in splits.items():
        stride = args.sample_stride if name == "train" else 1
        index = [(int(e), t) for e in eps
                 for t in range(1, len(episodes[e]["pressure"]), stride)]
        ds = TrackDataset(episodes, index, offsets,
                          augment=(name == "train" and args.augment),
                          state_noise=args.state_noise if name == "train" else 0,
                          strength=strength, seed=args.seed)
        loaders[name] = DataLoader(ds, batch_size=args.batch,
                                   shuffle=(name == "train"),
                                   num_workers=args.workers, pin_memory=False)
        sizes[name] = len(ds)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PickPlaceVLA(frame_history=len(offsets), arch=args.arch,
                         num_views=NUM_VIEWS)

    train_states = np.concatenate(
        [episodes[e]["pressure"][:-1] for e in splits["train"]])
    train_actions = np.concatenate(
        [np.diff(episodes[e]["pressure"], axis=0) for e in splits["train"]])
    model.set_normalisation(train_states, train_actions)
    model = model.to(device)

    print(f"  demos      : {len(episodes)} episodes "
          f"({meta['frames_kept']:,} frames at "
          f"{1 / meta['policy_dt_s']:.0f} Hz)")
    print(f"  condition  : {label}")
    print(f"  frames     : offsets {offsets} policy steps "
          f"(negative = history, 0 = now, positive = goal)")
    print(f"  channels   : {3 * NUM_VIEWS * len(offsets)} "
          f"({len(offsets)} frames x {NUM_VIEWS} views x RGB)")
    print(f"  samples    : " + ", ".join(f"{k} {v:,}" for k, v in sizes.items())
          + "   (split by episode)")
    print(f"  encoder    : {args.arch}   augment: "
          + ("photometric only" if args.augment else "none"))
    print(f"  parameters : {count_parameters(model):,}")
    print(f"  device     : {device}")
    print()

    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=4, min_lr=1e-6)
    criterion = nn.MSELoss()

    ckpt_path = (Path(args.out) if args.out else
                 CKPT_DIR / f"track_vla_{args.conditioning}.pt")
    best, best_epoch, stale, history = float("inf"), -1, 0, []
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for images, state, target in loaders["train"]:
            images, state = images.to(device), state.to(device)
            target = target.to(device)
            optimiser.zero_grad(set_to_none=True)
            loss = criterion(model(images, state), target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            running += loss.item() * images.shape[0]
            seen += images.shape[0]

        val = evaluate(model, loaders["val"], device)
        scheduler.step(val["mae_bar"])
        history.append({"epoch": epoch, "train_mse": running / max(seen, 1),
                        **val})
        print(f"  epoch {epoch:03d}/{args.epochs}  "
              f"train {running / max(seen, 1):.6f}  "
              f"val MAE {val['mae_bar']:.5f} bar  "
              f"({100 * val['fraction_of_baseline']:.0f}% of the do-nothing "
              f"baseline)")

        if val["mae_bar"] < best:
            best, best_epoch, stale = val["mae_bar"], epoch, 0
            save_checkpoint(ckpt_path, model, extra={
                "epoch": epoch, "val_mae_bar": val["mae_bar"],
                "image_size": meta["image_size"],
                "policy_dt_s": meta["policy_dt_s"],
                "cameras": meta["cameras"],
                "task": meta["task"],
                "conditioning": args.conditioning,
                "conditioning_label": label,
                "n_past": n_past, "n_future": n_future,
                "frame_offsets": offsets,
                "past_stride": C.PAST_STRIDE,
                "future_stride": C.FUTURE_STRIDE,
                "augmentation": ("photometric only" if args.augment
                                 else "none"),
                "chunk_steps": CHUNK_STEPS,
                "pressure_labels": list(PRESSURE_LABELS),
                "dataset_metadata": {k: meta[k] for k in
                                     ("episodes_kept", "frames_kept",
                                      "envelope", "speed_mps", "duration_s",
                                      "shape_reject", "gripper")},
            })
        else:
            stale += 1
            if stale >= args.patience:
                print("  early stopping")
                break

    from vla_model import load_checkpoint
    model, _ = load_checkpoint(ckpt_path, device)
    test = evaluate(model, loaders["test"], device)
    elapsed = time.perf_counter() - started

    report = {"conditioning": args.conditioning, "label": label,
              "n_past": n_past, "n_future": n_future,
              "frame_offsets": offsets, "arch": args.arch,
              "augmentation": "photometric only" if args.augment else "none",
              "best_epoch": best_epoch, "best_val_mae_bar": best,
              "test": test, "splits": sizes,
              "training_seconds": elapsed, "history": history}
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path.with_name(ckpt_path.stem + "_report.json").write_text(
        json.dumps(report, indent=2))

    print()
    print("=" * 76)
    print("TRAINING COMPLETE")
    print("=" * 76)
    print(f"  conditioning : {label}")
    print(f"  best epoch   : {best_epoch}")
    print(f"  test MAE     : {test['mae_bar']:.5f} bar of pressure change")
    print(f"  baseline     : {test['baseline_mae_bar']:.5f} bar if it "
          f"predicted no change -> {100 * test['fraction_of_baseline']:.0f}%")
    print(f"  checkpoint   : {ckpt_path}")
    print(f"  wall clock   : {elapsed / 60:.1f} min")
    print()
    print("  This number decides nothing - judge the policy by deploy.py,")
    print("  which measures millimetres on the actual circle.")
    print("=" * 76)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Train the trajectory-tracking VLA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--conditioning", default=C.DEFAULT_VARIANT,
                   choices=tuple(C.VARIANTS),
                   help="Which frames the policy sees; see conditioning.py.")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--arch", choices=tuple(VISUAL_ENCODERS),
                   default="resnet18")
    p.add_argument("--augment", dest="augment", action="store_true",
                   default=True)
    p.add_argument("--no-augment", dest="augment", action="store_false")
    p.add_argument("--aug-scale", type=float, default=1.0)
    p.add_argument("--sample-stride", type=int, default=2)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--state-noise", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
