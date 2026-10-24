#!/usr/bin/env python3
"""
Learn the three pick-and-drop tasks one after another.

    python train_continual.py --replay latent
    python train_continual.py --replay none     # the forgetting baseline
    python train_continual.py --replay frozen   # freezing alone, no replay
    python train_continual.py --replay raw      # image replay, upper bound

Stage k trains on task k's demonstrations ONLY. Nothing else about the task
sequence is given to the network: no task index, no "this is stage 2", no
access to task k-1's data except through whatever the replay buffer kept.
After every stage the checkpoint is evaluated offline on every task seen so
far, and `deploy.py` closes the loop on them in simulation.

THE FOUR ARMS, AND WHY THERE ARE FOUR
-------------------------------------
Latent replay does two things at once - it rehearses old samples AND it has
to freeze the encoders to keep those samples valid. Reporting it against
plain fine-tuning alone would not say which half did the work, so freezing
is also run on its own:

    arm       encoders          buffer            what it measures
    none      keep learning     -                 catastrophic forgetting
    frozen    frozen after t1   -                 what freezing alone buys
    latent    frozen after t1   768-d latents     the method
    raw       keep learning     demo frames       the upper bound, same
                                                  sample budget

Stage 1 is identical in all four arms (nothing is frozen yet and there is
nothing to replay), so it is trained ONCE and reused - `checkpoints/
stage1_cube.pt`. Any difference between arms in the stage-1 row of the
results is therefore a bug, which makes it a free correctness check.

WHAT IS HELD FIXED ACROSS STAGES
--------------------------------
The network's shape, the vocabulary, and the input/output standardisation -
all fixed before stage 1 and never touched. Restandardising at stage 2 would
change what the head's numbers mean and silently invalidate every latent
already in the buffer; growing the vocabulary would resize the embedding
table mid-experiment. Both would show up as "forgetting" that was really
surgery.

THE OFFLINE NUMBER RANKS BADLY
------------------------------
Validation MAE on the pressure change is used to pick the epoch, because
something has to be. It is a poor judge of whether a policy can do the task:
on the pick-and-place version a 0/10 policy scored better offline than a
4/10 one. `deploy.py` is the measurement; this is bookkeeping.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
SIMULATOR = HERE.parent.parent / "simulator"
for path in (str(SIMULATOR), str(HERE)):
    while path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)

import tasks  # noqa: E402
from augment import DEFAULTS  # noqa: E402
from data import Corpus, DemoDataset  # noqa: E402
from language import Vocabulary  # noqa: E402
from replay import STRATEGIES, make_buffer  # noqa: E402
from vla_model import (  # noqa: E402
    ABLATIONS,
    MINILM_ARCHS,
    CHUNK_STEPS,
    FRAME_HISTORY,
    FRAME_STRIDE,
    NUM_VIEWS,
    PRESSURE_LABELS,
    ContinualVLA,
    count_parameters,
    load_checkpoint,
    save_checkpoint,
)

# where checkpoints and results go: this folder, or CONTINUAL_OUT (e.g.
# seeds/, the 3-seed study run by run_seeds.sh); relative = to this folder
OUT_DIR = HERE / os.environ.get("CONTINUAL_OUT", ".")
CKPT_DIR = OUT_DIR / "checkpoints"
RESULTS_DIR = OUT_DIR / "results"
SHARED_STAGE1 = CKPT_DIR / "stage1_cube.pt"


def shared_stage1(arch, data_dir, ablate=()):
    """Stage 1 is only identical across arms that share the architecture,
    the removed modules and the demonstrations, so each combination gets its
    own copy. The original resnet18 / dataset pair keeps the historical name;
    a model with FiLM (the pre-2026-10-03 default) has no suffix."""
    if arch == "resnet18" and data_dir == "dataset":
        return SHARED_STAGE1
    tag = f"_minus_{'_'.join(ablate)}" if ablate else ""
    return CKPT_DIR / f"stage1_cube_{arch}{tag}_{Path(data_dir).name}.pt"

RAW_SAMPLE_BYTES = FRAME_HISTORY * NUM_VIEWS * 112 * 112 * 3


def encoder_fingerprint(model):
    """A short hash of the encoder weights.

    The latent buffer refuses a second fill under a different fingerprint:
    latents produced by encoders that have since moved are not training data,
    they are noise, and that failure is otherwise invisible.
    """

    digest = hashlib.blake2b(digest_size=8)
    for module in model.encoder_modules():
        for name, p in sorted(module.state_dict().items()):
            digest.update(name.encode())
            digest.update(p.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def dev(x, device):
    """Move a batch field to the device; raw instruction strings stay put."""
    return x.to(device) if torch.is_tensor(x) else x


def forward_batch(model, images, state, ids, mask):
    """The policy's prediction, skipping the encoder backward when frozen."""

    if model.encoders_frozen:
        with torch.no_grad():
            latent = model.encode(images, state, ids, mask)
        return model.head(latent)
    return model(images, state, ids, mask)


def evaluate(model, loader, device):
    """Error of the predicted pressure change against predicting nothing."""

    was_training = model.training
    model.eval()
    errors, targets = [], []
    with torch.no_grad():
        for images, state, ids, mask, target, _ in loader:
            out = model(images.to(device), state.to(device), dev(ids, device),
                        dev(mask, device))
            errors.append((out.cpu() - target).numpy().reshape(-1,
                                                               out.shape[-1]))
            targets.append(target.numpy().reshape(-1, out.shape[-1]))
    model.train(was_training)
    err, tgt = np.concatenate(errors), np.concatenate(targets)
    return {"mae_bar": float(np.abs(err).mean()),
            "rmse_bar": float(np.sqrt((err ** 2).mean())),
            "grip_mae_bar": float(np.abs(err[:, -1]).mean()),
            "baseline_mae_bar": float(np.abs(tgt).mean()),
            "fraction_of_baseline": float(np.abs(err).mean()
                                          / max(np.abs(tgt).mean(), 1e-9))}


def make_loader(corpus, samples, vocab, args, train, seed, batch=None,
                workers=None, aux=False):
    strength = None
    if args.augment and args.aug_scale != 1.0:
        strength = {k: v * args.aug_scale for k, v in DEFAULTS.items()}
    ds = DemoDataset(corpus, samples, vocab,
                     augment=train and args.augment,
                     state_noise=args.state_noise if train else 0.0,
                     strength=(strength if args.augment else {}), seed=seed,
                     aux=aux, raw_text=args.arch in MINILM_ARCHS)
    return DataLoader(ds, batch_size=batch or args.batch, shuffle=train,
                      num_workers=(args.workers if workers is None
                                   else workers),
                      pin_memory=False, drop_last=train)


def cycling(loader):
    while True:
        for batch in loader:
            yield batch


# ---------------------------------------------------------------------------
# Filling the latent buffer
# ---------------------------------------------------------------------------

def fill_latent_buffer(model, corpus, task, vocab, args, buffer, device):
    """Store `capacity` replay-layer latents from `task`'s training split.

    The latents are computed WITH photometric jitter and WITH the state noise
    the trainer uses, and with a randomly drawn paraphrase - so a stored
    latent has the same distribution as a live one. Storing clean latents
    would mean rehearsing a distribution the trunk never sees otherwise.
    """

    samples = corpus.samples(task, "train")
    rng = np.random.default_rng(args.seed + 991)
    if len(samples) > buffer.capacity:
        pick = rng.choice(len(samples), size=buffer.capacity, replace=False)
        samples = [samples[int(i)] for i in np.sort(pick)]

    loader = make_loader(corpus, samples, vocab, args, train=True,
                         seed=args.seed + 991, batch=args.batch)
    model.eval()
    latents, targets = [], []
    with torch.no_grad():
        for images, state, ids, mask, target, _ in loader:
            latent = model.encode(images.to(device), state.to(device),
                                  dev(ids, device), dev(mask, device))
            latents.append(latent.cpu().numpy())
            targets.append(target.numpy())
    n = buffer.add(task, np.concatenate(latents), np.concatenate(targets),
                   rng, fingerprint=encoder_fingerprint(model))
    return n


def fill_raw_buffer(corpus, task, args, buffer):
    rng = np.random.default_rng(args.seed + 991)
    index = corpus.index[task]["train"]
    return buffer.add(task, index, rng)


# ---------------------------------------------------------------------------
# One stage
# ---------------------------------------------------------------------------

def train_stage(model, vocab, corpus, task, seen, args, buffer, device,
                epochs, ckpt_path, stage, joint=False):
    """Train on `task` alone, rehearsing whatever `buffer` holds.

    `seen` is every task learned so far INCLUDING this one; it decides what
    validation covers, because the thing being optimised in a continual
    setting is performance on everything learned, not on the newest task.
    """

    # joint=True is the multitask reference: every task in `seen` at once,
    # shuffled together, selected on the mean over all of them
    train_samples = ([s for name in seen for s in corpus.samples(name, "train")]
                     if joint else corpus.samples(task, "train"))
    grounding = args.aux_weight > 0 and model.has_grounding_head
    train_loader = make_loader(corpus, train_samples, vocab,
                               args, train=True, seed=args.seed + stage,
                               aux=grounding)
    val_loaders = {
        name: make_loader(corpus, corpus.samples(name, "val"), vocab, args,
                          train=False, seed=args.seed, workers=2)
        for name in seen}

    replay_loader = None
    if buffer is not None and buffer.kind == "raw" and len(buffer):
        picks = [(t, e, f) for t in buffer.task_names
                 for e, f in buffer.tasks[t]]
        replay_loader = cycling(make_loader(
            corpus, picks, vocab, args, train=True, seed=args.seed + 555,
            batch=args.replay_batch, workers=2, aux=grounding))

    trainable = (model.trunk_parameters() if model.encoders_frozen
                 else list(model.parameters()))
    optimiser = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=3, min_lr=1e-6)
    criterion = nn.MSELoss()
    rng = np.random.default_rng(args.seed + 100 * stage)

    replaying = (buffer is not None and len(buffer) > 0)
    print(f"    training {sum(p.numel() for p in trainable):,} of "
          f"{count_parameters(model):,} parameters"
          + (f", rehearsing {len(buffer):,} {buffer.kind} samples from "
             f"{', '.join(buffer.task_names)}" if replaying else ""))

    best, best_epoch, stale, history = float("inf"), -1, 0, []
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        running_new, running_old, running_aux, seen_n = 0.0, 0.0, 0.0, 0
        for batch in train_loader:
            images, state, ids, mask, target = batch[:5]
            images, state = images.to(device), state.to(device)
            ids, mask, target = (dev(ids, device), dev(mask, device),
                                 target.to(device))

            optimiser.zero_grad(set_to_none=True)
            if grounding:
                # the action loss plus "where is the object you were told
                # to pick", read from the same latent the policy acts on
                latent = model.encode(images, state, ids, mask)
                loss_new = criterion(model.head(latent), target)
                aux = batch[6].to(device)
                weight = aux[:, 2]
                error = ((model.target_xy(latent) - aux[:, :2]) ** 2).sum(1)
                loss_aux = (error * weight).sum() / weight.sum().clamp(min=1.0)
                loss = loss_new + args.aux_weight * loss_aux
                running_aux += loss_aux.item() * images.shape[0]
            else:
                loss_new = criterion(forward_batch(model, images, state, ids,
                                                   mask), target)
                loss = loss_new
            loss_old = torch.zeros((), device=device)

            if replaying and buffer.kind == "latent":
                lat, tgt, _ = buffer.sample(args.replay_batch, rng)
                loss_old = criterion(
                    model.head(torch.from_numpy(lat).to(device)),
                    torch.from_numpy(tgt).to(device))
                loss = loss + args.replay_weight * loss_old
            elif replaying and replay_loader is not None:
                rbatch = next(replay_loader)
                ri, rs, rids, rmask, rtgt = (dev(x, device) for x in rbatch[:5])
                if grounding:
                    # the grounding loss on REHEARSED frames too: this is
                    # where it contrasts instructions - an old frame, its old
                    # instruction, and where THAT object is. Within a stage
                    # every new sample names the same object, so the new
                    # task alone could be grounded from vision.
                    rlatent = model.encode(ri, rs, rids, rmask)
                    loss_old = criterion(model.head(rlatent), rtgt)
                    raux = rbatch[6].to(device)
                    rweight = raux[:, 2]
                    rerror = ((model.target_xy(rlatent) - raux[:, :2]) ** 2
                              ).sum(1)
                    loss_old_aux = ((rerror * rweight).sum()
                                    / rweight.sum().clamp(min=1.0))
                    loss = loss + args.aux_weight * loss_old_aux
                else:
                    loss_old = criterion(
                        forward_batch(model, ri, rs, rids, rmask), rtgt)
                loss = loss + args.replay_weight * loss_old

            loss.backward()
            nn.utils.clip_grad_norm_(trainable, 1.0)
            optimiser.step()

            running_new += loss_new.item() * images.shape[0]
            running_old += float(loss_old.detach()) * images.shape[0]
            seen_n += images.shape[0]

        val = {name: evaluate(model, loader, device)
               for name, loader in val_loaders.items()}
        joint = float(np.mean([v["mae_bar"] for v in val.values()]))
        # THE EPOCH IS CHOSEN ON THE CURRENT TASK'S VALIDATION SPLIT ONLY.
        # Selecting on the joint score would hand the no-buffer arm a
        # retention mechanism it is not supposed to have - early stopping the
        # moment the old task starts to degrade is itself a use of the old
        # task's data. Every arm may look at the task it is learning, so
        # that is what selects. The old tasks' scores are computed and
        # logged, and used for nothing.
        select = joint if joint else val[task]["mae_bar"]
        scheduler.step(select)
        history.append({"epoch": epoch,
                        "train_mse_new": running_new / max(seen_n, 1),
                        "train_mse_replay": running_old / max(seen_n, 1),
                        "val_mae_bar": {k: v["mae_bar"] for k, v in val.items()},
                        "selection_val_mae_bar": select,
                        "joint_val_mae_bar": joint,
                        "train_grounding_mse": (running_aux / max(seen_n, 1)
                                                if grounding else None)})

        line = "  ".join(f"{k} {v['mae_bar']:.5f}" for k, v in val.items())
        print(f"    epoch {epoch:03d}/{epochs}  new {running_new / max(seen_n, 1):.6f}"
              + (f"  replay {running_old / max(seen_n, 1):.6f}" if replaying
                 else "")
              + (f"  where {running_aux / max(seen_n, 1):.4f}" if grounding
                 else "")
              + f"  |  val {line}  (joint {joint:.5f}, not used)")

        # `--select last` keeps every epoch's weights and never stops early,
        # so every arm gets exactly the same number of passes over the new
        # task. See the note in main()'s docstring: with a validation metric
        # this flat, "best epoch" is close to a coin toss, and it lands on a
        # very different epoch for an arm with 4.7 M free parameters than for
        # one with 16.2 M - which confounds the comparison with how much each
        # arm was allowed to change.
        improved = select < best - 1e-9
        if improved or args.select == "last":
            best = min(best, select)
            best_epoch, stale = epoch, 0
            save_checkpoint(ckpt_path, model, vocab, extra={
                "stage": stage, "task": task, "seen": list(seen),
                "replay": args.replay, "epoch": epoch,
                "selection_val_mae_bar": select,
                "joint_val_mae_bar": joint,
                "val_mae_bar": {k: v["mae_bar"] for k, v in val.items()},
                "image_size": corpus.meta[task]["image_size"],
                "policy_dt_s": corpus.meta[task]["policy_dt_s"],
                "cameras": corpus.meta[task]["cameras"],
                "instructions": {n: tasks.instructions(n)
                                 for n in tasks.TASK_NAMES},
                "augmentation": ("photometric only" if args.augment
                                 else "none"),
                "pressure_labels": list(PRESSURE_LABELS),
                "replay_report": (buffer.report() if buffer is not None
                                  else None),
            })
        else:
            stale += 1
            if stale >= args.patience:
                print("    early stopping")
                break

    return {"select": args.select, "stage": stage, "task": task,
            "seen": list(seen),
            "epochs_run": len(history), "best_epoch": best_epoch,
            "best_selection_val_mae_bar": best,
            "selected_on": ("mean over every task's validation split" if joint
                            else f"{task} validation split only"),
            "history": history,
            "seconds": time.perf_counter() - started,
            "trainable_parameters": sum(p.numel() for p in trainable),
            "checkpoint": str(ckpt_path)}


# ---------------------------------------------------------------------------

def main(args):
    # the weight initialisation too, not only the split and sampling, so a
    # seed reproduces a run (as ../traj_track_exp/train.py does)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    order = tasks.stages(args.upto)
    label = args.label or args.replay
    arm_dir = CKPT_DIR / label
    arm_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 78)
    print(f"CONTINUAL PICK-AND-DROP — INCREMENTAL TRAINING, arm={label} "
          f"(replay={args.replay}, capacity={args.capacity})")
    print("=" * 78)

    vocab = Vocabulary.standard()
    corpus = Corpus(order, data_dir=HERE / args.data_dir, seed=args.seed,
                    train_stride=args.sample_stride,
                    eval_stride=args.eval_stride)

    print(f"  order      : {' -> '.join(order)}")
    print(f"  vocabulary : {len(vocab)} words, frozen before stage 1"
          + ("  (not used: MiniLM reads the raw sentence)"
             if args.arch in MINILM_ARCHS else ""))
    for name in order:
        s = corpus.splits[name]
        print(f"  {name:6s}     {len(s['train'])} train / {len(s['val'])} val"
              f" / {len(s['test'])} test episodes, "
              f"{len(corpus.index[name]['train']):,} train samples")
    print(f"  device     : {device}")

    # FiLM is off by default since 2026-10-03 (the ablation study: removing
    # it raised success 73 -> 83% and cut forgetting); --with-film restores
    # the earlier model
    if (args.arch in MINILM_ARCHS and not args.with_film
            and "film" not in args.ablate):
        args.ablate = ["film"] + list(args.ablate)
    args.ablate = [a for a in ABLATIONS if a in args.ablate]
    print(f"  removed    : {', '.join(args.ablate) or 'nothing (with FiLM)'}")
    model = ContinualVLA(len(vocab), arch=args.arch,
                         ablate=args.ablate).to(device)
    first = order[0]
    model.set_normalisation(
        np.concatenate([corpus.episodes[first][int(e)]["pressure"][:-1]
                        for e in corpus.splits[first]["train"]]),
        np.concatenate([np.diff(corpus.episodes[first][int(e)]["pressure"],
                                axis=0)
                        for e in corpus.splits[first]["train"]]))
    report = model.parameter_report()
    print(f"  parameters : {report['total']:,} "
          f"({report['encoders']:,} encoders + {report['trunk']:,} trunk)")

    buffer = make_buffer(args.replay if args.replay in ("latent", "raw")
                         else "none",
                         args.capacity, model.latent_dim, CHUNK_STEPS,
                         model.action_dim, RAW_SAMPLE_BYTES)
    freeze_after_first = args.replay in ("latent", "frozen")

    if args.joint:
        return train_joint(model, vocab, corpus, order, args, device,
                           arm_dir, label, report)

    shared = shared_stage1(args.arch, args.data_dir, args.ablate)
    stages_log = []
    for stage, task in enumerate(order, start=1):
        seen = order[:stage]
        ckpt = arm_dir / f"stage{stage}_{task}.pt"
        print()
        print(f"  STAGE {stage}: {task}   \"{tasks.instruction(task)}\"")

        if stage == 1 and shared.exists() and not args.fresh_stage1:
            print(f"    reusing {shared.name} (stage 1 is identical in "
                  f"every arm)")
            model, vocab, ck = load_checkpoint(shared, device)
            assert model.arch == args.arch, (
                f"{shared} is arch={model.arch}, run asked for {args.arch}")
            assert list(model.ablate) == list(args.ablate), (
                f"{shared} removed {model.ablate}, run asked for {args.ablate}")
            shutil.copy2(shared, ckpt)
            stages_log.append({"stage": 1, "task": task, "seen": seen,
                               "reused": str(shared),
                               "best_selection_val_mae_bar":
                                   ck.get("selection_val_mae_bar"),
                               "checkpoint": str(ckpt)})
        else:
            epochs = args.epochs if stage == 1 else args.epochs_increment
            log = train_stage(model, vocab, corpus, task, seen, args, buffer,
                              device, epochs, ckpt, stage)
            stages_log.append(log)
            model, vocab, _ = load_checkpoint(ckpt, device)
            if stage == 1:
                CKPT_DIR.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ckpt, shared)

        # remember this task before moving on
        if buffer is not None and stage < len(order):
            if freeze_after_first:
                model.freeze_encoders(True)
            if buffer.kind == "latent":
                n = fill_latent_buffer(model, corpus, task, vocab, args,
                                       buffer, device)
            else:
                n = fill_raw_buffer(corpus, task, args, buffer)
            print(f"    buffer  +{n:,} {buffer.kind} samples from {task}"
                  f"   ({buffer.nbytes() / 1e6:.1f} MB total)")
        elif freeze_after_first and stage < len(order):
            model.freeze_encoders(True)

        if freeze_after_first and stage == 1:
            print("    encoders frozen from here on")

    # -- final offline table over every task, from every stage checkpoint ----
    print()
    print("  offline MAE of each stage's checkpoint on every task's test "
          "split:")
    table = {}
    for stage, task in enumerate(order, start=1):
        net, voc, _ = load_checkpoint(arm_dir / f"stage{stage}_{task}.pt",
                                      device)
        row = {}
        for name in order:
            loader = make_loader(corpus, corpus.samples(name, "test"), voc,
                                 args, train=False, seed=args.seed, workers=2)
            row[name] = evaluate(net, loader, device)["mae_bar"]
        table[f"stage{stage}_{task}"] = row
        print(f"    after {task:6s} " + "  ".join(
            f"{n} {row[n]:.5f}" for n in order))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {"replay": args.replay,
           "order": order,
           "arm": label,
           "encoders_frozen_after_stage1": freeze_after_first,
           "replay_report": buffer.report() if buffer is not None else None,
           "settings": {k: getattr(args, k) for k in (
               "epochs", "epochs_increment", "batch", "replay_batch",
               "replay_weight", "capacity", "lr", "arch", "augment",
               "sample_stride", "eval_stride", "state_noise", "select",
               "data_dir", "seed", "aux_weight", "ablate", "with_film")},
           "parameters": report,
           "stages": stages_log,
           "offline_test_mae_bar": table,
           "caveat": "offline MAE ranks policies badly on this task; see "
                     "deploy.py"}
    (RESULTS_DIR / f"train_{label}.json").write_text(
        json.dumps(out, indent=2, default=float))

    print()
    print("=" * 78)
    print(f"  checkpoints in {arm_dir}")
    print(f"  log          {RESULTS_DIR / f'train_{label}.json'}")
    print("=" * 78)


def train_joint(model, vocab, corpus, order, args, device, arm_dir, label,
                report):
    """The multitask reference: one stage, every task at once. Not a
    continual arm - it answers whether the architecture can follow the
    instruction at all when nothing is ever forgotten."""

    ckpt = arm_dir / "joint.pt"
    print()
    print(f"  JOINT: {', '.join(order)} together, {args.epochs} epochs")
    log = train_stage(model, vocab, corpus, order[-1], order, args, None,
                      device, args.epochs, ckpt, 1, joint=True)
    net, voc, _ = load_checkpoint(ckpt, device)
    row = {}
    for name in order:
        loader = make_loader(corpus, corpus.samples(name, "test"), voc,
                             args, train=False, seed=args.seed, workers=2)
        row[name] = evaluate(net, loader, device)["mae_bar"]
    print("  offline test MAE  " + "  ".join(f"{n} {row[n]:.5f}"
                                             for n in order))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"train_{label}.json").write_text(json.dumps(
        {"joint": True, "order": order, "arm": label,
         "settings": {k: getattr(args, k) for k in (
             "epochs", "batch", "lr", "arch", "augment", "sample_stride",
             "eval_stride", "state_noise", "select", "data_dir", "seed",
             "aux_weight")},
         "parameters": report, "stages": [log],
         "offline_test_mae_bar": {"joint": row}}, indent=2, default=float))
    print(f"  checkpoint   {ckpt}")
    print(f"  log          {RESULTS_DIR / f'train_{label}.json'}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Incremental training of the language-conditioned VLA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--replay", choices=("none", "frozen", "latent", "raw"),
                   default="latent")
    p.add_argument("--epochs", type=int, default=45,
                   help="Epochs for stage 1 (the whole network).")
    p.add_argument("--epochs-increment", type=int, default=20,
                   help="Epochs for each later stage.")
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--select", choices=("best", "last"), default="last",
                   help="Which epoch's weights to keep. 'last' also disables "
                        "early stopping, so every arm gets the same number of "
                        "passes over the new task - which matters here "
                        "because the validation metric is too flat to pick "
                        "an epoch meaningfully.")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--replay-batch", type=int, default=32,
                   help="Rehearsed samples per step, the same for every arm "
                        "that rehearses. Half of --batch, so a step is two "
                        "parts new task to one part old.")
    p.add_argument("--replay-weight", type=float, default=1.0)
    p.add_argument("--capacity", type=int, default=2000,
                   help="Samples remembered per finished task.")
    p.add_argument("--data-dir", default="dataset",
                   help="Demonstration folder: 'dataset' (one object on the "
                        "table) or 'dataset_clutter' (all three).")
    p.add_argument("--label", default=None,
                   help="Name for this arm's checkpoint folder and log "
                        "(default: the --replay value). Lets the same "
                        "strategy be run twice at different capacities.")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--arch", default="resnet18")
    p.add_argument("--augment", dest="augment", action="store_true",
                   default=True)
    p.add_argument("--no-augment", dest="augment", action="store_false")
    p.add_argument("--aug-scale", type=float, default=1.0)
    p.add_argument("--sample-stride", type=int, default=3,
                   help="Use every Nth frame as a training sample; "
                        "neighbouring frames are near-duplicates.")
    p.add_argument("--eval-stride", type=int, default=4,
                   help="Same, for the validation and test splits.")
    p.add_argument("--state-noise", type=float, default=0.01)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--upto", type=int, default=None,
                   help="Stop after this many tasks (debugging).")
    p.add_argument("--fresh-stage1", action="store_true",
                   help="Retrain stage 1 instead of reusing the shared "
                        "checkpoint.")
    p.add_argument("--aux-weight", type=float, default=1e-4,
                   help="Weight of the grounding loss (predict the named "
                        "object's start xy); only arch minilm_aux has the "
                        "head. The action MSE is ~3e-5, the grounding MSE "
                        "starts ~1, so 1e-4 makes it dominant early and "
                        "comparable once it is learned.")
    p.add_argument("--joint", action="store_true",
                   help="Multitask reference: train on every task at once "
                        "(one stage, --epochs epochs) instead of in order.")
    p.add_argument("--ablate", nargs="*", default=[], choices=ABLATIONS,
                   help="Modules to switch off (ablation study): attention, "
                        "transformer, state (film is already off unless "
                        "--with-film) - see vla_model.ABLATIONS.")
    p.add_argument("--with-film", action="store_true",
                   help="Keep FiLM, the model before 2026-10-03. By default "
                        "the MiniLM archs are built without it.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
