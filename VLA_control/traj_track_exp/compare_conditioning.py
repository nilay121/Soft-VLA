#!/usr/bin/env python3
"""
Does tracking need past frames, future frames, or both?

    python compare_conditioning.py              # collect + plot (no training)
    python compare_conditioning.py --run        # train and deploy all three

The question this experiment exists to answer. Three policies, identical in
every way except which frames go into the stack - same architecture, same
four frames per camera, same parameter count, same demonstrations, same
seeds:

    past_only     4 past,  0 future    what pick-and-place used
    future_only   1 past,  3 future    the goal, but no velocity
    both          2 past,  2 future

`--run` does the whole thing:

    train.py --conditioning X      for each X
    deploy.py --conditioning X     for each X, on the taught circles

and then this script collects `results/track_<X>.json` into
results/conditioning_comparison.{json,png}.

Read the closed-loop millimetres, not the offline loss. On the
pick-and-place task a policy that achieved nothing scored a better
validation error than one that worked.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SIMULATOR = HERE.parent.parent / "simulator"
for path in (str(SIMULATOR), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

import conditioning as C  # noqa: E402

RESULTS = HERE / "results"
CKPTS = HERE / "checkpoints"


def run_all(args):
    for name in C.VARIANTS:
        for script, extra in (("train.py", ["--epochs", str(args.epochs)]),
                              ("deploy.py", ["--videos", str(args.videos)])):
            cmd = [sys.executable, str(HERE / script),
                   "--conditioning", name] + extra
            print(f"\n$ {' '.join(cmd[1:])}\n", flush=True)
            subprocess.run(cmd, check=True)


def collect():
    rows = []
    for name, spec in C.VARIANTS.items():
        deploy = RESULTS / f"track_{name}.json"
        report = CKPTS / f"track_vla_{name}_report.json"
        if not deploy.exists():
            raise SystemExit(f"missing {deploy} - run with --run first")
        d = json.loads(deploy.read_text())
        r = json.loads(report.read_text()) if report.exists() else {}
        rows.append({
            "conditioning": name, "label": spec["label"],
            "n_past": spec["n_past"], "n_future": spec["n_future"],
            "frame_offsets": d["frame_offsets"],
            "mean_mm": d["summary"]["mean_mm"],
            "expert_mean_mm": d["summary"]["expert_mean_mm"],
            "per_circle": {c["name"]: round(c["mean_mm"], 2)
                           for c in d["circles"]},
            "worst_circle_mm": max(c["mean_mm"] for c in d["circles"]),
            "error_pct_of_radius": float(np.mean(
                [c["error_pct_of_radius"] for c in d["circles"]])),
            "offline_mae_bar": r.get("test", {}).get("mae_bar"),
            "offline_fraction_of_baseline":
                r.get("test", {}).get("fraction_of_baseline"),
            "best_epoch": r.get("best_epoch"),
        })
    return rows


def plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"{r['conditioning']}\n{r['n_past']}p+{r['n_future']}f"
              for r in rows]
    x = np.arange(len(rows))
    expert = rows[0]["expert_mean_mm"]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))

    # colour by outcome, not by index, so adding a variant cannot run the
    # palette out
    worst = max(r["mean_mm"] for r in rows)
    colours = ["#d9534f" if r["mean_mm"] > 0.25 * worst else "#2f9e44"
               for r in rows]
    bars = axes[0].bar(x, [r["mean_mm"] for r in rows], 0.55, color=colours,
                       edgecolor="black", lw=0.5)
    axes[0].axhline(expert, ls="--", lw=1.2, color="black",
                    label=f"expert {expert:.2f} mm")
    axes[0].set_yscale("log")
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, fontsize=8)
    axes[0].set_ylabel("mean tracking error [mm, log]")
    axes[0].set_title("Closed loop — the answer")
    axes[0].legend(fontsize=8); axes[0].grid(axis="y", alpha=0.3)
    for b, r in zip(bars, rows):
        axes[0].text(b.get_x() + b.get_width() / 2, b.get_height(),
                     f"{r['mean_mm']:.1f}", ha="center", va="bottom",
                     fontsize=8)

    circles = list(rows[0]["per_circle"])
    width = 0.8 / len(rows)
    for i, r in enumerate(rows):
        axes[1].bar(np.arange(len(circles)) + i * width - 0.4 + width / 2,
                    [r["per_circle"][c] for c in circles], width,
                    label=r["conditioning"], edgecolor="black", lw=0.4)
    axes[1].set_yscale("log")
    axes[1].set_xticks(range(len(circles)))
    axes[1].set_xticklabels(circles, fontsize=7, rotation=15)
    axes[1].set_ylabel("mean error [mm, log]")
    axes[1].set_title("Per circle"); axes[1].legend(fontsize=7)
    axes[1].grid(axis="y", alpha=0.3)

    frac = [100 * (r["offline_fraction_of_baseline"] or 0) for r in rows]
    axes[2].bar(x, frac, 0.55, color="#8e5bb5", edgecolor="black", lw=0.5)
    axes[2].axhline(100, ls="--", lw=1.0, color="black",
                    label="predicting no change")
    axes[2].set_xticks(x); axes[2].set_xticklabels(labels, fontsize=8)
    axes[2].set_ylabel("offline test error [% of baseline]")
    axes[2].set_title("Offline — barely separates them")
    axes[2].legend(fontsize=8); axes[2].grid(axis="y", alpha=0.3)

    fig.suptitle("Trajectory tracking: what the policy must be shown "
                 "(identical network, 4 frames per camera in all three)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(args):
    if args.run:
        run_all(args)

    rows = collect()
    RESULTS.mkdir(parents=True, exist_ok=True)
    payload = {
        "question": "does trajectory tracking need past frames, future "
                    "(goal) frames, or both?",
        "controlled": "same architecture, same 4 frames per camera, same "
                      "parameter count, same demonstrations and seeds; only "
                      "the meaning of the frames differs",
        "expert_mean_mm": rows[0]["expert_mean_mm"],
        "variants": rows,
    }
    (RESULTS / "conditioning_comparison.json").write_text(
        json.dumps(payload, indent=2))
    plot(rows, RESULTS / "conditioning_comparison.png")

    print()
    print("=" * 78)
    print("CONDITIONING COMPARISON")
    print("=" * 78)
    print(f"  {'variant':14s} {'frames':10s} {'mean':>9s} {'worst':>9s} "
          f"{'% of r':>8s} {'offline':>9s}")
    for r in rows:
        print(f"  {r['conditioning']:14s} "
              f"{r['n_past']}p+{r['n_future']}f{'':4s} "
              f"{r['mean_mm']:7.2f}mm {r['worst_circle_mm']:7.2f}mm "
              f"{r['error_pct_of_radius']:7.1f}% "
              f"{100 * (r['offline_fraction_of_baseline'] or 0):7.0f}%")
    print(f"  {'expert':14s} {'-':10s} {rows[0]['expert_mean_mm']:7.2f}mm")
    print()
    print(f"  outputs in {RESULTS}")
    print("=" * 78)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Compare past / future / both conditioning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--run", action="store_true",
                   help="Train and deploy all three variants first.")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--videos", type=int, default=1)
    return p.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
