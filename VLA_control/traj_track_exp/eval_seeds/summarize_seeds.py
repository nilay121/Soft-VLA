#!/usr/bin/env python3
"""
Combine the evaluations of several TRAINING seeds into one table.

    python summarize_seeds.py <run_dir> 0 1 2

Reads  <run_dir>/results/train_seed<s>/{clean,seed0,seed1,seed2}/summary.json
       <run_dir>/checkpoints/track_vla_future_only_seed<s>_report.json
Writes <run_dir>/results/training_seeds_summary.{json,csv,png}

For every shape: clean error and noisy error (mean of the 3 noise seeds),
each as mean +- std over the training seeds; plus each model's offline
test MAE and best epoch.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

run = Path(sys.argv[1])
seeds = [int(s) for s in sys.argv[2:]] or [0, 1, 2]
res = run / "results"
NOISE = ["seed0", "seed1", "seed2"]

models = {}
for s in seeds:
    d = res / f"train_seed{s}"
    clean = json.loads((d / "clean" / "summary.json").read_text())
    noisy = [json.loads((d / c / "summary.json").read_text()) for c in NOISE
             if (d / c / "summary.json").exists()]
    rep = json.loads((run / "checkpoints" /
                      f"track_vla_future_only_seed{s}_report.json").read_text())
    models[s] = {
        "clean": {r["name"]: r["mean_mm"] for r in clean["shapes"]},
        "noisy": {r["name"]: float(np.mean([next(x["mean_mm"] for x in n["shapes"]
                                                 if x["name"] == r["name"])
                                            for n in noisy]))
                  for r in clean["shapes"]},
        "best_epoch": rep["best_epoch"],
        "test_mae_bar": rep["test"]["mae_bar"],
        "test_fraction_of_baseline": rep["test"]["fraction_of_baseline"],
        "training_minutes": rep["training_seconds"] / 60,
    }

shapes = list(next(iter(models.values()))["clean"])
out = {"training_seeds": seeds, "per_model": models, "shapes": {}}
rows = []
for sh in shapes + ["ALL"]:
    entry = {}
    for kind in ("clean", "noisy"):
        per = [np.mean(list(m[kind].values())) if sh == "ALL" else m[kind][sh]
               for m in models.values()]
        entry[kind] = {"mean": float(np.mean(per)), "std": float(np.std(per)),
                       "per_training_seed": [float(v) for v in per]}
    out["shapes"][sh] = entry
    rows.append([sh, *(f"{entry[k]['mean']:.3f}" for k in ("clean", "noisy")),
                 *(f"{entry[k]['std']:.3f}" for k in ("clean", "noisy")),
                 *(f"{v:.3f}" for v in entry["clean"]["per_training_seed"]),
                 *(f"{v:.3f}" for v in entry["noisy"]["per_training_seed"])])

(res / "training_seeds_summary.json").write_text(json.dumps(out, indent=2))
with open(res / "training_seeds_summary.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["shape", "clean_mean_mm", "noisy_mean_mm", "clean_std_mm",
                "noisy_std_mm", *(f"clean_train_seed{s}" for s in seeds),
                *(f"noisy_train_seed{s}" for s in seeds)])
    w.writerows(rows)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
x = np.arange(len(shapes) + 1)
labels = shapes + ["all six"]
fig, ax = plt.subplots(figsize=(11, 4.4))
for off, kind, col in ((-0.2, "clean", "#9ecae1"), (0.2, "noisy", "#2f6fb0")):
    ax.bar(x + off, [out["shapes"][s][kind]["mean"]
                     for s in shapes + ["ALL"]], 0.4,
           yerr=[out["shapes"][s][kind]["std"] for s in shapes + ["ALL"]],
           capsize=4, color=col, edgecolor="black", lw=0.5,
           label=f"{kind} (mean ± std over {len(seeds)} training seeds)")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("mean tracking error [mm]"); ax.grid(axis="y", alpha=0.3)
ax.legend(fontsize=8)
a = out["shapes"]["ALL"]
ax.set_title(f"track_vla_future_only, {len(seeds)} training seeds — clean "
             f"{a['clean']['mean']:.2f} ± {a['clean']['std']:.2f} mm, noisy "
             f"{a['noisy']['mean']:.2f} ± {a['noisy']['std']:.2f} mm")
fig.tight_layout(); fig.savefig(res / "training_seeds_summary.png", dpi=130)

print(f"\n  {'shape':12s} {'clean mm':>16s} {'noisy mm':>16s}")
for sh in shapes + ["ALL"]:
    e = out["shapes"][sh]
    print(f"  {sh:12s} {e['clean']['mean']:8.2f} ± {e['clean']['std']:.2f}"
          f"   {e['noisy']['mean']:8.2f} ± {e['noisy']['std']:.2f}")
for s, m in models.items():
    print(f"  training seed {s}: best epoch {m['best_epoch']}, test MAE "
          f"{m['test_mae_bar']:.5f} bar ({100 * m['test_fraction_of_baseline']:.0f}% "
          f"of do-nothing), {m['training_minutes']:.0f} min")
print(f"  -> {res / 'training_seeds_summary.json'}")
