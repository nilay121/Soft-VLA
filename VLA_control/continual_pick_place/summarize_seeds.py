#!/usr/bin/env python3
"""
Combine the evaluations of several training seeds of the continual
pick-and-place model into one table (used by run_seeds.sh).

    python summarize_seeds.py <run_dir> 0 1 2

Reads  <run_dir>/results/s<S>_{stage1,stage2,final}_L<layout>.json and
       s<S>_alone_<object>.json (combined_deploy.py outputs), and
       <run_dir>/results/train_seed<S>.json (train_continual.py log)
Writes <run_dir>/results/seeds_summary.{json,csv,png}

Everything is under the standard protocol (arm re-homed, bowl emptied
between instructions), three layout sets per model.
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

run = Path(sys.argv[1])
seeds = [int(s) for s in sys.argv[2:]] or [0, 1, 2]
res = run / "results"
LAYOUTS = (5000, 6000, 7000)
OBJECTS = ("cube", "apple", "can")


def tally(paths):
    t = {"correct": 0, "wrong": 0, "none": 0, "n": 0, "whole": 0, "episodes": 0,
         "per_object": defaultdict(lambda: [0, 0]),
         "per_position": defaultdict(lambda: [0, 0])}
    for p in paths:
        for e in json.loads(Path(p).read_text())["episodes"]:
            t["episodes"] += 1
            t["whole"] += all(s["outcome"] == "correct" for s in e["segments"])
            for i, s in enumerate(e["segments"]):
                t[s["outcome"]] += 1
                t["n"] += 1
                ok = s["outcome"] == "correct"
                t["per_object"][s["target"]][0] += ok
                t["per_object"][s["target"]][1] += 1
                t["per_position"][i + 1][0] += ok
                t["per_position"][i + 1][1] += 1
    t["rate"] = t["correct"] / max(t["n"], 1)
    t["per_object"] = {k: v for k, v in t["per_object"].items()}
    t["per_position"] = {str(k): v for k, v in sorted(t["per_position"].items())}
    return t


models = {}
for s in seeds:
    f = lambda kind: [res / f"s{s}_{kind}_L{L}.json" for L in LAYOUTS]
    train = json.loads((res / f"train_seed{s}.json").read_text())
    models[s] = {
        "stage1": tally(f("stage1")),
        "stage2": tally(f("stage2")),
        "final": tally(f("final")),
        "alone": {o: json.loads((res / f"s{s}_alone_{o}.json").read_text())
                  ["summary"]["correct_rate"] for o in OBJECTS},
        "offline_test_mae_bar": train["offline_test_mae_bar"],
        "training_minutes": sum(st.get("seconds") or 0
                                for st in train["stages"]) / 60,
    }


def ms(values):
    return {"mean": float(np.mean(values)), "std": float(np.std(values)),
            "per_seed": [float(v) for v in values]}


summary = {"training_seeds": seeds, "layout_sets": list(LAYOUTS),
           "per_model": models, "across_seeds": {}}
A = summary["across_seeds"]
for kind in ("stage1", "stage2", "final"):
    A[f"{kind}_rate"] = ms([m[kind]["rate"] for m in models.values()])
    A[f"{kind}_wrong"] = ms([m[kind]["wrong"] for m in models.values()])
A["final_whole_episode_rate"] = ms([m["final"]["whole"] / m["final"]["episodes"]
                                    for m in models.values()])
for o in OBJECTS:
    A[f"final_{o}_rate"] = ms([m["final"]["per_object"][o][0]
                               / m["final"]["per_object"][o][1]
                               for m in models.values()])
    A[f"alone_{o}_rate"] = ms([m["alone"][o] for m in models.values()])
for pos in ("1", "2", "3"):
    A[f"final_position{pos}_rate"] = ms(
        [m["final"]["per_position"][pos][0] / m["final"]["per_position"][pos][1]
         for m in models.values()])
(res / "seeds_summary.json").write_text(json.dumps(summary, indent=2))

with open(res / "seeds_summary.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["metric", "mean", "std", *(f"seed{s}" for s in seeds)])
    for k, v in A.items():
        w.writerow([k, f"{v['mean']:.4f}", f"{v['std']:.4f}",
                    *(f"{x:.4f}" for x in v["per_seed"])])

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
keys = ["stage1_rate", "stage2_rate", "final_rate", "final_cube_rate",
        "final_apple_rate", "final_can_rate", "final_whole_episode_rate"]
labels = ["stage 1\ncube", "stage 2\ncube+apple", "final\nall three",
          "final\ncube", "final\napple", "final\ncan", "final\nwhole episode"]
fig, ax = plt.subplots(figsize=(11, 4.4))
x = np.arange(len(keys))
ax.bar(x, [100 * A[k]["mean"] for k in keys], 0.6,
       yerr=[100 * A[k]["std"] for k in keys], capsize=4, color="#b0307f",
       edgecolor="black", lw=0.5)
for i, k in enumerate(keys):
    for j, v in enumerate(A[k]["per_seed"]):
        ax.plot(i - 0.15 + 0.15 * j, 100 * v, "o", color="black", ms=3)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("success [%]"); ax.set_ylim(0, 105); ax.grid(axis="y", alpha=0.3)
ax.set_title(f"continual pick-and-place (MiniLM + position loss, raw replay "
             f"2 000), {len(seeds)} training seeds — final "
             f"{100 * A['final_rate']['mean']:.0f} ± {100 * A['final_rate']['std']:.0f}%")
fig.tight_layout(); fig.savefig(res / "seeds_summary.png", dpi=130)

print(f"\n  {'':28s} {'mean ± std':>14s}   per seed")
for k, v in A.items():
    scale = 1 if "wrong" in k else 100
    unit = "" if "wrong" in k else "%"
    print(f"  {k:28s} {scale * v['mean']:6.1f} ± {scale * v['std']:4.1f}{unit:1s}   "
          + "  ".join(f"{scale * x:.1f}" for x in v["per_seed"]))
for s, m in models.items():
    print(f"  seed {s}: final {m['final']['correct']}/{m['final']['n']} "
          f"(wrong {m['final']['wrong']}), stage 2 {m['stage2']['correct']}/"
          f"{m['stage2']['n']}, stage 1 {m['stage1']['correct']}/{m['stage1']['n']}, "
          f"alone {100 * np.mean(list(m['alone'].values())):.0f}%, {m['training_minutes']:.0f} min training")
print(f"  -> {res / 'seeds_summary.json'}")
