#!/usr/bin/env python3
"""
Seeded teach-and-repeat evaluation of a trained tracking VLA, saving
everything needed for later analysis and plotting. Nothing is trained.

    python eval_seeds/eval_seeds.py                     # clean + seeds 0 1 2
    python eval_seeds/eval_seeds.py --conditions seed0  # one condition

WHAT A SEED MEANS HERE
----------------------
The original evaluation (../deploy.py) is fully deterministic: fixed taught
goal recordings, fixed start pose, a deterministic policy - three seeds
would give three identical runs. A seed therefore draws the two noise
sources the training demonstrations themselves contain (generate_demos.py):

    start jitter   the arm's balanced start pressure p0 += U(-0.05, 0.05) bar
    disturbance    each policy step, N(0, 0.004 bar) added to the 6 arm
                   pressures; the disturbed pressure is what is applied AND
                   what the policy sees as "previous command" next step,
                   exactly as in the demonstrations

`clean` runs without either and reproduces ../deploy.py (1.54 mm).

The closed loop is ../deploy.run_circle's, re-implemented here only to (a)
inject the seeded noise and (b) record every policy call. The original
scripts are imported, never modified.

WHAT IS SAVED  (results/<condition>/<shape>.npz, one per rollout)
------------------------------------------------------------------
per policy step (T = steps of the taught recording, 20 Hz):
    time_s [T]            reference [T,3] m      tcp [T,3] m (tool point)
    error_mm [T]          on_shape [T] bool      expert_tcp [T,3] m
    command [T,7] bar     the pressures applied (after rate limit, clip, noise)
    target [T,7] bar      the policy's chunk entry for this step (before limits)
    disturbance [T,6] bar the injected arm noise (zeros when clean)
    expert_pressure [T,7] the expert's pressures on the taught recording
per policy call (N ~ T/5):
    call_step [N]         step index at which the policy was queried
    call_state [N,7]      the previous-command input it saw
    call_delta [N,10,7]   predicted pressure CHANGES (the network output)
    call_chunk [N,10,7]   predicted absolute pressures = state + cumsum(delta)
    call_frames [N,4]     frame index behind each image slot (<=0: own past,
                          >0: taught goal frame index)
    call_ms [N]           policy latency
plus scalars: p0, seed, radius_m, centre_m, frame_offsets.

results/<condition>/summary.json   per-shape metrics for that condition
results/all_runs.csv               one row per (condition, shape)
results/aggregate.json             per shape: mean +- std over the 3 seeds
results/aggregate.png              the same, plotted
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))

import deploy as D  # noqa: E402  (unmodified; also sets up sys.path)
import mujoco  # noqa: E402

C, P = D.C, D.P
RESULTS = HERE / "results"
START_JITTER = 0.05        # bar, generate_demos.py --start-jitter default
DISTURB = 0.004            # bar, generate_demos.py --disturb default
LAG_RANGE = 20             # policy steps searched either way for the lag


def rollout(net, ckpt, device, spec, args, seed):
    rng = np.random.default_rng(seed) if seed is not None else None
    name = spec["name"]
    goal_frames = np.load(D.GOAL_DIR / f"{name}.npy", mmap_mode="r")
    taught = np.load(D.GOAL_DIR / f"{name}.npz")
    reference, on_shape = taught["reference"], taught["on_shape"]
    n_steps = len(goal_frames)
    offsets = list(ckpt["frame_offsets"])
    n_past = int(ckpt["n_past"])

    p0 = args.p0 + (rng.uniform(-START_JITTER, START_JITTER) if rng else 0.0)
    scene = D.TrackScene(args.control_dt)
    arm = D.balanced_pressure(p0)
    scene.reset(arm)
    model, data = scene.model, scene.data
    D.hide_link_capsules(model)
    D.style_scene(model)
    cameras, _ = D.policy_cameras(scene, model)
    image_size = int(ckpt.get("image_size", args.image))
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, image_size)
    model.vis.global_.offheight = max(model.vis.global_.offheight, image_size)
    renderer = mujoco.Renderer(model, height=image_size, width=image_size)

    every = max(1, int(round(C.POLICY_DT / scene.control_dt)))
    command = np.r_[arm, scene.grip].astype(np.float32)
    history = deque(maxlen=max(1, (n_past - 1) * C.PAST_STRIDE + 1))
    step_limit = np.r_[[args.max_rate * C.POLICY_DT] * 6,
                       args.grip_rate * C.POLICY_DT]

    rec = {k: [] for k in ("tcp", "command", "target", "disturbance")}
    calls = {k: [] for k in ("step", "state", "delta", "chunk", "frames", "ms")}
    chunk, chunk_at = None, 0
    started = time.perf_counter()

    for k in range(n_steps):
        views = []
        for cam in cameras:
            renderer.update_scene(data, camera=cam)
            views.append(renderer.render().copy())
        frame = np.stack(views)
        if not history:
            history.extend([frame] * history.maxlen)
        history.append(frame)

        if chunk is None or chunk_at >= min(D.CHUNK_EXECUTE, len(chunk)):
            picks, idx = [], []
            for o in offsets:
                if o <= 0:
                    picks.append(history[max(0, len(history) - 1 + o)])
                    idx.append(max(0, k + o))
                else:
                    g = min(k + o, n_steps - 1)
                    picks.append(np.asarray(goal_frames[g]))
                    idx.append(g)
            images = torch.from_numpy(D.stack_frames(picks)).unsqueeze(0).to(device)
            state = torch.from_numpy(command).unsqueeze(0).to(device)
            tick = time.perf_counter()
            with torch.no_grad():
                delta = net(images, state)
                out = state.unsqueeze(1) + torch.cumsum(delta, dim=1)
            ms = 1000 * (time.perf_counter() - tick)
            chunk, chunk_at = out.cpu().numpy()[0], 0
            calls["step"].append(k)
            calls["state"].append(command.copy())
            calls["delta"].append(delta.cpu().numpy()[0])
            calls["chunk"].append(chunk.copy())
            calls["frames"].append(idx)
            calls["ms"].append(ms)

        target = chunk[chunk_at].copy()
        chunk_at += 1
        command = (command + np.clip(target - command, -step_limit,
                                     step_limit)).astype(np.float32)
        command[:6] = np.clip(command[:6], args.arm_p_min, D.P_MAX)
        noise = (rng.normal(0.0, DISTURB, 6).astype(np.float32) if rng
                 else np.zeros(6, np.float32))
        command[:6] = np.clip(command[:6] + noise, args.arm_p_min, D.P_MAX)
        command[6] = float(scene.grip)                 # gripper stays closed

        scene.apply(command[:6])
        for _ in range(every):
            scene.step()

        rec["tcp"].append(scene.tcp().copy())
        rec["command"].append(command.copy())
        rec["target"].append(target)
        rec["disturbance"].append(noise)

    tcp = np.asarray(rec["tcp"])
    err = np.linalg.norm(tcp - reference, axis=1) * 1000
    on = np.asarray(on_shape, dtype=bool)

    # lag: the time shift that best aligns the path with the reference,
    # and the error left once it is removed (pure shape error)
    best = (0, float(err[on].mean()))
    for s in range(-LAG_RANGE, LAG_RANGE + 1):
        ref_s = reference[np.clip(np.arange(n_steps) - s, 0, n_steps - 1)]
        e = np.linalg.norm(tcp - ref_s, axis=1)[on].mean() * 1000
        if e < best[1]:
            best = (s, float(e))

    arrays = dict(
        time_s=np.arange(n_steps) * C.POLICY_DT, reference=reference, tcp=tcp,
        error_mm=err, on_shape=on, expert_tcp=taught["tcp"],
        expert_pressure=taught["pressure"],
        command=np.asarray(rec["command"]), target=np.asarray(rec["target"]),
        disturbance=np.asarray(rec["disturbance"]),
        call_step=np.asarray(calls["step"]),
        call_state=np.asarray(calls["state"]),
        call_delta=np.asarray(calls["delta"]),
        call_chunk=np.asarray(calls["chunk"]),
        call_frames=np.asarray(calls["frames"]),
        call_ms=np.asarray(calls["ms"]),
        p0=np.float32(p0), seed=np.int64(-1 if seed is None else seed),
        radius_m=np.float32(spec["radius"]),
        centre_m=np.asarray(spec["centre"], np.float32),
        frame_offsets=np.asarray(offsets))
    metrics = {
        "name": name, "radius_m": spec["radius"],
        "centre_m": list(spec["centre"]), "steps": n_steps,
        "p0_bar": float(p0),
        "mean_mm": float(err[on].mean()),
        "rmse_mm": float(np.sqrt((err[on] ** 2).mean())),
        "median_mm": float(np.median(err[on])),
        "p95_mm": float(np.percentile(err[on], 95)),
        "max_mm": float(err[on].max()),
        "final_mm": float(err[-1]),
        "error_pct_of_radius": float(100 * err[on].mean()
                                     / (spec["radius"] * 1000)),
        "lag_s": best[0] * C.POLICY_DT,
        "mean_mm_lag_removed": best[1],
        "expert_mean_mm": float(np.linalg.norm(
            taught["tcp"] - reference, axis=1)[on].mean() * 1000),
        "policy_calls": len(calls["step"]),
        "policy_ms_per_call": float(np.mean(calls["ms"])),
        "wall_clock_s": round(time.perf_counter() - started, 1),
    }
    return arrays, metrics


def run_condition(cond, args):
    seed = None if cond == "clean" else int(cond.replace("seed", ""))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, ckpt = D.load_checkpoint(Path(args.ckpt), device)
    manifest = json.loads((D.GOAL_DIR / "manifest.json").read_text())
    taught = {c["name"] for c in manifest["circles"]}
    specs = [s for s in P.EVAL_SHAPES if s["name"] in taught]
    out = RESULTS / cond
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, spec in enumerate(specs):
        # a different, reproducible stream per shape within a seed
        s = None if seed is None else seed * 1000 + i
        arrays, m = rollout(net, ckpt, device, spec, args, s)
        np.savez_compressed(out / f"{spec['name']}.npz", **arrays)
        m.update(condition=cond, seed=seed, shape_seed=s)
        rows.append(m)
        print(f"  [{cond}] {m['name']:12s} mean {m['mean_mm']:5.2f} mm  "
              f"p95 {m['p95_mm']:5.2f}  max {m['max_mm']:5.2f}  "
              f"lag {m['lag_s']:+.2f} s  (lag-removed {m['mean_mm_lag_removed']:.2f})",
              flush=True)
    summary = {
        "condition": cond, "seed": seed,
        "noise": ({"start_jitter_bar": START_JITTER, "disturb_bar": DISTURB}
                  if seed is not None else "none"),
        "checkpoint": str(Path(args.ckpt).resolve()),
        "conditioning": ckpt["conditioning"],
        "conditioning_label": ckpt["conditioning_label"],
        "frame_offsets": list(ckpt["frame_offsets"]),
        "settings": {k: getattr(args, k) for k in
                     ("p0", "arm_p_min", "max_rate", "grip_rate", "control_dt")},
        "mean_mm": float(np.mean([r["mean_mm"] for r in rows])),
        "shapes": rows,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def aggregate(conds):
    summaries = {c: json.loads((RESULTS / c / "summary.json").read_text())
                 for c in conds if (RESULTS / c / "summary.json").exists()}
    keys = ("mean_mm", "rmse_mm", "median_mm", "p95_mm", "max_mm", "final_mm",
            "error_pct_of_radius", "lag_s", "mean_mm_lag_removed")
    with open(RESULTS / "all_runs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "seed", "shape", "p0_bar", *keys,
                    "expert_mean_mm", "policy_ms_per_call"])
        for c, s in summaries.items():
            for r in s["shapes"]:
                w.writerow([c, s["seed"], r["name"], f"{r['p0_bar']:.4f}",
                            *(f"{r[k]:.4f}" for k in keys),
                            f"{r['expert_mean_mm']:.4f}",
                            f"{r['policy_ms_per_call']:.2f}"])
    seeds = [c for c in summaries if c != "clean"]
    shapes = [r["name"] for r in next(iter(summaries.values()))["shapes"]]
    agg = {"checkpoint": Path(next(iter(summaries.values()))["checkpoint"]).stem,
           "seeds": seeds, "noise": {"start_jitter_bar": START_JITTER,
                                     "disturb_bar": DISTURB}, "shapes": {}}
    for sh in shapes:
        row = {}
        for k in keys:
            v = [next(r[k] for r in summaries[c]["shapes"] if r["name"] == sh)
                 for c in seeds]
            row[k] = {"mean": float(np.mean(v)), "std": float(np.std(v)),
                      "per_seed": v}
        if "clean" in summaries:
            row["clean_mean_mm"] = next(r["mean_mm"] for r in
                                        summaries["clean"]["shapes"]
                                        if r["name"] == sh)
        agg["shapes"][sh] = row
    per_seed_means = [summaries[c]["mean_mm"] for c in seeds]
    agg["overall"] = {"mean_mm": float(np.mean(per_seed_means)),
                      "std_mm": float(np.std(per_seed_means)),
                      "per_seed_mean_mm": per_seed_means,
                      "clean_mean_mm": summaries.get("clean", {}).get("mean_mm")}
    (RESULTS / "aggregate.json").write_text(json.dumps(agg, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.arange(len(shapes))
    fig, ax = plt.subplots(figsize=(10, 4.2))
    m = [agg["shapes"][s]["mean_mm"]["mean"] for s in shapes]
    e = [agg["shapes"][s]["mean_mm"]["std"] for s in shapes]
    ax.bar(x - 0.2, m, 0.4, yerr=e, capsize=4, color="#2f6fb0",
           edgecolor="black", lw=0.5, label=f"seeded noise, mean ± std over {len(seeds)} seeds")
    if "clean" in summaries:
        ax.bar(x + 0.2, [agg["shapes"][s]["clean_mean_mm"] for s in shapes],
               0.4, color="#9ecae1", edgecolor="black", lw=0.5, label="clean")
    ax.set_xticks(x); ax.set_xticklabels(shapes)
    ax.set_ylabel("mean tracking error [mm]"); ax.grid(axis="y", alpha=0.3)
    ax.set_title(f"{agg.get('checkpoint', 'model')} — seeded {agg['overall']['mean_mm']:.2f} "
                 f"± {agg['overall']['std_mm']:.2f} mm, clean "
                 f"{agg['overall']['clean_mean_mm'] or float('nan'):.2f} mm")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(RESULTS / "aggregate.png", dpi=130)
    plt.close(fig)
    return agg


if __name__ == "__main__":
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--conditions", nargs="+",
                   default=["clean", "seed0", "seed1", "seed2"])
    p.add_argument("--aggregate-only", action="store_true")
    p.add_argument("--results-dir", default=None,
                   help="Where to write (default: eval_seeds/results).")
    extra, rest = p.parse_known_args()
    if extra.results_dir:
        RESULTS = Path(extra.results_dir)
    args = D.parse_args(rest)
    args.ckpt = args.ckpt or str(D.CKPT_DIR / f"track_vla_{args.conditioning}.pt")
    if not extra.aggregate_only:
        for cond in extra.conditions:
            run_condition(cond, args)
    # aggregate only when this call covers every condition (or is asked to):
    # conditions run as separate parallel processes must not read each
    # other's half-written summaries
    if not (extra.aggregate_only or
            set(extra.conditions) >= {"clean", "seed0", "seed1", "seed2"}):
        sys.exit(0)
    agg = aggregate(["clean", "seed0", "seed1", "seed2"])
    o = agg["overall"]
    print(f"\n  seeds: {o['mean_mm']:.2f} ± {o['std_mm']:.2f} mm "
          f"(per seed {', '.join(f'{v:.2f}' for v in o['per_seed_mean_mm'])}); "
          f"clean {o['clean_mean_mm']}")
