#!/usr/bin/env python3
"""
Record pick-and-drop demonstrations for every continual task.

    python generate_demos.py                       # all three tasks
    python generate_demos.py --task can --episodes 150
    python generate_demos.py --workers 12

Output: dataset/<task>/
    episode_XXX.npy   frames [T, 2, 112, 112, 3] uint8, memory-mapped at
                      train time
    episode_XXX.npz   pressure [T, 7] and the layout
    metadata.json     cameras, rates, the task's instructions, per-episode
                      outcome

The demonstrator is `heuristic.py`, which is ../pick_place_exp's controller
on the five-object scene, driven through its `on_step` hook. The loop is not
copied, so a demonstration here is by construction the same behaviour that
file measures.

EVERY TASK GETS ITS OWN RANDOM LAYOUTS
--------------------------------------
Each task draws from its own seed block (`--seed + 10000 * task index + i`),
so episode i of the apple task starts the apple somewhere different from
where episode i of the cube task started the cube. An earlier version shared
the seeds across tasks, to control for layout difficulty; the cost was that
all three objects sat in exactly the same places, which is both unrealistic
and makes the three tasks differ by appearance alone.

The start is drawn per object anyway now, because the objects are no longer
the same size: a 44 mm apple centred where a 22 mm can is legal would hang
over the edge of the table's hole. `continual_scene.sample_object_xy`
rejects positions that do not leave the object's own footprint on the table.

Recorded per policy step (20 Hz): two 112x112 RGB views - the external one
`camera_study.py` chose and the wrist one - and the 7-line pressure command.
Nothing else. Not the tool point, not the object position, not the bowl. The
instruction is not recorded because it is a property of the task and is
regenerated at train time from `tasks.py`, which lets the trainer draw a
different paraphrase every time it sees the same frame.

Only successful demonstrations are kept: cloning a failed drop teaches the
policy to fail.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

HERE = Path(__file__).resolve().parent
SIMULATOR = HERE.parent.parent / "simulator"
for path in (str(SIMULATOR), str(HERE)):
    while path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)

import tasks  # noqa: E402

DATA_DIR = HERE / "dataset"
# the five-object set gets its own folder, so the original stays intact
CLUTTER_DIR = HERE / (("dataset_clutter" if tasks.TASK_NAMES == ["cube", "apple", "can"]
                       else "dataset_clutter_" + "_".join(tasks.TASK_NAMES[3:]))
                      + ("_spread" if os.environ.get("CONTINUAL_LAYOUT") == "spread"
                         else ""))


def run_episode(job):
    """One demonstration. Returns (info dict)."""

    index, seed, task, args = job

    import mujoco

    import heuristic as Hx                      # rebinds the scene class
    from bowl_cameras import policy_cameras
    from continual_scene import sample_layout, sample_multi_layout
    from sim.mujoco_env import P_MAX
    from viz.common import hide_link_capsules, style_scene
    import vla_model as V

    rng = np.random.default_rng(seed)
    target = tasks.BY_NAME[task]["object"]

    if args.clutter:
        # every object on the table, every episode; the instruction is the
        # only thing that says which one to fetch
        names = [tasks.BY_NAME[n]["object"] for n in tasks.TASK_NAMES]
        layout, bowl_xy = sample_multi_layout(rng, names)
        object_xy = layout.pop(target)
        Hx.use_clutter(layout)
    else:
        object_xy, bowl_xy = sample_layout(rng, target)
        layout = {}
        Hx.use_clutter({})

    hargs = Hx.expert_args(task)
    hargs.object_xy, hargs.bowl_xy = object_xy, bowl_xy
    hargs.p0 = float(hargs.p0 + rng.uniform(-args.start_jitter,
                                            args.start_jitter))

    state = {"renderer": None, "cameras": None, "cfg": None, "every": 1}
    frames, pressures = [], []

    def on_step(step, scene, pressure, grip, t):
        if state["renderer"] is None:
            model = scene.model
            hide_link_capsules(model)
            style_scene(model)
            model.vis.global_.offwidth = max(model.vis.global_.offwidth,
                                             args.image)
            model.vis.global_.offheight = max(model.vis.global_.offheight,
                                              args.image)
            cameras, cfg = policy_cameras(scene, model)
            state.update(renderer=mujoco.Renderer(model, height=args.image,
                                                  width=args.image),
                         cameras=cameras, cfg=cfg,
                         every=max(1, int(round(V.POLICY_DT
                                                / scene.control_dt))))

        if step % state["every"]:
            return None

        # knock the arm off the ideal path once per policy step; the expert
        # corrects and the policy learns the correction
        if args.disturb:
            pressure = np.clip(
                pressure + rng.normal(0.0, args.disturb, pressure.shape),
                hargs.arm_p_min, P_MAX)

        views = []
        for cam in state["cameras"]:
            state["renderer"].update_scene(scene.data, camera=cam)
            views.append(state["renderer"].render().copy())
        frames.append(np.stack(views))
        pressures.append(np.r_[pressure, grip].astype(np.float32))

        return pressure if args.disturb else None

    summary = Hx.run(hargs, on_step=on_step)

    info = {"index": index, "seed": int(seed), "task": task,
            "object": hargs.object,
            "distractors_mm": {k: (v * 1000).round(1).tolist()
                               for k, v in layout.items()},
            "object_xy_mm": (object_xy * 1000).round(1).tolist(),
            "bowl_xy_mm": (bowl_xy * 1000).round(1).tolist(),
            "success": bool(summary["success"]),
            "picked": bool(summary["picked"]),
            "in_bowl": bool(summary["in_bowl"]),
            "bowl_offset_mm": round(summary["bowl_offset_mm"], 1),
            "frames": len(frames), "camera": state["cfg"]}

    if summary["success"]:
        out = (CLUTTER_DIR if args.clutter else DATA_DIR) / task
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / f"episode_{index:03d}.npy",
                np.asarray(frames, dtype=np.uint8))
        np.savez(out / f"episode_{index:03d}.npz",
                 pressure=np.asarray(pressures, dtype=np.float32),
                 object_xy=object_xy.astype(np.float32),
                 bowl_xy=bowl_xy.astype(np.float32))

    return info


def generate_task(task, args):
    out = (CLUTTER_DIR if args.clutter else DATA_DIR) / task
    out.mkdir(parents=True, exist_ok=True)
    for pattern in ("episode_*.npz", "episode_*.npy"):
        for old in out.glob(pattern):
            old.unlink()

    print()
    print("-" * 74)
    print(f"  task {tasks.TASK_NAMES.index(task) + 1}/{len(tasks.TASK_NAMES)}"
          f"  {task}   \"{tasks.instruction(task)}\"")
    print("-" * 74)

    started = time.perf_counter()
    # its OWN seed block per task - see the module docstring
    base = args.seed + 10000 * tasks.TASK_NAMES.index(task)
    jobs = [(i, base + i, task, args) for i in range(args.episodes)]
    with ProcessPoolExecutor(args.workers,
                             mp_context=mp.get_context("spawn")) as ex:
        infos = list(ex.map(run_episode, jobs))

    kept = [i for i in infos if i["success"]]
    frames = sum(i["frames"] for i in kept)

    import build_bowl_scene as BS
    import build_continual_scene as CS

    metadata = {
        "task": task,
        "goal": f"pick the {tasks.phrase(task)} up and drop it into the bowl",
        "clutter": bool(args.clutter),
        "objects_on_table": ([tasks.BY_NAME[n]["object"]
                              for n in tasks.TASK_NAMES] if args.clutter
                             else [tasks.BY_NAME[task]["object"]]),
        "instructions": tasks.instructions(task),
        "object": tasks.BY_NAME[task]["object"],
        "scene": CS.OUTPUT.name,
        "demonstrator": "heuristic.py = ../pick_place_exp/heuristic_pick_drop"
                        ".py on the five-object scene, driven through its "
                        "on_step hook",
        "episodes_run": len(infos),
        "episodes_kept": len(kept),
        "frames_kept": frames,
        "policy_dt_s": 0.05,
        "image_size": args.image,
        "cameras": infos[0]["camera"],
        "num_views": 2,
        "disturb_bar": args.disturb,
        "start_jitter_bar": args.start_jitter,
        "layout_seeds": [base, base + args.episodes - 1],
        "object_radius_range_m": list(BS.OBJECT_RADIUS_RANGE),
        "object_angle_range_deg": list(BS.OBJECT_ANGLE_RANGE_DEG),
        "bowl_centre_m": list(BS.HOLE_CENTRE),
        "bowl_jitter_m": BS.BOWL_JITTER,
        "bowl": {"rim_z_m": BS.BOWL_RIM_Z, "floor_z_m": BS.BOWL_FLOOR_Z,
                 "mouth_radius_m": BS.BOWL_R_RIM, "depth_m": BS.BOWL_DEPTH},
        "inputs": "camera frames + previous 7 pressures + the instruction "
                  "(no tip/object/bowl position)",
        "action": "next 7 pressures [A1 A2 A3 B1 B2 B3 grip], bar",
        "success_rule": "object below the bowl rim and inside the mouth "
                        "radius",
        "generation_seconds": time.perf_counter() - started,
        "episodes": infos,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))

    failed = [i for i in infos if not i["success"]]
    picked = sum(i["picked"] for i in failed)
    print(f"  kept {len(kept)}/{len(infos)}, {frames:,} frames "
          f"({frames * 2 * args.image ** 2 * 3 / 1e9:.2f} GB)"
          + (f"   discarded {len(failed)}: {picked} picked but missed, "
             f"{len(failed) - picked} never picked" if failed else ""))
    print(f"  {(time.perf_counter() - started) / 60:.1f} min -> {out}")
    return metadata


def main(args):
    which = tasks.TASK_NAMES if args.task == "all" else [args.task]

    print()
    print("=" * 74)
    print("CONTINUAL PICK-AND-DROP — DEMONSTRATIONS")
    print("=" * 74)
    print(f"  tasks     : {', '.join(which)}")
    print(f"  episodes  : {args.episodes} per task, its own seed block each "
          f"(base {args.seed} + 10000 x task index)")
    print(f"  recording : 2 views of {args.image}x{args.image} RGB + "
          f"7 pressures at 20 Hz")
    print(f"  workers   : {args.workers}")

    started = time.perf_counter()
    summaries = {t: generate_task(t, args) for t in which}

    root = CLUTTER_DIR if args.clutter else DATA_DIR
    root.mkdir(parents=True, exist_ok=True)
    index = {
        "clutter": bool(args.clutter),
        "tasks": tasks.TASK_NAMES,
        "learning_order": tasks.TASK_NAMES,
        "episodes_per_task": args.episodes,
        "layout_seeds_shared": False,
        "generated": {t: {"episodes_kept": m["episodes_kept"],
                          "frames_kept": m["frames_kept"],
                          "instructions": m["instructions"]}
                      for t, m in summaries.items()},
        "total_minutes": (time.perf_counter() - started) / 60,
    }
    path = root / "index.json"
    if path.exists() and args.task != "all":
        old = json.loads(path.read_text())
        old["generated"].update(index["generated"])
        index = old
    path.write_text(json.dumps(index, indent=2))

    print()
    print("=" * 74)
    for t, m in summaries.items():
        print(f"  {t:6s} {m['episodes_kept']:4d} episodes  "
              f"{m['frames_kept']:7,} frames")
    print(f"  wall clock {(time.perf_counter() - started) / 60:.1f} min")
    print("=" * 74)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Demonstrations for the continual pick-and-drop tasks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--task", default="all",
                   choices=("all", *tasks.TASK_NAMES))
    p.add_argument("--clutter", action="store_true",
                   help="Put ALL THREE objects on the table in every episode "
                        "and let the instruction pick the target. Writes to "
                        "dataset_clutter/ instead of dataset/.")
    p.add_argument("--episodes", type=int, default=150)
    p.add_argument("--disturb", type=float, default=0.004,
                   help="Std of the pressure noise injected once per policy "
                        "step [bar]; the expert's corrections become recovery "
                        "demonstrations.")
    p.add_argument("--start-jitter", type=float, default=0.05)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--image", type=int, default=112)
    p.add_argument("--seed", type=int, default=0,
                   help="First layout seed. Deployment uses 1000+, so these "
                        "are disjoint.")
    return p.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
