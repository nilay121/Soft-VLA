#!/usr/bin/env python3
"""
The final test: all three objects on the table, one instruction at a time.

    python combined_deploy.py --ckpt checkpoints/none/stage3_can.pt
    python combined_deploy.py --ckpt ... --episodes 10 --videos 2

This is the scenario the whole experiment was building towards. The cube, the
apple and the can are all on the table at once, at random reachable positions.
The policy is given one instruction — "pick up the red apple and drop it into
the bowl" — and a time budget; then the next instruction, on whatever is left;
then the third. The scene persists between them, so an object already
delivered stays delivered and the table genuinely empties.

WHY THIS IS THE ONLY TEST THAT CAN MEASURE THE LANGUAGE
-------------------------------------------------------
In the single-object evaluation the instruction is redundant: whatever is on
the table is the thing to pick, so a policy that ignores the sentence scores
exactly the same as one that reads it (`deploy.py --instruct` measured a
5-point difference, inside the noise). Here three objects are within reach and
only the sentence says which one is wanted, so the outcome splits three ways:

    correct     the named object ended up in the bowl
    WRONG       a different object ended up in the bowl  <- read the sentence,
                                                            and got it wrong
    none        nothing was delivered in the time budget

The wrong-object rate is the number that matters. A policy that cannot read
the instruction does not fail by freezing; it fails by confidently fetching
whichever object it happens to be looking at.

A NOTE ON WHAT IS BEING ASKED OF THE POLICY
-------------------------------------------
Every stage of training saw exactly ONE object on the table. This test puts
three. That is a real distribution shift on top of the language question, and
the two are not separable from this test alone: a failure here can be "cannot
read the instruction" or "has never seen clutter". `--only` runs the same
layouts with the other objects parked, which isolates the clutter half.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SIMULATOR = HERE.parent.parent / "simulator"
for path in (str(SIMULATOR), str(HERE)):
    while path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)

import tasks  # noqa: E402
from bowl_cameras import policy_cameras  # noqa: E402
from continual_scene import (  # noqa: E402
    ContinualScene,
    balanced_pressure,
    sample_multi_layout,
)
from sim.mujoco_env import P_MAX  # noqa: E402
from vla_model import (  # noqa: E402
    CHUNK_EXECUTE,
    FRAME_HISTORY,
    FRAME_STRIDE,
    POLICY_DT,
    PRESSURE_LABELS,
    MINILM_ARCHS,
    load_checkpoint,
    stack_frames,
)
from viz.common import (  # noqa: E402
    COLOR_TIP,
    COLOR_TRACE,
    VideoRecorder,
    add_geom,
    add_polyline,
    free_camera,
    hide_link_capsules,
    style_scene,
    write_json,
)

# this folder, or CONTINUAL_OUT (relative = to this folder), as train_continual.py
RESULTS_DIR = HERE / os.environ.get("CONTINUAL_OUT", ".") / "results"

# words a user might use for each object -> task
OBJECT_WORDS = {"cube": ("cube", "block", "box"),
                "apple": ("apple",),
                "can": ("can", "cylinder", "tin")}


def parse_command(command):
    """A multi-step instruction -> [(task, step sentence)], in order.

    "Pick the cube and place it in the bowl, then pick the can ... and
    finally pick the apple ..." is split at "then" / "finally" / ";", each
    step is matched to the object it names (synonyms above), and rewritten
    in the words the policy's vocabulary knows:
    "pick up the <colour> <object> and place it in the bowl".
    """

    import re

    steps = [x for x in re.split(r"\b(?:and\s+)?(?:then|finally|afterwards|"
                                 r"next)\b|;", command.lower()) if x.strip()]
    plan = []
    for step in steps:
        found = [task for task, words in OBJECT_WORDS.items()
                 if any(re.search(rf"\b{w}s?\b", step) for w in words)]
        if len(found) != 1:
            raise SystemExit(f"cannot tell which object this step names: "
                             f"{step.strip()!r} (found {found})")
        task = found[0]
        plan.append((task, f"pick up the {tasks.phrase(task)} and place it "
                           f"in the bowl"))
    return plan


def run_episode(net, vocab, device, args, seed, video_path=None):
    rng = np.random.default_rng(seed)
    names = list(tasks.TASK_NAMES)
    objects = [tasks.BY_NAME[n]["object"] for n in names]
    layout, bowl_xy = sample_multi_layout(rng, objects)

    scene = ContinualScene(args.control_dt)
    if args.only:
        # the clutter control: same positions, other objects parked
        keep = tasks.BY_NAME[args.only]["object"]
        layout = {keep: layout[keep]}
    scene.set_multi_layout(layout, bowl_xy, active=next(iter(layout)))
    arm = balanced_pressure(args.p0)
    scene.reset(arm, 0.0)

    model, data = scene.model, scene.data
    hide_link_capsules(model)
    style_scene(model)

    cameras, _ = policy_cameras(scene, model)
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.image,
                                     args.width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, args.image,
                                      args.height)
    renderer = mujoco.Renderer(model, height=args.image, width=args.image)

    view = free_camera(model, lookat=[0.02, 0.0, scene.table_top + 0.06],
                       distance=args.cam_distance, azimuth=args.cam_azimuth,
                       elevation=args.cam_elevation)
    video = VideoRecorder(model, video_path or HERE / "unused.mp4", args.width,
                          args.height, args.fps, view, shell_alpha=1.0,
                          hide_links=True, enabled=video_path is not None)

    wanted = [args.only] if args.only else list(args.tasks or names)
    order = [wanted[i] for i in rng.permutation(len(wanted))]
    if args.plan:
        order = [task for task, _ in args.plan]      # the user's order

    every = max(1, int(round(POLICY_DT / scene.control_dt)))
    steps = int(round(args.segment / POLICY_DT))
    render_every = max(1, int(round((1.0 / args.fps) / POLICY_DT)))

    command = np.r_[arm, 0.0].astype(np.float32)
    history = deque(maxlen=(FRAME_HISTORY - 1) * FRAME_STRIDE + 1)
    on_table = set(layout)
    segments, trace = [], []
    started = time.perf_counter()

    for task in order:
        target = tasks.BY_NAME[task]["object"]
        text = (dict(args.plan)[task] if args.plan
                else args.instruction_map[task] if args.instruction_map
                else tasks.instruction(task, args.template))
        if getattr(net, "arch", "") in MINILM_ARCHS:
            # the raw sentence goes to MiniLM's own vocabulary, not through
            # the project's 23-word one, so unseen words are understood
            ids, mask = [text], None
        else:
            ids, mask = vocab.encode_batch([text], device=device)
        before = scene.objects_in_bowl(on_table)
        chunk, chunk_at = None, 0
        # how far the target has been pushed from where it started, by the
        # arm's earlier instructions, before this one begins
        moved_mm = float(np.linalg.norm(
            scene.object_pos_named(target)[:2] - np.asarray(layout[target]))
            * 1000)

        for k in range(steps):
            views = []
            for cam in cameras:
                renderer.update_scene(data, camera=cam)
                views.append(renderer.render().copy())
            frame = np.stack(views)
            if not history:
                history.extend([frame] * history.maxlen)
            history.append(frame)

            if chunk is None or chunk_at >= min(CHUNK_EXECUTE, len(chunk)):
                picks = [history[max(0, len(history) - 1 - j * FRAME_STRIDE)]
                         for j in range(FRAME_HISTORY - 1, -1, -1)]
                images = torch.from_numpy(stack_frames(picks)
                                          ).unsqueeze(0).to(device)
                with torch.no_grad():
                    out = net.pressure_chunk(
                        images,
                        torch.from_numpy(command).unsqueeze(0).to(device),
                        ids, mask)
                chunk, chunk_at = out.cpu().numpy()[0], 0

            step_target = chunk[chunk_at]
            chunk_at += 1
            step_limit = np.r_[[args.max_rate * POLICY_DT] * 6,
                               args.grip_rate * POLICY_DT]
            command = (command + np.clip(step_target - command, -step_limit,
                                         step_limit)).astype(np.float32)
            command[:6] = np.clip(command[:6], args.arm_p_min, P_MAX)
            command[6] = np.clip(command[6], 0.0, args.grip_max)

            scene.apply(command[:6], command[6])
            for _ in range(every):
                scene.step()
            trace.append(scene.tcp().copy())

            if video.enabled and k % render_every == 0:
                now = scene.objects_in_bowl(on_table)
                delivered = [n for n in on_table
                             if now[n] and not before[n]]
                good = target in delivered
                wrong = [n for n in delivered if n != target]

                def decorate(scn, xy=layout, target=target):
                    add_polyline(scn, trace[-400:], 0.0018, COLOR_TRACE)
                    for n in on_table:
                        add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE,
                                 [0.005] * 3, scene.object_pos_named(n),
                                 COLOR_TIP if n == target
                                 else (0.6, 0.6, 0.6, 0.8))

                video.capture(data, decorate, hud=[
                    (f"ALL THREE OBJECTS ON THE TABLE — \"{text}\"",
                     (255, 255, 255)),
                    (f"target: the {tasks.phrase(task)}   |   also present: "
                     + ", ".join(sorted(n for n in on_table if n != target)),
                     (200, 210, 255)),
                    (f"t = {k * POLICY_DT:5.2f} s / {args.segment:.0f} s"
                     f"     fingers {scene.finger_contacts()}",
                     (150, 230, 150)),
                    (f"arm  [{' '.join(f'{v:.2f}' for v in command[:6])}] bar"
                     f"   grip {command[6]:.2f}", (200, 200, 200)),
                    ("DELIVERED: " + (f"{target}  CORRECT" if good else
                                      (f"{wrong[0]}  WRONG OBJECT" if wrong
                                       else "-")),
                     (120, 230, 255) if good else
                     ((255, 140, 120) if wrong else (220, 220, 220))),
                ])

        after = scene.objects_in_bowl(on_table)
        delivered = [n for n in on_table if after[n] and not before[n]]
        outcome = ("correct" if target in delivered
                   else ("wrong" if delivered else "none"))
        segments.append({
            "task": task, "instruction": text, "target": target,
            "delivered": delivered, "outcome": outcome,
            "wrong_object": next((n for n in delivered if n != target), None),
            "target_moved_mm": round(moved_mm, 1),
            "objects_in_bowl_at_start": sorted(n for n, v in before.items()
                                               if v),
        })
        on_table -= set(delivered)

        if args.empty_bowl and task != order[-1]:
            # --empty-bowl: lift everything delivered so far out of the bowl
            # and park it off camera, so the next instruction sees an empty
            # bowl - as every demonstration does
            for n in scene.objects_in_bowl(list(layout)):
                if not scene.in_bowl(n):
                    continue
                o = scene.objects[n]
                jnt = model.body_jntadr[o["body"]]
                qa, da = model.jnt_qposadr[jnt], model.jnt_dofadr[jnt]
                data.qpos[qa:qa + 3] = [*o["parking"], o["rest"] + 1e-4]
                data.qpos[qa + 3:qa + 7] = [1, 0, 0, 0]
                data.qvel[da:da + 6] = 0.0
            mujoco.mj_forward(model, data)

        if args.rehome > 0 and task != order[-1]:
            # --rehome: drive the arm back to the start pose with the
            # gripper open, at the policy's own rate limits, before the
            # next instruction - every demonstration starts from there.
            # The scene is NOT reset: delivered objects stay in the bowl.
            home = np.r_[arm, 0.0].astype(np.float32)
            step_limit = np.r_[[args.max_rate * POLICY_DT] * 6,
                               args.grip_rate * POLICY_DT]
            for _ in range(int(round(args.rehome / POLICY_DT))):
                command = (command + np.clip(home - command, -step_limit,
                                             step_limit)).astype(np.float32)
                scene.apply(command[:6], command[6])
                for _ in range(every):
                    scene.step()
                trace.append(scene.tcp().copy())
            history.clear()        # the next instruction starts fresh too

    video.close()
    return {"seed": int(seed),
            "layout_mm": {k: (v * 1000).round(1).tolist()
                          for k, v in layout.items()},
            "order": order, "segments": segments,
            "correct": sum(s["outcome"] == "correct" for s in segments),
            "wrong": sum(s["outcome"] == "wrong" for s in segments),
            "none": sum(s["outcome"] == "none" for s in segments),
            "wall_clock_s": round(time.perf_counter() - started, 1)}


def main(args):
    print()
    print("=" * 78)
    print("COMBINED ENVIRONMENT — all three objects, one instruction at a time")
    print("=" * 78)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, vocab, ckpt = load_checkpoint(args.ckpt, device)
    args.plan = parse_command(args.command) if args.command else None
    if args.plan and args.trained_wording:
        # same steps and order, but each in the exact training sentence
        args.plan = [(t, tasks.instruction(t, args.template))
                     for t, _ in args.plan]
    args.instruction_map = None
    if args.instructions:
        src = Path(args.instructions)
        args.instruction_map = json.loads(src.read_text() if src.exists()
                                          else args.instructions)
        print("  instructions (custom):")
        for task, text in args.instruction_map.items():
            print(f"    {task:6s} \"{text}\"")
    if args.plan:
        from language import tokenize
        unknown = sorted({w for _, text in args.plan for w in tokenize(text)
                          if w not in vocab.index})
        if unknown and net.arch not in MINILM_ARCHS:
            raise SystemExit(f"words outside the policy's vocabulary: "
                             f"{unknown}")
        print(f"  command    : {args.command!r}")
        for k, (task, text) in enumerate(args.plan, 1):
            print(f"    step {k}  {task:6s} \"{text}\"")
    args.image = int(ckpt.get("image_size", args.image))

    print(f"  checkpoint : {Path(args.ckpt).parent.name}/"
          f"{Path(args.ckpt).name}  (stage {ckpt.get('stage')}, "
          f"replay {ckpt.get('replay')})")
    print(f"  on the table: " + (args.only or ", ".join(tasks.TASK_NAMES)))
    print(f"  instructions: "
          + ", ".join(args.tasks or tasks.TASK_NAMES))
    print(f"  episodes   : {args.episodes}, {args.segment:.0f} s per "
          f"instruction, "
          + (f"re-homed {args.rehome:.0f} s" if args.rehome > 0 else "no re-homing")
          + (", bowl emptied" if args.empty_bowl else ", bowl kept") + ", "
          + ("in the command's order" if args.plan
             else "order shuffled per episode"))
    print()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = args.tag or (f"combined_{Path(args.ckpt).parent.name}"
                       f"_{Path(args.ckpt).stem}"
                       + (f"_only_{args.only}" if args.only else ""))
    episodes = []
    for i in range(args.episodes):
        video = (RESULTS_DIR / "videos" / f"{tag}_ep{i}.mp4"
                 if i < args.videos else None)
        if video is not None:
            video.parent.mkdir(parents=True, exist_ok=True)
        info = run_episode(net, vocab, device, args, args.seed + i, video)
        episodes.append(info)
        line = "  ".join(f"{s['task']}:{s['outcome']}"
                         + (f"({s['wrong_object']})" if s["wrong_object"]
                            else "")
                         for s in info["segments"])
        print(f"  episode {i:2d}   {line}")

    n = sum(len(e["segments"]) for e in episodes)
    correct = sum(e["correct"] for e in episodes)
    wrong = sum(e["wrong"] for e in episodes)
    none = sum(e["none"] for e in episodes)

    per_task = {}
    for name in tasks.TASK_NAMES:
        got = [s for e in episodes for s in e["segments"] if s["task"] == name]
        if got:
            per_task[name] = {
                "instructions": len(got),
                "correct": sum(s["outcome"] == "correct" for s in got),
                "wrong": sum(s["outcome"] == "wrong" for s in got),
                "none": sum(s["outcome"] == "none" for s in got)}

    results = {
        "test": "all three objects on the table, one instruction at a time",
        "checkpoint": str(args.ckpt),
        "only": args.only,
        "instructions_given": n,
        "summary": {"correct": correct, "wrong_object": wrong,
                    "nothing": none,
                    "correct_rate": correct / max(n, 1),
                    "wrong_object_rate": wrong / max(n, 1)},
        "per_task": per_task,
        "settings": {k: getattr(args, k) for k in
                     ("episodes", "segment", "seed", "template", "only",
                      "tasks")},
        "episodes": episodes,
    }
    write_json(RESULTS_DIR / f"{tag}.json", results)

    print()
    print("=" * 78)
    print(f"  {n} instructions:  correct {correct} ({correct / n:.0%})   "
          f"WRONG OBJECT {wrong} ({wrong / n:.0%})   nothing {none} "
          f"({none / n:.0%})")
    for name, row in per_task.items():
        print(f"    {name:6s} correct {row['correct']}/{row['instructions']}"
              f"   wrong {row['wrong']}   none {row['none']}")
    print(f"  -> {RESULTS_DIR / (tag + '.json')}")
    print("=" * 78)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Final test: three objects, instructions one at a time.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--tag", default=None)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--segment", type=float, default=22.0,
                   help="Seconds allowed per instruction.")
    p.add_argument("--only", default=None, choices=tasks.TASK_NAMES,
                   help="Clutter control: same layouts, but park the other "
                        "objects and give only this instruction.")
    p.add_argument("--tasks", nargs="+", default=None,
                   choices=tasks.TASK_NAMES,
                   help="Which instructions to give (default: all three). "
                        "All three objects stay on the table either way, so "
                        "`--tasks cube` on a stage-1 checkpoint asks it for "
                        "the only thing it knows, among objects it does not.")
    p.add_argument("--template", type=int, default=0)
    p.add_argument("--videos", type=int, default=0)
    p.add_argument("--command", default=None,
                   help="One multi-step instruction, e.g. \"pick the cube and "
                        "place it in the bowl, then the can, and finally the "
                        "apple\". Split into steps run in that order.")
    p.add_argument("--instructions", default=None,
                   help="The sentence to give for each task, as JSON (or a "
                        "JSON file): '{\"cube\": \"grab the blue block\", ...}'. "
                        "MiniLM models read it with MiniLM's full vocabulary.")
    p.add_argument("--trained-wording", action="store_true",
                   help="With --command: phrase each step exactly as training "
                        "template --template does, instead of the command's "
                        "'place it in the bowl' form.")
    # THE STANDARD PROTOCOL: between instructions the arm goes back home and
    # the delivered objects are taken out of the bowl, so every instruction
    # starts from the state every demonstration starts from.
    p.add_argument("--empty-bowl", dest="empty_bowl", action="store_true",
                   default=True,
                   help="Between instructions, park every delivered object "
                        "off camera so the bowl starts each instruction "
                        "empty (default).")
    p.add_argument("--keep-bowl", dest="empty_bowl", action="store_false",
                   help="Leave delivered objects in the bowl (old protocol).")
    p.add_argument("--rehome", type=float, default=3.0,
                   help="Seconds spent driving the arm back to the start "
                        "pose (gripper open) between instructions (default "
                        "3). 0 = the next instruction starts wherever the "
                        "last one ended.")
    p.add_argument("--seed", type=int, default=5000)

    p.add_argument("--p0", type=float, default=0.4)
    p.add_argument("--arm-p-min", type=float, default=0.25)
    p.add_argument("--grip-max", type=float, default=1.0)
    p.add_argument("--max-rate", type=float, default=1.2)
    p.add_argument("--grip-rate", type=float, default=2.0)
    p.add_argument("--control-dt", type=float, default=0.01)
    p.add_argument("--image", type=int, default=112)

    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--cam-distance", type=float, default=1.05)
    p.add_argument("--cam-azimuth", type=float, default=115.0)
    p.add_argument("--cam-elevation", type=float, default=-18.0)
    return p.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
