#!/usr/bin/env python3
"""
The expert, reused: heuristic_pick_drop.py (this folder) on the
five-object scene.

    python heuristic.py --task can
    python heuristic.py --task apple --no-video

Not a copy. The pick-and-drop controller - min-jerk tool-point waypoints
tracked by the 3x6 Jacobian with damped least-norm, gripper line ramped - is
imported and its scene class is rebound to `ContinualScene`. Everything it
does is already measured in that file, and keeping one copy means the
demonstrations for this experiment cannot silently drift from the ones the
pick-and-place policy learned from.

The controller reads the object's size out of the scene (`scene.obj["rest"]`
and `["top"]`), so the can needed no per-object tuning - the same reason
`--object apple|banana|ball` already worked there.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

HERE = Path(__file__).resolve().parent
SIMULATOR = HERE.parent.parent / "simulator"

def _fix_path():
    """HERE first, then the simulator.

    `heuristic_pick_drop` re-inserts its own folder and the simulator when
    it is imported, so the order is re-asserted after that import.
    """

    for path in (str(SIMULATOR), str(HERE)):
        while path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)


_fix_path()
import heuristic_pick_drop as H  # noqa: E402
_fix_path()

from continual_scene import ContinualScene  # noqa: E402
import tasks  # noqa: E402

# the one substitution: a scene with the can in it
H.BowlScene = ContinualScene


def use_clutter(distractors):
    """Make the expert run with other objects also on the table.

    `heuristic_pick_drop.run` builds its scene and then calls
    `scene.set_layout(object, object_xy, bowl_xy)`. Rather than edit that
    file - it is `pick_place_exp`'s, and its behaviour is measured - the
    SCENE is swapped for one whose `set_layout` also puts the distractors
    down. The controller is unchanged and still drives the named object,
    because `set_multi_layout` sets `active` to it.

    `distractors` maps object name -> (x, y). Pass {} to go back to a bare
    table.
    """

    class _Cluttered(ContinualScene):
        def set_layout(self, obj=None, object_xy=None, bowl_xy=None,
                       place_xy=None):
            layout = dict(distractors)
            if obj is not None and object_xy is not None:
                layout[obj] = object_xy
            self.set_multi_layout(layout,
                                  bowl_xy if bowl_xy is not None else place_xy,
                                  active=obj)

    H.BowlScene = _Cluttered if distractors else ContinualScene
    return H.BowlScene

run = H.run
plan = H.plan
RESULTS_DIR = HERE / "results"


def expert_args(task, extra=(), quiet=True):
    """Argument namespace for the expert on `task`.

    `--object` is set after parsing: the upstream parser's `choices` predate
    the can, and widening them there would edit a file `pick_place_exp` is
    measured with.
    """

    argv = ["--no-video"] + (["--no-outputs"] if quiet else []) + list(extra)
    args = H.parse_args(argv)
    args.object = tasks.BY_NAME[task]["object"]
    args.tag = args.tag or f"expert_{task}"
    return args


def main(argv=None):
    import argparse

    p = argparse.ArgumentParser(
        description="Run the expert on one continual task.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--task", default="cube", choices=tasks.TASK_NAMES)
    p.add_argument("--no-video", action="store_true")
    p.add_argument("rest", nargs="*", help="passed through to the expert")
    args = p.parse_args(argv)

    extra = list(args.rest)
    if not args.no_video:
        extra = [a for a in extra if a != "--no-video"]
    hargs = expert_args(args.task, extra=extra, quiet=False)
    hargs.tag = f"expert_{args.task}"
    H.RESULTS_DIR = RESULTS_DIR
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not args.no_video:
        hargs.no_video = False
    summary = run(hargs)
    print(f"  instruction: {tasks.instruction(args.task)!r}")
    return summary


if __name__ == "__main__":
    main()
