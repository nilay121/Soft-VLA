"""
What the tracking policy is allowed to look at, and why it is a question.

THE PROBLEM WITH REUSING THE PICK-AND-PLACE INPUTS
--------------------------------------------------
The pick-and-place policy saw three PAST frames and nothing else. That was
enough because the goal was physically in the picture: the cube and the bowl
were right there, so "what should I do next" was answerable from the current
image alone, and the history only supplied velocity.

Trajectory tracking is not like that. Given one image of the arm hanging in
empty space, there is no answer to "where next": the same picture is equally
consistent with tracing a circle clockwise, tracing it the other way, or
holding still. The target is not in the scene - by construction, since
`build_track_scene.py` deliberately draws no marker. So the reference has to
enter through the input, and the way it enters here is as GOAL IMAGES:
frames of the arm at the configurations it is supposed to pass through next.

    past frames      where the arm has been -> where it is going now, i.e.
                     velocity, which matters because the pneumatics lag
    future frames    where the arm should be -> the reference itself

At training time both come from the same recorded demonstration. At
evaluation the past frames come from the policy's own rollout while the
future frames come from a separate expert recording of the circle - which
is exactly "teach and repeat": show the robot a trace of the motion and ask
it to reproduce it.

THE ABLATION
------------
Whether history, future, or both are needed is an empirical question, so the
three are run head to head. All three use FOUR frames per camera, so the
network, its parameter count and its input size are identical and only the
MEANING of the frames changes:

    past_only     4 past,  0 future    the pick-and-place recipe
    future_only   1 past,  3 future    the goal, but no velocity
    both          2 past,  2 future    the default

`compare_conditioning.py` trains and deploys all three.
"""

from __future__ import annotations

POLICY_DT = 0.05          # s, 20 Hz; matches the pick-and-place policy

# Spacing between the sampled frames, in policy steps.
PAST_STRIDE = 4           # 0.20 s apart, as in pick-and-place
FUTURE_STRIDE = 5         # 0.25 s apart, so the goals span the action chunk
                          # (10 steps = 0.5 s) and a little beyond it

VARIANTS = {
    "past_only":   {"n_past": 4, "n_future": 0,
                    "label": "4 past, 0 future (pick-and-place recipe)"},
    "future_only": {"n_past": 1, "n_future": 3,
                    "label": "1 past, 3 future (goal, no velocity)"},
    "both":        {"n_past": 2, "n_future": 2,
                    "label": "2 past, 2 future"},
    # Six frames, not four, so this one is NOT part of the controlled
    # four-frame ablation above - it tests whether pushing the lookahead
    # further still helps, once 1p+3f had already beaten 2p+2f.
    "future5":     {"n_past": 1, "n_future": 5,
                    "label": "1 past, 5 future (1.25 s of lookahead)"},
}
# Measured: among the four-frame variants future_only won (1.40 mm against
# both's 2.08 mm and past_only's 75.65 mm). With a fixed budget, a slot spent
# on history is a goal frame not spent, and the previous-pressure input
# already tells the policy what it just did. `future5` extends that logic to
# six frames. See results/conditioning_comparison.json.
# Measured on the final dataset over circle/rectangle/heart/lemniscate:
# 1p+3f 1.63 mm, 1p+5f 1.95 mm. More lookahead did NOT help - 1p+3f won 3 of
# the 4 shapes - so the default stays at three goal frames.
DEFAULT_VARIANT = "future_only"


def frame_offsets(n_past, n_future):
    """Offsets in policy steps of every frame in the stack, oldest first.

    Negative and zero are the arm's own history (0 is now); positive are the
    goal frames. The order is fixed so a checkpoint and a rollout can never
    disagree about which channel group means what.
    """

    past = [-(n_past - 1 - i) * PAST_STRIDE for i in range(n_past)]
    future = [(j + 1) * FUTURE_STRIDE for j in range(n_future)]
    return past + future


def total_frames(n_past, n_future):
    return n_past + n_future


def resolve(variant):
    """(n_past, n_future, label) for a named variant."""

    if variant not in VARIANTS:
        raise SystemExit(f"unknown conditioning {variant!r}; "
                         f"choose from {list(VARIANTS)}")
    v = VARIANTS[variant]
    return v["n_past"], v["n_future"], v["label"]


__all__ = ["POLICY_DT", "PAST_STRIDE", "FUTURE_STRIDE", "VARIANTS",
           "DEFAULT_VARIANT", "frame_offsets", "total_frames", "resolve"]
