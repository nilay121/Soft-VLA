"""
The three tasks, in the order they are learned, and the words that name them.

    1. cube     "pick up the blue cube and drop it into the bowl"
    2. apple    "pick up the red apple and drop it into the bowl"
    3. can      "pick up the silver can and drop it into the bowl"

Each task is one OBJECT. The bowl, the table, the arm, the cameras and the
success test are shared, so what changes between tasks is only the thing
being picked up - which is what makes this a clean class-incremental
problem rather than three unrelated jobs.

THE INSTRUCTIONS ARE PARAPHRASED
--------------------------------
One fixed sentence per task would make the language input a one-hot task
label wearing a costume: the network could key on any token at all, since
every token is perfectly correlated with the task. So each task draws from
four sentence templates, and the templates are SHARED across tasks. The only
tokens that differ between two instructions for two different tasks are the
colour and the noun. The language branch therefore has to learn to read
those two words, because nothing else in the sentence carries task identity.

WHY THE COLOURS ARE NOT THE ONES IN THE EXAMPLE
----------------------------------------------
The cube in this scene has been blue since it was built (rgba 0.15 0.6 0.9)
and the apple is the red one. Calling the cube red in the instruction would
be the one thing in the sentence that does not correspond to anything the
camera sees, so the true colours are used.

WHAT LANGUAGE IS AND IS NOT DOING HERE
--------------------------------------
Exactly one object is on the table at a time, so the instruction is
redundant WITH THE IMAGE: a policy could in principle work out which object
it is looking at from pixels alone. Language is still load-bearing for the
continual-learning method, because with latent replay the encoders are
frozen after task 1 and the language latent is the one input that names the
task unambiguously and cheaply. `deploy.py --swap-instruction` measures how
much the policy actually leans on it, rather than assuming. Putting
distractor objects on the table would make language strictly necessary; see
the README's "what is missing" section.
"""

from __future__ import annotations

import os

# order = the order they are learned in
ALL_TASKS = [
    {"name": "cube",  "object": "cube",  "colour": "blue",
     "noun": "cube",
     "note": "the task pick_place_exp was trained on, reused as task 1"},
    {"name": "apple", "object": "apple", "colour": "red",
     "noun": "apple",
     "note": "rounder and 6 mm taller than the cube; rolls if nudged"},
    {"name": "can",   "object": "can",   "colour": "silver",
     "noun": "can",
     "note": "a 29 x 33 mm cylinder, added for this experiment"},
    {"name": "cup",   "object": "cup",   "colour": "yellow",
     "noun": "cup",
     "note": "26 mm wide, 32 mm tall; the handle is paint only"},
    {"name": "teddy", "object": "teddy", "colour": "brown",
     "noun": "teddy bear",
     "note": "round body and head, 37 mm tall; limbs are paint only"},
]

# The task SET is chosen once per process, from the environment, so every
# script and every spawned worker agrees. CONTINUAL_TASKS is a count (the
# first N of ALL_TASKS; 3 = the original cube/apple/can experiment, the
# default) or a comma-separated list of task names in learning order, e.g.
# "cube,apple,can,cup". The vocabulary, the objects on the table and the
# learning order all follow it.
_SPEC = os.environ.get("CONTINUAL_TASKS", "3").strip()
_BY = {t["name"]: t for t in ALL_TASKS}
if _SPEC.isdigit():
    TASKS = ALL_TASKS[:int(_SPEC)]
else:
    _names = [n.strip() for n in _SPEC.split(",") if n.strip()]
    _bad = [n for n in _names if n not in _BY]
    if _bad:
        raise SystemExit(f"CONTINUAL_TASKS: unknown task(s) {_bad}; known: "
                         f"{list(_BY)}")
    TASKS = [_BY[n] for n in _names]
TASK_COUNT = len(TASKS)

TASK_NAMES = [t["name"] for t in TASKS]
BY_NAME = {t["name"]: t for t in TASKS}

# Shared across tasks on purpose: see the module docstring.
TEMPLATES = (
    "pick up the {phrase} and drop it into the bowl",
    "grasp the {phrase} and place it in the bowl",
    "put the {phrase} into the bowl",
    "lift the {phrase} and release it over the bowl",
)


def phrase(task):
    return f"{BY_NAME[task]['colour']} {BY_NAME[task]['noun']}"


def instructions(task):
    """Every instruction that may be used for `task`."""
    return [t.format(phrase=phrase(task)) for t in TEMPLATES]


def instruction(task, index=0):
    return instructions(task)[index % len(TEMPLATES)]


def all_instructions():
    """Every instruction of every task - the corpus the vocabulary is built
    from, once, before any training happens."""
    return [s for name in TASK_NAMES for s in instructions(name)]


def stages(upto=None):
    """Task names in learning order, optionally truncated."""
    return TASK_NAMES[:upto] if upto else list(TASK_NAMES)


__all__ = ["TASKS", "ALL_TASKS", "TASK_COUNT", "TASK_NAMES", "BY_NAME", "TEMPLATES", "instructions",
           "instruction", "all_instructions", "phrase", "stages"]
