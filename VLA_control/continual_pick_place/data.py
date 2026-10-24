"""
The demonstrations, as training samples: shared by the trainer and the
buffer filler so there is one definition of "a sample".

A sample is a (task, episode, frame) triple turned into

    images   FRAME_HISTORY frames from each of the 2 views at t, t-s, t-2s
    state    the 7 pressures commanded at t-1
    tokens   one of the task's four instructions, drawn at random
    target   the CHANGE to those pressures over the next CHUNK_STEPS steps

THE SPLIT IS BY EPISODE
-----------------------
Neighbouring frames of one episode are near-duplicates, so splitting by
frame would put the same moment in train and in validation and report a
validation score that means nothing. Ten per cent of each task's episodes go
to validation and ten per cent to test, drawn with a fixed seed so every arm
of the experiment sees exactly the same split - otherwise a difference
between two arms could just be a difference in which episodes they trained
on.

THE INSTRUCTION IS DRAWN PER SAMPLE, NOT PER EPISODE
----------------------------------------------------
The same frame is seen with all four of its task's paraphrases over the
course of training. That is what stops the language branch keying on some
incidental token: "into" appears in two of the four templates, so it cannot
be what tells the network this is the cube task.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

import tasks
from augment import photometric_jitter, view_channel_groups
from language import MAX_TOKENS
from vla_model import (
    CHUNK_STEPS,
    FRAME_HISTORY,
    FRAME_STRIDE,
    NUM_VIEWS,
    stack_frames,
)

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "dataset"


GROUNDING_SCALE = 0.05      # m; object start positions lie within ~0.06
REACH_PHASE = 0.4           # fraction of an episode before the pick


def load_task(task, data_dir=DATA_DIR):
    """Frames stay on disk (memory-mapped); only the pressures are read."""

    folder = Path(data_dir) / task
    files = sorted(folder.glob("episode_*.npz"))
    if not files:
        raise SystemExit(f"no demonstrations in {folder} - run "
                         f"generate_demos.py --task {task}")
    episodes = []
    for f in files:
        d = np.load(f)
        episodes.append({"frames": np.load(f.with_suffix(".npy"),
                                           mmap_mode="r"),
                         "pressure": d["pressure"], "name": f.stem,
                         # where the NAMED object started - the label for
                         # the grounding loss of arch minilm_aux
                         "object_xy": (d["object_xy"] if "object_xy" in d.files
                                       else np.zeros(2, np.float32))})
    meta = json.loads((folder / "metadata.json").read_text())
    return episodes, meta


def split_episodes(n, seed=0, val_fraction=0.1):
    """(train, val, test) episode indices - identical for every arm."""

    order = np.random.default_rng(seed).permutation(n)
    k = max(1, int(val_fraction * n))
    if n < 3 * k:
        raise SystemExit(f"{n} episodes is too few to split 10/10/80")
    return order[2 * k:], order[:k], order[k:2 * k]


def sample_index(episodes, which, stride=1):
    """[(episode, t)] over the given episodes, every `stride`th frame."""

    return [(int(e), t) for e in which
            for t in range(1, len(episodes[int(e)]["pressure"]), stride)]


class Corpus:
    """Every task's demonstrations and splits, loaded once."""

    def __init__(self, task_names=None, data_dir=DATA_DIR, seed=0,
                 train_stride=2, eval_stride=1):
        self.names = list(task_names or tasks.TASK_NAMES)
        self.episodes, self.meta, self.splits, self.index = {}, {}, {}, {}
        for name in self.names:
            eps, meta = load_task(name, data_dir)
            self.episodes[name] = eps
            self.meta[name] = meta
            train, val, test = split_episodes(len(eps), seed=seed)
            self.splits[name] = {"train": train, "val": val, "test": test}
            self.index[name] = {
                "train": sample_index(eps, train, train_stride),
                # Validation and test are strided too. They are only ever a
                # relative number - "is this epoch better than the last" -
                # and every second frame of an episode is very nearly the
                # previous one, so a stride costs almost no resolution and
                # buys back a large share of each epoch's wall clock.
                "val": sample_index(eps, val, eval_stride),
                "test": sample_index(eps, test, eval_stride),
            }

    def samples(self, task, split):
        return [(task, e, t) for e, t in self.index[task][split]]

    def pooled(self, task_names, split):
        return [s for name in task_names for s in self.samples(name, split)]


class DemoDataset(Dataset):
    """Samples drawn as (task, episode, frame); see the module docstring."""

    def __init__(self, corpus, samples, vocab, augment=False, state_noise=0.0,
                 strength=None, seed=0, instruction_index=None, aux=False,
                 raw_text=False):
        self.corpus = corpus
        self.samples = list(samples)
        self.augment = augment
        self.state_noise = state_noise
        self.strength = strength
        self.groups = view_channel_groups(NUM_VIEWS, FRAME_HISTORY)
        self.seed = seed
        self.instruction_index = instruction_index    # None = draw at random
        self.aux = aux    # also return the grounding target (see __getitem__)
        # raw_text: hand the instruction over as the sentence itself (MiniLM
        # models read it with their own tokenizer) instead of as ids in the
        # project vocabulary, which only the learned-embedding archs need
        self.raw_text = raw_text
        self._rng = None

        # every (task, template) tokenised once
        self.tokens = {}
        for name in corpus.names:
            for k, text in enumerate(tasks.instructions(name)):
                ids, mask = vocab.encode(text, MAX_TOKENS)
                self.tokens[(name, k)] = (np.asarray(ids, dtype=np.int64),
                                          np.asarray(mask, dtype=np.float32))
        self.task_ids = {name: i for i, name in enumerate(tasks.TASK_NAMES)}

    @property
    def rng(self):
        """One generator per worker AND per epoch.

        torch.initial_seed() is the worker's own seed, which DataLoader
        redraws from the main generator every epoch, so the jitter and the
        instruction choice differ between workers and between epochs.
        Seeding from self.seed alone would replay the same draws for ever.
        """

        if self._rng is None:
            info = torch.utils.data.get_worker_info()
            worker = 0 if info is None else info.id
            self._rng = np.random.default_rng(
                (int(torch.initial_seed()) + 7919 * (worker + 1)
                 + self.seed) % (2 ** 63))
        return self._rng

    def __len__(self):
        return len(self.samples)

    def build(self, task, ep, t, rng):
        episode = self.corpus.episodes[task][ep]
        frames, pressure = episode["frames"], episode["pressure"]

        picks = [max(0, t - k * FRAME_STRIDE)
                 for k in range(FRAME_HISTORY - 1, -1, -1)]
        images = stack_frames([frames[p] for p in picks])
        if self.augment:
            images = photometric_jitter(images, rng, self.groups,
                                        self.strength)

        previous = pressure[t - 1].copy()
        # the next CHUNK_STEPS changes; the tail of an episode repeats its
        # last command, i.e. asks for no further change
        future = np.concatenate(
            [pressure[t:t + CHUNK_STEPS],
             np.repeat(pressure[-1:], CHUNK_STEPS, axis=0)])[:CHUNK_STEPS]
        delta = np.diff(np.vstack([previous, future]),
                        axis=0).astype(np.float32)
        if self.state_noise:
            previous = previous + rng.normal(
                0.0, self.state_noise, previous.shape).astype(np.float32)

        k = (self.instruction_index if self.instruction_index is not None
             else int(rng.integers(0, len(tasks.TEMPLATES))))
        if self.raw_text:
            return (images, previous.astype(np.float32),
                    tasks.instructions(task)[k], np.zeros(0, np.float32), delta)
        ids, mask = self.tokens[(task, k)]
        return images, previous.astype(np.float32), ids, mask, delta

    def __getitem__(self, i):
        task, ep, t = self.samples[i]
        images, previous, ids, mask, delta = self.build(task, ep, t, self.rng)
        return (torch.from_numpy(np.ascontiguousarray(images)),
                torch.from_numpy(previous),
                ids if self.raw_text else torch.from_numpy(ids),
                torch.from_numpy(mask),
                torch.from_numpy(delta),
                self.task_ids[task]) + ((self.grounding(task, ep, t),)
                                        if self.aux else ())

    def grounding(self, task, ep, t):
        """[x, y, weight]: the named object's start position in units of
        GROUNDING_SCALE, weighted 1 only in the reach phase - the first
        REACH_PHASE of the episode, before the object has been picked up and
        its start position stops being visible in the recent frames."""

        episode = self.corpus.episodes[task][ep]
        xy = np.asarray(episode["object_xy"], np.float32) / GROUNDING_SCALE
        reach = float(t < REACH_PHASE * len(episode["pressure"]))
        return torch.tensor([xy[0], xy[1], reach], dtype=torch.float32)


__all__ = ["Corpus", "DemoDataset", "load_task", "split_episodes",
           "sample_index", "DATA_DIR"]
