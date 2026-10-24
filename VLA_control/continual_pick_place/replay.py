"""
The replay buffers: what gets remembered from a task once it is over.

Three strategies are implemented so the method can be measured against
something rather than just demonstrated.

    none     nothing is remembered. Fine-tune on the new task and see how
             much of the old one survives. This is the lower bound, and the
             thing catastrophic forgetting is a name for.

    latent   LatentReplay. For a fixed number of samples per finished task,
             store the 768-float REPLAY-LAYER LATENT (the concatenated
             vision, proprioception and language codes) and the action chunk
             that followed it. Replaying one costs a forward pass through the
             trunk alone. This is the requested method.

    raw      RawReplay. Store the same NUMBER of samples, but as pointers
             back to the demonstrations on disk, and replay them through the
             whole network with fresh augmentation. Nothing is stale and the
             encoders keep learning, so this is the upper bound that latent
             replay is trading accuracy for memory against.

WHAT THE BUDGET ACTUALLY BUYS
-----------------------------
Per stored sample:

    latent   768 x float16 latent + 10 x 7 x float16 target  =    1 676 B
    raw      3 frames x 2 views x 112 x 112 x 3 uint8         =  225 792 B

so at equal sample count the latent buffer is ~135x smaller. Comparing them
at equal SAMPLES rather than equal BYTES is deliberately generous to raw
replay: at equal bytes it would be allowed 15 samples per task and would not
be an upper bound on anything.

WHY SAMPLING IS BALANCED ACROSS TASKS
-------------------------------------
Drawing uniformly from the pooled buffer would weight a task by how many of
its samples happen to be stored. Each finished task contributes an equal
share of every replay batch instead, so "how much of task 1 is being
rehearsed" does not quietly change when task 2 joins.
"""

from __future__ import annotations

import numpy as np

STRATEGIES = ("none", "latent", "raw")


def _subsample(n, capacity, rng):
    """Indices of `capacity` samples spread over n, or all of them."""

    if n <= capacity:
        return np.arange(n)
    return np.sort(rng.choice(n, size=capacity, replace=False))


class LatentReplay:
    """Fixed number of replay-layer latents per finished task.

    A stored latent is only a correct training pair while the encoders that
    produced it are unchanged, which is why `train_continual.py` freezes them
    the moment the first latents are written. The buffer records the fact so
    a mismatch is loud rather than silent.
    """

    kind = "latent"

    def __init__(self, capacity_per_task=2000, latent_dim=768,
                 chunk_steps=10, action_dim=7):
        self.capacity = int(capacity_per_task)
        self.latent_dim = latent_dim
        self.chunk_steps = chunk_steps
        self.action_dim = action_dim
        self.tasks = {}                 # name -> {"latent", "target"}
        self.encoder_fingerprint = None

    def add(self, task, latents, targets, rng, fingerprint=None):
        latents = np.asarray(latents, dtype=np.float16)
        targets = np.asarray(targets, dtype=np.float16)
        assert latents.shape[0] == targets.shape[0]
        keep = _subsample(latents.shape[0], self.capacity, rng)
        self.tasks[task] = {"latent": latents[keep], "target": targets[keep]}
        if fingerprint is not None:
            if (self.encoder_fingerprint is not None
                    and fingerprint != self.encoder_fingerprint):
                raise RuntimeError(
                    "the encoders changed between two fills of the latent "
                    "buffer, so the latents already in it no longer mean "
                    "what the trunk will read them as. Freeze the encoders "
                    "before the first fill.")
            self.encoder_fingerprint = fingerprint
        return len(keep)

    @property
    def task_names(self):
        return list(self.tasks)

    def __len__(self):
        return sum(t["latent"].shape[0] for t in self.tasks.values())

    def sample(self, batch, rng):
        """-> (latent [B, D] float32, target [B, C, A] float32, counts)."""

        if not self.tasks:
            return None
        names = self.task_names
        share = max(1, batch // len(names))
        latents, targets, counts = [], [], {}
        for name in names:
            store = self.tasks[name]
            n = store["latent"].shape[0]
            pick = rng.integers(0, n, size=share)
            latents.append(store["latent"][pick])
            targets.append(store["target"][pick])
            counts[name] = int(share)
        return (np.concatenate(latents).astype(np.float32),
                np.concatenate(targets).astype(np.float32), counts)

    def nbytes(self):
        return int(sum(t["latent"].nbytes + t["target"].nbytes
                       for t in self.tasks.values()))

    def report(self):
        return {"kind": self.kind, "capacity_per_task": self.capacity,
                "tasks": {k: int(v["latent"].shape[0])
                          for k, v in self.tasks.items()},
                "samples": len(self), "bytes": self.nbytes(),
                "bytes_per_sample": (self.nbytes() // max(len(self), 1))}


class RawReplay:
    """Fixed number of DEMONSTRATION SAMPLES per finished task, by reference.

    Nothing is copied: a sample is (task, episode index, frame index) and the
    trainer rebuilds the network input from the memory-mapped demo file, with
    a freshly drawn instruction and fresh photometric jitter, exactly as it
    does for the current task.
    """

    kind = "raw"

    def __init__(self, capacity_per_task=2000, sample_bytes=225792):
        self.capacity = int(capacity_per_task)
        self.sample_bytes = sample_bytes
        self.tasks = {}                 # name -> [(episode, t)]

    def add(self, task, index, rng, **_):
        index = list(index)
        keep = _subsample(len(index), self.capacity, rng)
        self.tasks[task] = [index[i] for i in keep]
        return len(keep)

    @property
    def task_names(self):
        return list(self.tasks)

    def __len__(self):
        return sum(len(v) for v in self.tasks.values())

    def sample(self, batch, rng):
        """-> [(task, episode, t)] balanced across the finished tasks."""

        if not self.tasks:
            return None
        names = self.task_names
        share = max(1, batch // len(names))
        picks = []
        for name in names:
            pool = self.tasks[name]
            for i in rng.integers(0, len(pool), size=share):
                picks.append((name, *pool[int(i)]))
        return picks

    def nbytes(self):
        """What it WOULD cost to hold these samples, for the comparison."""
        return int(len(self) * self.sample_bytes)

    def report(self):
        return {"kind": self.kind, "capacity_per_task": self.capacity,
                "tasks": {k: len(v) for k, v in self.tasks.items()},
                "samples": len(self), "bytes": self.nbytes(),
                "bytes_per_sample": self.sample_bytes,
                "note": "stored by reference into the demo files; `bytes` is "
                        "what the frames themselves would occupy"}


def make_buffer(strategy, capacity_per_task, latent_dim, chunk_steps,
                action_dim, sample_bytes):
    if strategy == "none":
        return None
    if strategy == "latent":
        return LatentReplay(capacity_per_task, latent_dim, chunk_steps,
                            action_dim)
    if strategy == "raw":
        return RawReplay(capacity_per_task, sample_bytes)
    raise ValueError(f"unknown replay strategy {strategy!r}; "
                     f"choose from {STRATEGIES}")


__all__ = ["LatentReplay", "RawReplay", "make_buffer", "STRATEGIES"]
