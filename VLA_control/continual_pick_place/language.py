"""
The language side of the VLA: a fixed vocabulary and a token encoder.

WHY A LEARNED EMBEDDING AND NOT A PRETRAINED TEXT MODEL
-------------------------------------------------------
The instruction space here is twelve sentences over a twenty-word
vocabulary. A sentence encoder with hundreds of millions of parameters would
dominate the 16 M-parameter policy it is supposed to condition, and every
one of its weights would be another thing changing between continual stages.
A small embedding table, mean-pooled, is the right size for the problem and
leaves the continual-learning result about the policy.

WHY THE VOCABULARY IS FROZEN BEFORE TASK 1
------------------------------------------
This is a continual-learning experiment, so the network's SHAPE must not
change between stages. If the vocabulary grew when the apple was introduced,
the embedding table would be resized mid-stream, old checkpoints would stop
loading, and any measured forgetting would be confounded with that surgery.
So `Vocabulary.standard()` is built once from tasks.all_instructions() -
every word of every task, including tasks not yet learned - and saved into
every checkpoint. The embedding rows for words not yet seen are simply never
trained, which is exactly the honest situation: the model has the capacity
to read the word "apple" before it has any idea what an apple is.
"""

from __future__ import annotations

import re

import torch
import torch.nn as nn

PAD = "<pad>"
UNK = "<unk>"
MAX_TOKENS = 16          # the longest instruction is 11 tokens

_WORD = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return _WORD.findall(str(text).lower())


class Vocabulary:
    def __init__(self, words):
        self.words = [PAD, UNK] + sorted(set(words) - {PAD, UNK})
        self.index = {w: i for i, w in enumerate(self.words)}

    def __len__(self):
        return len(self.words)

    @classmethod
    def standard(cls):
        """The vocabulary of every instruction of every task."""
        import tasks
        words = [w for s in tasks.all_instructions() for w in tokenize(s)]
        return cls(words)

    def encode(self, text, max_tokens=MAX_TOKENS):
        """-> (ids [max_tokens] int64, mask [max_tokens] float32)."""
        ids = [self.index.get(w, 1) for w in tokenize(text)][:max_tokens]
        mask = [1.0] * len(ids) + [0.0] * (max_tokens - len(ids))
        ids = ids + [0] * (max_tokens - len(ids))
        return ids, mask

    def encode_batch(self, texts, max_tokens=MAX_TOKENS, device=None):
        pairs = [self.encode(t, max_tokens) for t in texts]
        ids = torch.tensor([p[0] for p in pairs], dtype=torch.long)
        mask = torch.tensor([p[1] for p in pairs], dtype=torch.float32)
        if device is not None:
            ids, mask = ids.to(device), mask.to(device)
        return ids, mask

    def state(self):
        return {"words": self.words}

    @classmethod
    def restore(cls, state):
        v = cls.__new__(cls)
        v.words = list(state["words"])
        v.index = {w: i for i, w in enumerate(v.words)}
        return v


class LanguageEncoder(nn.Module):
    """Token embedding -> masked mean -> MLP, out to `output_dim`.

    Mean pooling and not an RNN or a transformer: these instructions are a
    bag of words with no syntax to exploit - "put the red apple into the
    bowl" and "into the bowl put the red apple" should mean the same thing -
    and order-invariance is one fewer way to overfit twelve sentences.
    """

    def __init__(self, vocab_size, output_dim=256, embed_dim=64,
                 padding_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim,
                                      padding_idx=padding_idx)
        nn.init.normal_(self.embedding.weight, std=0.02)
        with torch.no_grad():
            self.embedding.weight[padding_idx].zero_()
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Linear(128, output_dim), nn.LayerNorm(output_dim), nn.GELU())

    def forward(self, ids, mask):
        embedded = self.embedding(ids)                    # [B, T, E]
        weight = mask.unsqueeze(-1)
        pooled = (embedded * weight).sum(1) / weight.sum(1).clamp(min=1.0)
        return self.projection(pooled)


__all__ = ["Vocabulary", "LanguageEncoder", "tokenize", "PAD", "UNK",
           "MAX_TOKENS"]
