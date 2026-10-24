"""
Photometric-only augmentation for the bowl-drop policy.

WHY NOTHING GEOMETRIC
---------------------
Both cameras are rigidly mounted - one on a tripod, one on the wrist - and
the policy is never told where anything is. With a fixed camera, WHERE a
thing appears in the frame IS the position signal: a cube at pixel (40, 70)
means "reach left and forward". Translate or rotate the image and that
cube now means something else, so the label no longer matches the input.

This is not speculation about this task. The same policy family was trained
on the pick-and-place data with +-6 px random shifts and it went from 4/10
picks to 0/10, while its offline error barely moved
(../../VLA_control/README.md). Shift augmentation teaches the network to
ignore exactly the cue it needs.

So this module only ever changes COLOUR AND INTENSITY. Every pixel stays
where it was:

    brightness   scale        a lamp at a different distance
    contrast     about the view's own mean
    saturation   blend toward greyscale
    hue          small luminance-preserving rotation about the grey axis
    noise        per-pixel Gaussian; sensor noise

ONE JITTER PER VIEW, SHARED BY THE FRAMES
-----------------------------------------
The two cameras are different devices, so they get independent jitter. The
FRAMES WITHIN one view do not: they are 200 ms apart and the network reads
their differences as motion. Jittering them independently would manufacture
brightness flicker that looks like movement, which is the same mistake as a
geometric transform, just on the time axis instead of the spatial one.
"""

from __future__ import annotations

import numpy as np

# Default strengths. Deliberately moderate: the bowl is off-white, the cube
# is blue and the table is brown, so colour carries real information here
# and a violent hue shift would destroy it.
DEFAULTS = {
    "brightness": 0.20,     # x *= U(1-b, 1+b)
    "contrast": 0.20,       # (x-mean) *= U(1-c, 1+c)
    "saturation": 0.25,     # blend to grey by U(1-s, 1+s)
    "hue": 0.020,           # turns of the RGB colour wheel, +-
    "noise": 0.015,         # max per-pixel Gaussian sigma
}

_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def _hue_matrix(turns):
    """Luminance-preserving rotation about the grey axis of RGB."""

    theta = 2.0 * np.pi * float(turns)
    cos, sin = np.cos(theta), np.sin(theta)
    r, g, b = 0.213, 0.715, 0.072
    return np.array([
        [r + cos * (1 - r) - sin * r,
         g - cos * g - sin * g,
         b - cos * b + sin * (1 - b)],
        [r - cos * r + sin * 0.143,
         g + cos * (1 - g) + sin * 0.140,
         b - cos * b - sin * 0.283],
        [r - cos * r - sin * (1 - r),
         g - cos * g + sin * g,
         b + cos * (1 - b) + sin * b],
    ], dtype=np.float32)


def view_channel_groups(num_views, frame_history):
    """Channel indices of each view in a stack_frames() tensor.

    stack_frames lays the planes out frame-major then view-major, three
    channels at a time, so view v owns
    ((f * num_views + v) * 3 + c) for every frame f and colour c.
    """

    return [np.array([(f * num_views + v) * 3 + c
                      for f in range(frame_history)
                      for c in range(3)], dtype=np.int64)
            for v in range(num_views)]


def photometric_jitter(images, rng, groups, strength=None):
    """Jitter a [C, H, W] float32 stack in [0, 1], in place-safe fashion.

    `groups` comes from view_channel_groups(); each group is jittered with
    its own random draw and every frame inside it shares that draw.
    """

    cfg = dict(DEFAULTS)
    if strength:
        cfg.update(strength)

    out = np.array(images, dtype=np.float32, copy=True)

    for channels in groups:
        # [frames*3, H, W] -> [frames, 3, H, W] so colour ops see RGB triples
        block = out[channels]
        frames = block.reshape(-1, 3, *block.shape[1:])

        if cfg["brightness"]:
            frames *= np.float32(
                rng.uniform(1 - cfg["brightness"], 1 + cfg["brightness"]))

        if cfg["contrast"]:
            mean = np.float32(frames.mean())
            frames[...] = (frames - mean) * np.float32(
                rng.uniform(1 - cfg["contrast"], 1 + cfg["contrast"])) + mean

        if cfg["saturation"]:
            grey = np.tensordot(_LUMA, frames, axes=([0], [1]))[:, None]
            frames[...] = grey + (frames - grey) * np.float32(
                rng.uniform(1 - cfg["saturation"], 1 + cfg["saturation"]))

        if cfg["hue"]:
            matrix = _hue_matrix(rng.uniform(-cfg["hue"], cfg["hue"]))
            frames[...] = np.einsum("ij,njhw->nihw", matrix, frames)

        if cfg["noise"]:
            sigma = rng.uniform(0.0, cfg["noise"])
            if sigma > 0:
                frames += rng.normal(0.0, sigma,
                                     frames.shape).astype(np.float32)

        out[channels] = frames.reshape(block.shape)

    return np.clip(out, 0.0, 1.0, out=out)


__all__ = ["photometric_jitter", "view_channel_groups", "DEFAULTS"]
