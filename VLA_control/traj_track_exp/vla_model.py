"""
Vision-only VLA for the hanging arm + soft gripper pick-and-place.

Adapted from ../../VLA_control/vla_model.py (same convolutional encoder,
fusion trunk and residual blocks), with the interfaces changed for this task:

                        tracking VLA (2_module)      this one
    image               1 frame                      FRAME_HISTORY frames
    instruction         32 tokens                    dropped (one task)
    state               tip xyz + 6 pressures        previous 7 pressures
    target_error        3-D vector                   DROPPED - see below
    ---------------------------------------------------------------
    action              6 pressure deltas            7 pressure deltas
                                                     -> next 7 pressures

WHY NO TARGET ERROR
-------------------
The tracking VLA was handed `goal - tip`, computed from the simulator's exact
tip and a known goal. Here nothing of the sort is given: where the cube is,
where the green place marker is, and how far the gripper still has to go must
all be read out of the camera image. The only non-visual input is the
robot's own last command, which is proprioception a real controller always
has.

WHY A STACK OF FRAMES
---------------------
A single frame does not say whether the arm is going down or coming up, nor
how fast. FRAME_HISTORY frames spaced FRAME_STRIDE policy steps apart give
the network that motion, and the previous gripper pressure tells it whether
the cube is already in hand.

WHY THE HEAD PREDICTS A CHUNK OF CHANGES
----------------------------------------
Predicting one 50 ms step at a time, the policy stalled: it ran the right
sequence far too slowly and then got stuck in a pose its demonstrations never
contained (measured: 1/10 picks, and giving it twice the time did not help).
So the head predicts CHUNK_STEPS future changes at once and the loop executes
CHUNK_EXECUTE of them before asking again. The policy commits to motion, and
there are 8x fewer chances to drift out of distribution.

WHY THE HEAD PREDICTS A CHANGE, NOT THE ABSOLUTE PRESSURE
---------------------------------------------------------
`next_pressure` given `previous_pressure` is almost the identity: the
expert's command moves at most 0.06 bar per 50 ms step. A network that
simply copies its input scores a *better* validation error than one that
learns the task, and that is exactly what the first version did - it froze
after two seconds and never closed the gripper. So the head predicts the
per-step change, standardised by the training set's own change statistics:
predicting "no change" then costs a full standard deviation and the network
has to learn the motion. `next_pressure()` adds the change back on.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------

NUM_ARM_CHAMBERS = 6
NUM_PRESSURES = NUM_ARM_CHAMBERS + 1        # + the single gripper line
ACTION_DIM = NUM_PRESSURES
STATE_DIM = NUM_PRESSURES                   # the previous command

IMAGE_SIZE = 112
NUM_VIEWS = 2                               # close + side camera
FRAME_HISTORY = 3                           # current + 2 older frames
FRAME_STRIDE = 4                            # policy steps between them (0.4 s span)
POLICY_DT = 0.05                            # s, 20 Hz
CHUNK_STEPS = 10                            # changes predicted per call (0.5 s)
CHUNK_EXECUTE = 5                           # of those, how many are executed

PRESSURE_LABELS = ("A1", "A2", "A3", "B1", "B2", "B3", "grip")


def stack_frames(frames):
    """[FRAME_HISTORY][NUM_VIEWS, H, W, 3] (or [H,W,3]) uint8/float
    -> [3*NUM_VIEWS*FRAME_HISTORY, H, W] float32 in [0,1], oldest first."""

    planes = []
    for f in frames:
        f = np.asarray(f, dtype=np.float32)
        if f.max() > 1.001:
            f = f / 255.0
        if f.ndim == 3:                       # single view
            f = f[None]
        for view in f:                        # [H,W,3]
            planes.append(np.transpose(view, (2, 0, 1)))
    return np.concatenate(planes, axis=0)


IN_CHANNELS = 3 * NUM_VIEWS * FRAME_HISTORY


# ---------------------------------------------------------------------------
# Encoders  (same shapes as the tracking VLA, wider first conv)
# ---------------------------------------------------------------------------

class VisualEncoder(nn.Module):
    """Five strided conv blocks then global average pool (resolution free)."""

    def __init__(self, in_channels, output_dim=256):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(in_channels, 32, 5, stride=2, padding=2),
            nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.GELU(),
            nn.Conv2d(256, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def forward(self, image):
        return self.projection(self.network(image))


class MLPEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128), nn.GELU(),
            nn.Linear(128, output_dim),
            nn.LayerNorm(output_dim), nn.GELU(),
        )

    def forward(self, x):
        return self.network(x)


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.fc2(self.dropout(F.gelu(self.fc1(self.norm(x)))))
        return residual + self.dropout(x)


class ResNetEncoder(nn.Module):
    """ImageNet-pretrained ResNet18 trunk, first conv inflated to the stacked
    input. Inflating (copying the RGB filters across the frame/view groups and
    dividing by their number) keeps the pretrained response to edges and
    colour while accepting all frames in one pass."""

    def __init__(self, in_channels, output_dim=256):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        groups = in_channels // 3
        old = net.conv1.weight.data
        conv = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)
        conv.weight.data = old.repeat(1, groups, 1, 1) / groups
        net.conv1 = conv
        net.fc = nn.Identity()
        self.net = net

        self.projection = nn.Sequential(
            nn.Linear(512, output_dim), nn.LayerNorm(output_dim), nn.GELU())

        # ImageNet statistics, tiled over the stacked frames
        mean = torch.tensor([0.485, 0.456, 0.406]).repeat(groups)
        std = torch.tensor([0.229, 0.224, 0.225]).repeat(groups)
        self.register_buffer("pixel_mean", mean.view(1, -1, 1, 1))
        self.register_buffer("pixel_std", std.view(1, -1, 1, 1))

    def forward(self, image):
        x = (image - self.pixel_mean) / self.pixel_std
        return self.projection(self.net(x))


class TransformerEncoder(nn.Module):
    """Conv stem, then a small transformer over the feature-map cells.

    The CNN encoder pools its 7x7 map into one vector and loses where things
    are; self-attention over the 49 cells keeps that spatial structure and
    lets the gripper cell attend to the cube cell.
    """

    def __init__(self, in_channels, output_dim=256, dim=256, depth=3,
                 heads=4):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 5, stride=2, padding=2),
            nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(dim), nn.GELU(),
        )
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.positions = nn.Parameter(torch.zeros(1, 49 + 1, dim))
        nn.init.trunc_normal_(self.positions, std=0.02)

        layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout=0.1,
                                           activation="gelu",
                                           batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, depth)
        self.projection = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, output_dim), nn.GELU())

    def forward(self, image):
        x = self.stem(image)                       # [B, dim, 7, 7]
        x = x.flatten(2).transpose(1, 2)           # [B, 49, dim]
        x = torch.cat([self.cls.expand(x.shape[0], -1, -1), x], dim=1)
        x = self.transformer(x + self.positions[:, :x.shape[1]])
        return self.projection(x[:, 0])


VISUAL_ENCODERS = {
    "cnn": lambda c, d: VisualEncoder(c, d),
    "resnet18": lambda c, d: ResNetEncoder(c, d),
    "transformer": lambda c, d: TransformerEncoder(c, d),
}


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

class PickPlaceVLA(nn.Module):
    def __init__(self, frame_history=FRAME_HISTORY, state_dim=STATE_DIM,
                 action_dim=ACTION_DIM, hidden_dim=512, num_residual_blocks=4,
                 chunk_steps=CHUNK_STEPS, arch="cnn", num_views=NUM_VIEWS):
        super().__init__()

        self.arch = arch
        self.num_views = num_views
        self.frame_history = frame_history
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_steps = chunk_steps

        # Standardisation, filled from the training set. Without it the
        # network barely sees small-magnitude inputs (the lesson from the
        # tracking VLA, where raw millimetre errors were invisible).
        self.register_buffer("state_mean", torch.zeros(state_dim))
        self.register_buffer("state_std", torch.ones(state_dim))
        # statistics of the pressure CHANGE per policy step
        self.register_buffer("action_mean", torch.zeros(action_dim))
        self.register_buffer("action_std", torch.ones(action_dim))

        self.visual_encoder = VISUAL_ENCODERS[arch](
            3 * num_views * frame_history, 256)
        self.state_encoder = MLPEncoder(state_dim, 256)

        self.fusion = nn.Sequential(
            nn.Linear(256 + 256, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.residual_blocks = nn.ModuleList(
            [ResidualMLPBlock(hidden_dim) for _ in range(num_residual_blocks)])
        self.action_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, action_dim * chunk_steps),
        )

    def set_normalisation(self, states, actions):
        for name, values in (("state", states), ("action", actions)):
            v = torch.as_tensor(np.asarray(values), dtype=torch.float32)
            getattr(self, f"{name}_mean").copy_(v.mean(0))
            getattr(self, f"{name}_std").copy_(v.std(0).clamp(min=1e-6))

    def forward(self, images, state):
        """images [B, 3*frames, H, W] in [0,1]; state = previous pressures.
        Returns [B, chunk_steps, 7] pressure CHANGES, in bar."""

        x = torch.cat([self.visual_encoder(images),
                       self.state_encoder((state - self.state_mean)
                                          / self.state_std)], dim=1)
        x = self.fusion(x)
        for block in self.residual_blocks:
            x = block(x)
        out = self.action_head(x).view(-1, self.chunk_steps, self.action_dim)
        return out * self.action_std + self.action_mean

    def pressure_chunk(self, images, state):
        """The next `chunk_steps` pressure commands in bar: the previous
        command plus the cumulative predicted changes."""
        return state.unsqueeze(1) + torch.cumsum(self.forward(images, state),
                                                 dim=1)


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

def save_checkpoint(path, model, extra=None):
    payload = {
        "model_state_dict": model.state_dict(),
        "frame_history": model.frame_history,
        "arch": model.arch,
        "num_views": model.num_views,
        "chunk_steps": model.chunk_steps,
        "frame_stride": FRAME_STRIDE,
        "state_dim": model.state_dim,
        "action_dim": model.action_dim,
        "convention": ("inputs: camera frames (oldest first) + previous 7 "
                       "pressures [A1 A2 A3 B1 B2 B3 grip]; output: the "
                       "pressure CHANGE for this step in bar, so next "
                       "pressure = previous + output"),
    }
    if extra:
        payload.update(extra)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path, device):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}\n"
                                "run generate_demos.py then train.py")

    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = PickPlaceVLA(frame_history=int(ckpt.get("frame_history",
                                                    FRAME_HISTORY)),
                         state_dim=int(ckpt.get("state_dim", STATE_DIM)),
                         action_dim=int(ckpt.get("action_dim", ACTION_DIM)),
                         chunk_steps=int(ckpt.get("chunk_steps", CHUNK_STEPS)),
                         arch=ckpt.get("arch", "cnn"),
                         num_views=int(ckpt.get("num_views", NUM_VIEWS)))
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, ckpt


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    images = torch.zeros(2, IN_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
    for arch in VISUAL_ENCODERS:
        m = PickPlaceVLA(arch=arch)
        out = m(images, torch.zeros(2, STATE_DIM))
        print(f"{arch:12s} {count_parameters(m):>10,} parameters   "
              f"forward -> {tuple(out.shape)} [batch, chunk, pressures]")
