#!/usr/bin/env python3
"""
Draw the VLA architecture (arch `minilm_aux`) and its training / deployment
loop as a two-page PDF.

    python make_architecture_diagram.py        -> vla_architecture.pdf (+ .png per page)

Every dimension below is the one in vla_model.py / train_continual.py for
the best checkpoint, checkpoints/minilm_aux_raw8000_spread/.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "vla_architecture.pdf"

# colours: one family per modality, grey = frozen, dashed = training only
C = {
    "input": "#f2f2f2",
    "vision": "#dbe8f7", "vision_d": "#2f6db3",
    "lang": "#e3f1df", "lang_d": "#3f8a36",
    "state": "#fbe7d3", "state_d": "#c46a12",
    "fusion": "#ece2f5", "fusion_d": "#7342a3",
    "trunk": "#fde3e3", "trunk_d": "#b83232",
    "frozen": "#d9d9d9", "frozen_d": "#555555",
    "out": "#fff4c2", "out_d": "#9a7b00",
    "aux": "#ffffff", "aux_d": "#666666",
    "latent": "#fff9e6",
}


def box(ax, x, y, w, h, title, body="", fc="#ffffff", ec="#333333",
        dashed=False, fs=8.2, tfs=9.2, lw=1.2, align="center"):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.35",
                                fc=fc, ec=ec, lw=lw,
                                ls=(0, (4, 2.5)) if dashed else "-"))
    if body:
        ax.text(x + w / 2, y + h - 0.45, title, ha="center", va="top",
                fontsize=tfs, fontweight="bold", color=ec)
        ax.text(x + (w / 2 if align == "center" else 0.35), y + h - 1.55,
                body, ha=align, va="top", fontsize=fs, color="#222222",
                linespacing=1.25)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=tfs, fontweight="bold", color=ec)


def arrow(ax, x1, y1, x2, y2, label="", color="#333333", fs=7.6, lw=1.2,
          dashed=False, lpos=0.5, loff=(0, 0.35), rad=0.0, ha="center"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                shrinkA=0, shrinkB=0, mutation_scale=11,
                                ls=(0, (4, 2.5)) if dashed else "-",
                                connectionstyle=f"arc3,rad={rad}"))
    if label:
        ax.text(x1 + (x2 - x1) * lpos + loff[0],
                y1 + (y2 - y1) * lpos + loff[1], label, ha=ha, va="bottom",
                fontsize=fs, color=color, style="italic")


def new_page(title, subtitle):
    fig = plt.figure(figsize=(23.4, 16.5))          # A2 landscape
    ax = fig.add_axes([0.01, 0.01, 0.98, 0.98])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 70)
    ax.axis("off")
    ax.text(50, 69.2, title, ha="center", va="top", fontsize=20,
            fontweight="bold")
    ax.text(50, 67.4, subtitle, ha="center", va="top", fontsize=11.5,
            color="#444444")
    return fig, ax


# ---------------------------------------------------------------------------
# Page 1: the network
# ---------------------------------------------------------------------------

def tag(ax, x, y, t, color, r=0.62):
    """A labelled connector: the same letter where a tensor is produced and
    where it is consumed, instead of a long wire across the page."""
    from matplotlib.patches import Circle
    ax.add_patch(Circle((x, y), r, fc="#ffffff", ec=color, lw=1.4, zorder=5))
    ax.text(x, y, t, ha="center", va="center", fontsize=8.6, fontweight="bold",
            color=color, zorder=6)


def wire(ax, pts, color="#333333", dashed=False, lw=1.2, label="", lidx=0,
         loff=(0, 0.3), ha="center", fs=7.6):
    """Orthogonal polyline with an arrowhead at the end."""
    xs, ys = zip(*pts)
    ax.plot(xs[:-1] + (xs[-1],), ys[:-1] + (ys[-1],), color=color, lw=lw,
            ls=(0, (4, 2.5)) if dashed else "-", solid_capstyle="butt")
    ax.annotate("", xy=pts[-1], xytext=pts[-2],
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                shrinkA=0, shrinkB=0, mutation_scale=11))
    if label:
        (x1, y1), (x2, y2) = pts[lidx], pts[lidx + 1]
        ax.text((x1 + x2) / 2 + loff[0], (y1 + y2) / 2 + loff[1], label,
                ha=ha, va="bottom", fontsize=fs, color=color, style="italic")


# experiment membership: shared / pick-and-place only / tracking only
SH, SH_D = "#dde9f7", "#2f5f9e"
PP, PP_D = "#fbe0ef", "#b0307f"
TR, TR_D = "#d3f1ea", "#11806e"
NEUTRAL = "#f2f2f2"


def hatched(ax, x, y, w, h, ec):
    """Hatching over a box: frozen weights."""
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.35",
                                fc="none", ec=ec, lw=0, hatch="///",
                                alpha=0.22, zorder=2))


def region(ax, x, y, w, h, color, title):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.6",
                                fc="none", ec=color, lw=1.1, ls=":"))
    ax.text(x + 0.5, y + h - 0.5, title, fontsize=10, fontweight="bold",
            color=color, va="top")


def page_architecture(pdf):
    fig, ax = new_page(
        "Soft-arm VLA architecture — pick-and-place and trajectory tracking",
        "one ResNet18 backbone and action trunk shared by both experiments (blue); "
        "pick-and-place adds language and multimodal fusion (magenta); "
        "tracking pools the image to one vector (teal)")

    # ---- inputs -----------------------------------------------------------
    box(ax, 0.6, 49.8, 11.8, 13.6, "CAMERA IMAGES",
        "112 × 112 RGB, stacked\nchannel-wise\n\npick-and-place:\nexternal + wrist view\n"
        "3 past frames\n(t, t−0.2 s, t−0.4 s) → 18 ch\n\ntracking:\n2 side views, 1 past +\n"
        "3 GOAL frames (t, +0.25,\n+0.5, +0.75 s) → 24 ch",
        fc=NEUTRAL, ec="#333333", fs=7.4)
    box(ax, 0.6, 27.0, 11.8, 11.0, "INSTRUCTION",
        "pick-and-place only\n\n\"pick up the red apple\nand drop it into the bowl\"\n"
        "raw text, any wording;\ntraining: 4 paraphrases\nper object",
        fc=NEUTRAL, ec=PP_D, fs=7.5)
    box(ax, 0.6, 7.6, 11.8, 10.0, "PREVIOUS COMMAND",
        "both experiments\n\n7 pressures, bar\n[A1 A2 A3 B1 B2 B3 grip]\nstandardised (train mean / std)",
        fc=NEUTRAL, ec="#333333", fs=7.5)

    # ---- vision -----------------------------------------------------------
    region(ax, 14.2, 43.0, 57.0, 22.8, SH_D,
           "VISION ENCODER — ImageNet ResNet18 backbone (shared)")
    y0, h = 56.2, 7.4
    stages = [
        ("stem", "conv1 7×7/2 → 64\ninflated to 18 | 24 ch\n(RGB filters tiled ÷ n)\nBN · ReLU · maxpool"),
        ("layer1", "2 basic blocks\n64 ch · 28×28"),
        ("layer2", "2 basic blocks\n128 ch · 14×14"),
        ("layer3", "2 basic blocks\n256 ch · 7×7"),
        ("layer4", "2 basic blocks\n512 ch · 4×4"),
    ]
    xs = [14.8, 23.4, 30.0, 36.6, 48.0]
    ws = [7.8, 5.8, 5.8, 6.2, 6.2]
    for (t, b), x, w in zip(stages, xs, ws):
        box(ax, x, y0, w, h, t, b, fc=SH, ec=SH_D, fs=7.4)
    for a, b in ((0, 1), (1, 2), (2, 3)):
        wire(ax, [(xs[a] + ws[a], y0 + h / 2), (xs[b], y0 + h / 2)], SH_D)
    wire(ax, [(12.4, y0 + h / 2), (14.8, y0 + h / 2)])
    # layer3 -> layer4 directly, in both experiments (FiLM was removed from
    # the pick-and-place model on 2026-10-03 after the ablation study)
    wire(ax, [(42.8, y0 + h / 2), (48.0, y0 + h / 2)], SH_D)
    # pick-and-place taps f3 (layer3 output) and f4 (layer4 output)
    wire(ax, [(54.2, y0 + h / 2), (56.7, y0 + h / 2)], PP_D)
    wire(ax, [(54.2, 56.9), (59.8, 56.9)], TR_D, lw=1.5)
    box(ax, 59.8, 56.2, 10.8, 7.4, "GLOBAL AVERAGE POOL",
        "tracking only\n\nf₄ 512×4×4 → mean → 512\nLinear 512→256 · LN · GELU\n= vₜ (256)",
        fc=TR, ec=TR_D, fs=7.4, tfs=8.4)
    tag(ax, 65.2, 55.5, "vₜ", TR_D)
    # pick-and-place: attention pool + spatial tokens on f3 and f4
    box(ax, 14.8, 47.0, 21.2, 7.0, "LANGUAGE-QUERIED ATTENTION POOL, on f₃ and f₄",
        "query q = W·ℓ (256→128);  keys 1×1 conv C→128;  values 1×1 conv C→C\n"
        "α = softmax(q·k / √128) over the 49 | 16 cells\n"
        "[Σα·v₃ (256) | Σα·v₄ (512) | mean f₄ (512)] → Linear 256 · LN · GELU = p",
        fc=PP, ec=PP_D, fs=7.4, tfs=8.4, align="left")
    box(ax, 37.2, 47.0, 20.8, 7.0, "SPATIAL TOKENS, from f₃ and f₄",
        "f₃ 256×7×7: 49 cells, 1×1 conv 256→256 + position\n"
        "f₄ 512×4×4: 16 cells, 1×1 conv 512→256 + position\n"
        "→ 65 visual tokens × 256 = c",
        fc=PP, ec=PP_D, fs=7.4, tfs=8.4, align="left")
    wire(ax, [(45.4, y0 + h / 2), (45.4, 54.0)], PP_D, label="f₃", lidx=0,
         loff=(0.8, -0.55), fs=8)
    wire(ax, [(56.7, y0 + h / 2), (56.7, 54.0)], PP_D, label="f₄", lidx=0,
         loff=(0.8, -0.55), fs=8)
    wire(ax, [(45.4, 55.0), (30.0, 55.0), (30.0, 54.0)], PP_D)
    wire(ax, [(56.7, 55.45), (33.0, 55.45), (33.0, 54.0)], PP_D)
    tag(ax, 16.2, 54.75, "ℓ", PP_D)
    tag(ax, 25.4, 46.3, "p", PP_D)
    tag(ax, 47.6, 46.3, "c", PP_D)

    # ---- language (pick-and-place only) -----------------------------------
    region(ax, 14.2, 21.6, 31.4, 20.4, PP_D,
           "LANGUAGE ENCODER — pick-and-place only")
    box(ax, 14.8, 22.6, 12.4, 16.6, "MiniLM-L6  ❄ frozen",
        "sentence-transformers/\nall-MiniLM-L6-v2\n22.7 M params, not trained\n\n"
        "reads the raw text with its\nown WordPiece tokenizer\n(30 522 tokens)\n"
        "6 layers · 384-d · ≤ 24 tokens\ncached per sentence",
        fc=PP, ec=PP_D, fs=7.4)
    hatched(ax, 14.8, 22.6, 12.4, 16.6, PP_D)
    box(ax, 29.2, 32.6, 15.2, 6.6, "sentence adapter",
        "mean-pool, L2-norm (384)\nLinear 384→256 · LN · GELU\n= language code ℓ (256)",
        fc=PP, ec=PP_D, fs=7.6, tfs=8.6)
    box(ax, 29.2, 22.6, 15.2, 7.8, "token adapter",
        "24 contextual word states × 384\nLinear 384→256 · LN\n= word tokens w (24 × 256)\n(padding masked)",
        fc=PP, ec=PP_D, fs=7.6, tfs=8.6)
    wire(ax, [(12.4, 32.5), (14.8, 32.5)], PP_D)
    wire(ax, [(27.2, 35.9), (29.2, 35.9)], PP_D)
    wire(ax, [(27.2, 26.5), (29.2, 26.5)], PP_D)
    tag(ax, 45.0, 35.9, "ℓ", PP_D)
    tag(ax, 45.0, 26.5, "w", PP_D)

    # ---- state (shared) ---------------------------------------------------
    box(ax, 14.8, 8.2, 13.0, 8.8, "STATE ENCODER (MLP)",
        "Linear 7→128 · LN · GELU\nLinear 128→256 · LN · GELU\n= state code s (256)",
        fc=SH, ec=SH_D, fs=7.8)
    wire(ax, [(12.4, 12.6), (14.8, 12.6)])
    tag(ax, 28.5, 12.6, "s", SH_D)

    # ---- fusion transformer (pick-and-place only) --------------------------
    region(ax, 47.4, 5.6, 22.8, 36.4, PP_D,
           "MULTIMODAL FUSION — pick-and-place only")
    rows = [("[CLS]  ×1     ← learned", None, PP_D),
            ("[state]  ×1     ← s", "s", SH_D),
            ("[sentence]  ×1     ← ℓ", "ℓ", PP_D),
            ("[words]  ×24     ← w", "w", PP_D),
            ("[layer3 cells]  ×49     ← c", "c", PP_D),
            ("[layer4 cells]  ×16     ← c", "c", PP_D)]
    ty = 37.5
    for t, tg, tc in rows:
        box(ax, 49.4, ty, 20.0, 1.95, t, fc=PP, ec=PP_D, tfs=8.0, lw=1.0)
        if tg:
            tag(ax, 48.4, ty + 0.97, tg, tc, r=0.55)
        ty -= 2.3
    ax.text(59.4, ty + 1.85, "92 tokens × 256  + token-type embedding (6 types)",
            ha="center", va="top", fontsize=7.8, color=PP_D)
    box(ax, 49.4, 16.6, 20.0, 6.8, "2 × Transformer encoder layer",
        "pre-norm · 4 heads · width 256 · FF 512\nGELU · dropout 0.1 · padding masked\n"
        "every token attends to every other",
        fc=PP, ec=PP_D, fs=7.6)
    box(ax, 49.4, 6.6, 20.0, 7.6, "vision latent",
        "[CLS] output → LN (256)\n[ p | CLS ] 512 → Linear 256 · LN · GELU\n= v (256)",
        fc=PP, ec=PP_D, fs=7.6, tfs=8.6)
    wire(ax, [(59.4, ty + 0.9), (59.4, 23.4)], PP_D)
    wire(ax, [(59.4, 16.6), (59.4, 14.2)], PP_D, label="CLS", lidx=0,
         loff=(0.9, -0.6), ha="left")
    tag(ax, 48.4, 10.4, "p", PP_D, r=0.55)
    wire(ax, [(48.95, 10.4), (49.4, 10.4)], PP_D)

    # ---- latent (shared, different width) ---------------------------------
    box(ax, 71.6, 6.0, 7.4, 26.4, "LATENT",
        "pick-and-place:\n[ v | s | ℓ ] = 768\n\ntracking:\n[ vₜ | s ] = 512\n\n"
        "v   vision 256\nvₜ  pooled vision 256\ns   state 256\nℓ   language 256\n\n"
        "(latent replay would\nstore these vectors)",
        fc=SH, ec=SH_D, fs=7.4)
    wire(ax, [(69.4, 10.4), (71.6, 10.4)], PP_D, label="v", lidx=0,
         loff=(0, 0.1))
    tag(ax, 70.7, 16.0, "s", SH_D, r=0.55)
    wire(ax, [(71.25, 16.0), (71.6, 16.0)], SH_D)
    tag(ax, 70.7, 21.0, "ℓ", PP_D, r=0.55)
    wire(ax, [(71.25, 21.0), (71.6, 21.0)], PP_D)
    tag(ax, 70.7, 26.5, "vₜ", TR_D, r=0.55)
    wire(ax, [(71.25, 26.5), (71.6, 26.5)], TR_D)

    # ---- grounding head (pick-and-place, training only) -------------------
    box(ax, 71.6, 35.6, 7.4, 13.4, "GROUNDING\nHEAD",
        "\npick-and-place,\ntraining only\n\nLinear 256→128\nGELU · Linear 128→2\n\n"
        "= named object's\nstart (x, y) ÷ 0.05 m\nloss weight 1e-4",
        fc="#ffffff", ec=PP_D, dashed=True, fs=7.2, tfs=8.4)
    tag(ax, 75.3, 34.0, "v", PP_D, r=0.55)
    wire(ax, [(75.3, 34.55), (75.3, 35.6)], PP_D, dashed=True)

    # ---- trunk + head (shared) --------------------------------------------
    region(ax, 80.4, 10.8, 19.0, 52.4, SH_D, "ACTION TRUNK + HEAD (shared)")
    box(ax, 81.2, 55.2, 17.4, 5.8, "fusion",
        "Linear 768→512 · LN · GELU  (pick-and-place)\nLinear 512→512 · LN · GELU  (tracking)",
        fc=SH, ec=SH_D, fs=7.4)
    box(ax, 81.2, 44.2, 17.4, 9.2, "4 × residual MLP block",
        "x ← x + Drop(Linear 1024→512(\n   Drop(GELU(Linear 512→1024(LN(x))))))\n"
        "width 512 · dropout 0.1 · 4.2 M params",
        fc=SH, ec=SH_D, fs=7.6)
    box(ax, 81.2, 33.6, 17.4, 8.4, "action head",
        "LN · Linear 512→256 · GELU\nLinear 256→70 → reshape 10 × 7\n× σ + μ (training action scaling)",
        fc=SH, ec=SH_D, fs=7.6)
    box(ax, 81.2, 21.6, 17.4, 9.8, "OUTPUT  Δp  (10 × 7, bar)",
        "pressure CHANGES for the next 10\npolicy steps (0.5 s at 20 Hz)\n\n"
        "p_{t+k} = p_t + Σ Δp\nfirst 5 executed (rate-limited),\nthen the policy is asked again",
        fc=SH, ec=SH_D, fs=7.6)
    box(ax, 81.2, 11.6, 17.4, 7.8, "SOFT ARM (MuJoCo)",
        "6 arm chambers (2 modules × 3) + gripper\npick-and-place: gripper line used\n"
        "tracking: gripper held closed",
        fc=NEUTRAL, ec="#333333", fs=7.4)
    wire(ax, [(79.0, 19.0), (79.8, 19.0), (79.8, 58.1), (81.2, 58.1)], SH_D,
         lw=1.4)
    for ya, yb in ((55.2, 53.4), (44.2, 42.0), (33.6, 31.4), (21.6, 19.4)):
        wire(ax, [(89.9, ya), (89.9, yb)], SH_D)
    wire(ax, [(89.9, 11.6), (89.9, 4.4), (6.4, 4.4), (6.4, 7.6)], "#777777",
         dashed=True, label="next step: the commanded pressures become the "
         "previous command; new camera frames", lidx=1, loff=(0, 0.15))

    # ---- legend -----------------------------------------------------------
    from matplotlib.patches import Circle
    ly = 1.3
    items = [(SH, SH_D, "shared by both experiments", None, False),
             (PP, PP_D, "pick-and-place only", None, False),
             (TR, TR_D, "trajectory tracking only", None, False),
             (PP, PP_D, "frozen (MiniLM)", "///", False),
             ("#ffffff", PP_D, "training only", None, True)]
    x = 0.6
    for fc, ec, lab, hatch, dashed in items:
        ax.add_patch(FancyBboxPatch((x, ly - 0.8), 1.5, 1.6,
                                    boxstyle="round,pad=0.02,rounding_size=0.2",
                                    fc=fc, ec=ec, lw=1.3, hatch=hatch,
                                    ls=(0, (3, 2)) if dashed else "-"))
        ax.text(x + 2.0, ly, lab, va="center", fontsize=8.6)
        x += 3.2 + 0.52 * len(lab)
    tag(ax, x + 0.6, ly, "ℓ", PP_D, r=0.55)
    ax.text(x + 1.5, ly, "connector: same letter = same tensor\n(ℓ language, "
            "w words, s state, p pooled, c cells, v / vₜ vision)",
            va="center", fontsize=7.8)
    ax.text(99.4, 2.95,
            "trainable parameters — pick-and-place: 18.46 M (+22.7 M frozen MiniLM)   "
            "·   tracking: 16.03 M (vision 11.37 M, state 0.03 M, trunk 4.62 M)",
            ha="right", va="center", fontsize=8.2, color="#444444")
    pdf.savefig(fig)
    fig.savefig(HERE / "vla_architecture_p1.png", dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Page 2: training and deployment
# ---------------------------------------------------------------------------

def page_training(pdf):
    fig, ax = new_page(
        "Continual training with replay, and the closed-loop test",
        "demonstrations with all three objects on the table (spread layout) · "
        "cube → apple → can, one object per stage · best model: raw replay, "
        "8 000 samples per finished object")
    T, O = C["trunk_d"], C["out_d"]
    fs, tfs = 12.2, 13.5

    # ---- left column: data and test ---------------------------------------
    box(ax, 0.8, 49.0, 22.0, 15.0, "EXPERT DEMONSTRATIONS",
        "scripted pick-and-drop expert\n(min-jerk waypoints, Jacobian\npressure control)\n\n"
        "all 3 objects on the table, ≥ 40 mm\napart where possible; the\ninstruction names the target\n\n"
        "163 / 164 / 170 demos\n(cube / apple / can), 380 frames\neach at 20 Hz",
        fc=C["input"], ec="#333333", fs=fs, tfs=tfs)
    box(ax, 0.8, 30.5, 22.0, 15.0, "ONE TRAINING SAMPLE",
        "3 frames × 2 views (18 × 112 × 112)\n+ previous pressures (+0.01 bar noise)\n"
        "+ one of 4 paraphrases (raw text)\n\ntarget: next 10 pressure changes\n\n"
        "grounding target: the named\nobject's start (x, y), reach phase\nonly (first 40% of the episode)",
        fc=C["input"], ec="#333333", fs=fs, tfs=tfs)
    wire(ax, [(11.8, 49.0), (11.8, 45.5)])
    box(ax, 0.8, 3.0, 22.0, 24.5, "CLOSED-LOOP TEST",
        "(standard protocol)\n\nall 3 objects on the table;\n3 instructions per episode,\nrandom order\n\n"
        "every 5 steps (0.25 s): images +\npressures + instruction → policy\n→ 10 changes, execute 5 at 20 Hz\n\n"
        "22 s per instruction, then the arm\nis driven home (3 s) and delivered\nobjects leave the bowl\n\n"
        "3 layout sets × 10 episodes",
        fc=C["out"], ec=O, fs=fs, tfs=tfs)

    # ---- stages -------------------------------------------------------------
    sx, sw, sy, sh = [26.0, 50.2, 74.4], 22.4, 55.0, 9.0
    stages = [("STAGE 1 — cube", "45 epochs · cube demos only\nnothing to rehearse yet\nall 18.46 M parameters train"),
              ("STAGE 2 — apple", "20 epochs · apple demos only\nrehearse: cube\nall parameters train"),
              ("STAGE 3 — can", "20 epochs · can demos only\nrehearse: cube + apple\nall parameters train")]
    for x, (t, b) in zip(sx, stages):
        box(ax, x, sy, sw, sh, t, b, fc=C["trunk"], ec=T, fs=fs, tfs=tfs)
    wire(ax, [(22.8, 59.5), (26.0, 59.5)])
    wire(ax, [(48.4, 59.5), (50.2, 59.5)], T)
    wire(ax, [(72.6, 59.5), (74.4, 59.5)], T)

    # ---- replay buffer ------------------------------------------------------
    box(ax, 26.0, 41.5, 70.8, 6.4, "REPLAY BUFFER (raw)",
        "8 000 training frames per finished object (indices into the demos),\n"
        "replayed through the whole network with the same augmentation",
        fc=C["latent"], ec=O, fs=fs, tfs=tfs)
    for x, lab, up in ((37.2, "store 8 000 cube", False),
                       (58.0, "store 8 000 apple", False),
                       (66.4, "rehearse cube", True),
                       (85.6, "rehearse cube + apple", True)):
        pts = [(x, 47.9), (x, sy)] if up else [(x, sy), (x, 47.9)]
        wire(ax, pts, O, label=lab, lidx=0, loff=(0.6, -0.9), ha="left", fs=11.5)

    # ---- one training step --------------------------------------------------
    box(ax, 26.0, 22.0, 70.8, 16.2, "ONE TRAINING STEP (stages 2 and 3)",
        "64 NEW samples (current object)   +   32 REHEARSED samples (from the buffer)\n\n"
        "L = MSE(Δp̂, Δp)_new + 1e-4·MSE(x̂y, xy)_new + MSE(Δp̂, Δp)_replay + 1e-4·MSE(x̂y, xy)_replay\n\n"
        "AdamW · lr 3e-4 · weight decay 1e-5 · gradient clip 1.0 · last epoch kept\n\n"
        "Why both terms matter: within a stage every NEW sample names the same object,\n"
        "so its position can be read from the image alone. Only the REHEARSED samples\n"
        "pair a different instruction with a different object's position — that contrast\n"
        "is what makes the network read the words.",
        fc="#ffffff", ec=T, fs=fs, tfs=tfs)
    wire(ax, [(61.4, 41.5), (61.4, 38.2)], O)

    # ---- result -------------------------------------------------------------
    box(ax, 26.0, 3.0, 70.8, 15.8, "RESULTS — standard protocol, 3 layout sets × 10 episodes",
        "best model (raw replay 8 000): 84/90 correct (93%) · 0 wrong-object picks\n"
        "cube 29/30 · apple 29/30 · can 26/30 · after stage 1: 28/30 · after stage 2: 57/60\n\n"
        "raw replay 2 000: 80/90 (89%)  ·  latent replay 2 000: 56/90 (62%, 9 wrong objects)\n"
        "unseen wording: paraphrase 92% · noun only 92% · colour only 71% · synonyms 60%\n"
        "stage-1 model asked for the apple or can: fetches the cube (29 of 31)",
        fc=C["out"], ec=O, fs=fs, tfs=tfs)
    wire(ax, [(22.8, 11.0), (26.0, 11.0)], O)
    pdf.savefig(fig)
    fig.savefig(HERE / "vla_architecture_p2.png", dpi=110)
    plt.close(fig)


def main():
    with PdfPages(OUT) as pdf:
        page_architecture(pdf)
        page_training(pdf)
        info = pdf.infodict()
        info["Title"] = "Soft-arm VLA architecture: pick-and-place and trajectory tracking"
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
