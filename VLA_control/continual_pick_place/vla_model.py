"""
The language-conditioned pick-and-drop VLA, split for latent replay.

It is pick_place_vla_model.py with two changes, and only two:

    1. a LANGUAGE branch alongside vision and proprioception, so the task is
       named rather than guessed;
    2. the forward pass is cut in half at the point where the three latents
       meet, into `encode()` and `head()`, so a stored latent can be pushed
       through the second half on its own.

Everything else - the ImageNet-pretrained ResNet18 with its first conv
inflated across the stacked frames, the 256-d projections, the 512-wide
fusion trunk, four residual blocks, a head that predicts CHUNK_STEPS
pressure CHANGES - is unchanged and imported, not copied. That matters: the
continual result has to be about continual learning, not about a network
that also happens to be different.

                                images            state          instruction
                                  |                 |                 |
                        ResNet18 (inflated)     MLP 7->256     embed+mean->256
                                  |                 |                 |
                                 256               256               256
                                  +--------- concat (768) ------------+
                                            |
                                     <-- the REPLAY LAYER
                                            |
                                  fusion 768 -> 512, 4 residual blocks
                                            |
                                     head -> [chunk, 7] pressure deltas, bar

WHY THE CUT IS HERE
-------------------
Latent replay (Pellegrini et al., 2020) stores activations from some
intermediate layer instead of raw inputs, and replays them through the rest
of the network. The concatenation point is the natural place for this model:
it is the narrowest waist in the graph (768 floats against 3x2x112x112 =
225 792 for the images it summarises, a 147x saving), and it is the first
point where all three modalities are present, so a replayed latent is a
complete description of one training sample.

THE PRICE, STATED UP FRONT
--------------------------
A stored latent is only a valid training target while the encoders that
produced it still produce it. So from the second task onward the encoders
are FROZEN and only fusion, the residual blocks and the head keep learning
(`freeze_encoders()`). That is the method, not a shortcut: it is what makes
the replayed latents exact rather than stale. What it costs is the ability
to learn new VISUAL features for a new object - the can has to be
representable by features trained on the cube. Whether it is, is the
experiment.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def _load_base():
    """The pick-and-place network module, loaded BY PATH.

    Its classes are registered under `pick_place_vla_model`, so it is loaded
    by path under that name rather than imported from sys.path.
    """

    source = HERE / "pick_place_vla_model.py"
    spec = importlib.util.spec_from_file_location("pick_place_vla_model",
                                                  source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# the encoder zoo, the residual block and the layout convention, shared with
# the pick-and-place policy rather than reimplemented
base = _load_base()

from language import MAX_TOKENS, LanguageEncoder, Vocabulary  # noqa: E402

ACTION_DIM = base.ACTION_DIM              # 7 pressure lines
STATE_DIM = base.STATE_DIM                # the previous 7 pressures
IMAGE_SIZE = base.IMAGE_SIZE              # 112
NUM_VIEWS = base.NUM_VIEWS                # external + wrist
FRAME_HISTORY = base.FRAME_HISTORY        # 3
FRAME_STRIDE = base.FRAME_STRIDE          # 4 policy steps apart
POLICY_DT = base.POLICY_DT                # 0.05 s
CHUNK_STEPS = base.CHUNK_STEPS            # 10 predicted
CHUNK_EXECUTE = base.CHUNK_EXECUTE        # 5 executed
PRESSURE_LABELS = base.PRESSURE_LABELS
VISUAL_ENCODERS = base.VISUAL_ENCODERS

stack_frames = base.stack_frames
count_parameters = base.count_parameters

FEATURE_DIM = 256                         # per modality
LATENT_DIM = 3 * FEATURE_DIM              # 768: vision + state + language


# ---------------------------------------------------------------------------
# Language-conditioned visual encoder
# ---------------------------------------------------------------------------

class FiLMResNetEncoder(nn.Module):
    """ImageNet ResNet18 whose features are steered by the instruction
    BEFORE they are pooled.

    THE PROBLEM THIS EXISTS TO FIX
    ------------------------------
    The plain encoder global-average-pools its 4x4x512 map into one vector
    and only then is the language code concatenated on. Pooling sums away
    every "which object is where" cue, so the instruction is being asked to
    select a location using a representation from which location has already
    been deleted. It cannot, and behaviour cloning finds the only remaining
    solution: ignore the sentence and output the average of the plausible
    targets. Measured, with three objects on the table: the policy delivered
    the named object at chance (3/9) and reached into the gap between objects
    90% of the time.

    TWO PLACES LANGUAGE GETS IN, BOTH BEFORE POOLING
    ------------------------------------------------
    * FiLM (Perez et al., 2018) at the last two residual stages: the
      instruction produces a per-channel scale and shift, so "red apple" can
      amplify the channels that respond to round and red and suppress the
      others while the feature map is still 7x7 and 4x4.
    * a language-QUERIED attention pool: the instruction forms a query, each
      spatial cell a key, and the map is pooled by the resulting weights
      instead of uniformly. This is the part that can actually say "that
      corner of the image", which uniform averaging structurally cannot.

    The mean-pooled vector is kept alongside the attended one, so the policy
    still has an unconditional summary of the scene to fall back on.
    """

    def __init__(self, in_channels, lang_dim=256, output_dim=256,
                 attention_dim=128, film=True, attention=True):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        # ablation switches (ABLATIONS): film=False leaves layer3/layer4
        # unmodulated; attention=False replaces the language-queried pools
        # by plain spatial means of the same maps
        self.use_film, self.use_attention = film, attention

        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        groups = in_channels // 3
        old = net.conv1.weight.data
        conv = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)
        # same inflation as the plain encoder: tile the RGB filters over the
        # frame/view groups and divide, so the pretrained edge and colour
        # response survives
        conv.weight.data = old.repeat(1, groups, 1, 1) / groups
        net.conv1 = conv

        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1, self.layer2 = net.layer1, net.layer2
        self.layer3, self.layer4 = net.layer3, net.layer4

        # FiLM generators. Zero-initialised so the network STARTS as the
        # plain pretrained encoder (gamma = 1, beta = 0) and has to earn any
        # dependence on the instruction.
        if film:
            self.film3 = nn.Linear(lang_dim, 2 * 256)
            self.film4 = nn.Linear(lang_dim, 2 * 512)
            for layer in (self.film3, self.film4):
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)

        # Attention is run at BOTH surviving resolutions. At 112x112 input,
        # layer4's map is only 4x4 - each cell covers 28x28 pixels, which is
        # about the size of a whole object, so it can barely distinguish two
        # objects standing 50 mm apart. layer3's 7x7 map (16x16 pixels per
        # cell) is where the "which one" question is actually answerable;
        # layer4 carries the more semantic "what is it" signal. Both are
        # pooled by the instruction and concatenated.
        if attention:
            self.query3 = nn.Linear(lang_dim, attention_dim)
            self.key3 = nn.Conv2d(256, attention_dim, 1)
            self.value3 = nn.Conv2d(256, 256, 1)
            self.query4 = nn.Linear(lang_dim, attention_dim)
            self.key4 = nn.Conv2d(512, attention_dim, 1)
            self.value4 = nn.Conv2d(512, 512, 1)
        self.attention_dim = attention_dim

        # attended layer3 + attended layer4 + mean layer4; without attention
        # the two attended pools become the means of the same maps
        self.projection = nn.Sequential(
            nn.Linear(256 + 512 + 512, output_dim), nn.LayerNorm(output_dim),
            nn.GELU())

        mean = torch.tensor([0.485, 0.456, 0.406]).repeat(groups)
        std = torch.tensor([0.229, 0.224, 0.225]).repeat(groups)
        self.register_buffer("pixel_mean", mean.view(1, -1, 1, 1))
        self.register_buffer("pixel_std", std.view(1, -1, 1, 1))

    @staticmethod
    def _film(x, params):
        gamma, beta = params.chunk(2, dim=1)
        return x * (1.0 + gamma[..., None, None]) + beta[..., None, None]

    def _attend(self, feature, language, query, key, value):
        b, c, h, w = feature.shape
        keys = key(feature).flatten(2)                      # [B, d, N]
        values = value(feature).flatten(2)                  # [B, c', N]
        q = query(language).unsqueeze(1)                    # [B, 1, d]
        weights = torch.softmax((q @ keys) / (self.attention_dim ** 0.5),
                                dim=-1)                     # [B, 1, N]
        return (values * weights).sum(-1), weights.view(b, h, w)

    def features(self, image, language):
        x = (image - self.pixel_mean) / self.pixel_std
        x = self.layer2(self.layer1(self.stem(x)))
        if not self.use_film:
            f3 = self.layer3(x)
            return f3, self.layer4(f3)
        f3 = self._film(self.layer3(x), self.film3(language))
        f4 = self._film(self.layer4(f3), self.film4(language))
        return f3, f4

    def pool(self, f3, f4, language):
        """The pooled 256-d vision vector."""
        if self.use_attention:
            a3, _ = self._attend(f3, language, self.query3, self.key3,
                                 self.value3)
            a4, _ = self._attend(f4, language, self.query4, self.key4,
                                 self.value4)
        else:
            a3, a4 = f3.mean((2, 3)), f4.mean((2, 3))
        return self.projection(torch.cat([a3, a4, f4.mean((2, 3))], dim=1))

    def forward(self, image, language):
        f3, f4 = self.features(image, language)
        return self.pool(f3, f4, language)

    def attention_map(self, image, language, stage=3):
        """The pooling weights, [B, H, W] - where the instruction looked.
        Diagnostic only; nothing in training calls it."""

        if not self.use_attention:
            raise ValueError("this model was built without attention")
        f3, f4 = self.features(image, language)
        if stage == 3:
            return self._attend(f3, language, self.query3, self.key3,
                                self.value3)[1]
        return self._attend(f4, language, self.query4, self.key4,
                            self.value4)[1]


MINILM_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MINILM_DIM = 384
MINILM_TOKENS = 24


class MiniLMLanguageEncoder(nn.Module):
    """A frozen pretrained sentence encoder with a small trainable adapter.

    WHY PRETRAINED, AND WHY FROZEN
    ------------------------------
    The learned 64-d embedding has to discover from the demonstrations alone
    that "red apple" and "blue cube" name different objects; the only signal
    it gets is the action loss. MiniLM already knows it: paraphrases of one
    instruction sit at cosine ~0.93, instructions for different objects at
    ~0.55-0.67. Freezing it keeps the continual-learning result honest - the
    language representation cannot drift or forget between stages, and it
    costs no trainable parameters. Only the adapters learn.

    The MiniLM weights live OUTSIDE the module tree, so they are neither in
    the state dict (checkpoints stay the same size and load strictly) nor in
    model.parameters() (nothing can accidentally train them). The input is
    the raw instruction text, read with MiniLM's own 30k-token tokenizer -
    there is no project vocabulary in this path, so any English wording
    reaches the model. MiniLM's outputs are cached per sentence, so the text
    model runs once per distinct sentence, not once per sample.

    Returns the sentence code [B, output_dim] and the per-word tokens
    [B, MINILM_TOKENS, token_dim] with their mask, for the fusion transformer.
    """

    def __init__(self, output_dim=256, token_dim=256, adapters=True):
        super().__init__()
        # ablation switch (ABLATIONS "adapters"): adapters=False keeps the
        # same maps - Linear -> LayerNorm (-> GELU) - but FROZEN at their
        # random initialisation, as buffers, with a parameter-free LayerNorm:
        # MiniLM's outputs reach the policy through a fixed random projection
        # of the same size and scale, and nothing on the language path learns
        self.use_adapters = adapters
        if adapters:
            self.sentence_adapter = nn.Sequential(
                nn.Linear(MINILM_DIM, output_dim), nn.LayerNorm(output_dim),
                nn.GELU())
            self.token_adapter = nn.Sequential(
                nn.Linear(MINILM_DIM, token_dim), nn.LayerNorm(token_dim))
        else:
            sentence = nn.Linear(MINILM_DIM, output_dim)    # default init,
            token = nn.Linear(MINILM_DIM, token_dim)        # then frozen
            self.register_buffer("sentence_weight", sentence.weight.detach().clone())
            self.register_buffer("sentence_bias", sentence.bias.detach().clone())
            self.register_buffer("token_weight", token.weight.detach().clone())
            self.register_buffer("token_bias", token.bias.detach().clone())
        self._cache = {}
        self.__dict__["_text"] = None       # (tokenizer, model), not a child

    def _project(self, sentence, hidden):
        """sentence [B, 384], word tokens [B, T, 384] -> the 256-d codes."""
        if self.use_adapters:
            return self.sentence_adapter(sentence), self.token_adapter(hidden)
        f = nn.functional
        s = f.linear(sentence, self.sentence_weight, self.sentence_bias)
        s = f.gelu(f.layer_norm(s, s.shape[-1:]))
        t = f.linear(hidden, self.token_weight, self.token_bias)
        return s, f.layer_norm(t, t.shape[-1:])

    def _text_model(self):
        if self._text is None:
            from transformers import AutoModel, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(MINILM_NAME)
            model = AutoModel.from_pretrained(MINILM_NAME).eval()
            for p in model.parameters():
                p.requires_grad_(False)
            self.__dict__["_text"] = (tokenizer, model)
        return self._text

    @torch.no_grad()
    def _embed(self, texts):
        tokenizer, model = self._text_model()
        batch = tokenizer(texts, padding="max_length", truncation=True,
                          max_length=MINILM_TOKENS, return_tensors="pt")
        hidden = model(**batch).last_hidden_state            # [n, T, 384]
        mask = batch["attention_mask"].float()
        sentence = ((hidden * mask[..., None]).sum(1)
                    / mask.sum(1, keepdim=True))
        sentence = nn.functional.normalize(sentence, dim=-1)
        return hidden, mask, sentence

    @staticmethod
    def normalise(text):
        """Lower case, words only - what a sentence becomes after a round
        trip through the project vocabulary, so a raw training sentence and
        its token ids give MiniLM exactly the same input."""
        from language import tokenize
        return " ".join(tokenize(text))

    def forward(self, texts, mask=None):
        """`texts`: the instructions as raw strings, [B]. They are read with
        MiniLM's own 30k-token vocabulary; there is no project vocabulary in
        this path. `mask` is ignored (kept for the common call signature)."""

        if torch.is_tensor(texts):
            raise TypeError("MiniLM models take the instruction as text, not "
                            "token ids - build the dataset with raw_text=True")
        keys = [self.normalise(t) for t in texts]
        missing = [k for k in dict.fromkeys(keys) if k not in self._cache]
        if missing:
            hidden, tmask, sentence = self._embed(missing)
            for i, k in enumerate(missing):
                self._cache[k] = (hidden[i], tmask[i], sentence[i])
        device = (self.sentence_adapter[0].weight.device if self.use_adapters
                  else self.sentence_weight.device)
        hidden = torch.stack([self._cache[k][0] for k in keys]).to(device)
        tmask = torch.stack([self._cache[k][1] for k in keys]).to(device)
        sentence = torch.stack([self._cache[k][2] for k in keys]).to(device)
        sentence_code, word_tokens = self._project(sentence, hidden)
        return sentence_code, word_tokens, tmask


class VLAFusionEncoder(FiLMResNetEncoder):
    """FiLM-conditioned ResNet18 followed by a multimodal transformer.

    The FiLM stages and the language-queried attention pool are inherited
    unchanged: that is the early, before-pooling conditioning that took the
    cluttered stage-1 success from 2/10 to 9/10. On top of it, the spatial
    maps are NOT pooled away but kept as tokens -

        [CLS] [state] [sentence] [word x24] [layer3 cell x49] [layer4 cell x16]

    - and two transformer layers let every modality attend to every other:
    the words to the image cells that match them, the state to where the
    gripper is, the CLS token to all of it. The CLS output and the attended
    pool are projected together to the 256-d vision latent.
    """

    def __init__(self, in_channels, lang_dim=256, output_dim=256,
                 attention_dim=128, width=256, layers=2, heads=4,
                 film=True, attention=True, transformer=True, state=True):
        super().__init__(in_channels, lang_dim=lang_dim,
                         output_dim=output_dim, attention_dim=attention_dim,
                         film=film, attention=attention)
        # ablation switches: transformer=False returns the pooled vector
        # alone; state=False drops the [state] token
        self.use_transformer, self.use_state = transformer, state
        if not transformer:
            return
        self.cell3 = nn.Conv2d(256, width, 1)
        self.cell4 = nn.Conv2d(512, width, 1)
        self.pos3 = nn.Parameter(torch.zeros(1, 49, width))
        self.pos4 = nn.Parameter(torch.zeros(1, 16, width))
        # one learned offset per token type: cls, state, sentence, word,
        # layer3 cell, layer4 cell
        self.type_embedding = nn.Parameter(torch.zeros(6, width))
        self.cls = nn.Parameter(torch.zeros(1, 1, width))
        for p in (self.pos3, self.pos4, self.type_embedding, self.cls):
            nn.init.normal_(p, std=0.02)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(width, heads, dim_feedforward=2 * width,
                                       dropout=0.1, batch_first=True,
                                       norm_first=True, activation="gelu"),
            num_layers=layers, enable_nested_tensor=False)
        self.fused_norm = nn.LayerNorm(width)
        self.fused_projection = nn.Sequential(
            nn.Linear(output_dim + width, output_dim),
            nn.LayerNorm(output_dim), nn.GELU())

    def forward(self, image, language, words, word_mask, state):
        f3, f4 = self.features(image, language)
        pooled = self.pool(f3, f4, language)
        if not self.use_transformer:
            return pooled

        b = image.shape[0]
        t = self.type_embedding
        head = [self.cls.expand(b, -1, -1) + t[0]]
        if self.use_state:
            head.append(state.unsqueeze(1) + t[1])
        head.append(language.unsqueeze(1) + t[2])
        tokens = torch.cat(head + [
            words + t[3],
            self.cell3(f3).flatten(2).transpose(1, 2) + self.pos3 + t[4],
            self.cell4(f4).flatten(2).transpose(1, 2) + self.pos4 + t[5],
        ], dim=1)
        keep = torch.ones(b, tokens.shape[1], device=image.device)
        keep[:, len(head):len(head) + words.shape[1]] = word_mask
        fused = self.transformer(tokens, src_key_padding_mask=keep == 0)
        cls = self.fused_norm(fused[:, 0])
        return self.fused_projection(torch.cat([pooled, cls], dim=1))


LANGUAGE_CONDITIONED = ("film", "minilm", "minilm_aux")
MINILM_ARCHS = ("minilm", "minilm_aux")

# Modules that can be switched off for the ablation study (MiniLM archs):
#   film         FiLM modulation of ResNet layer3 / layer4 by the sentence
#   attention    the language-queried spatial attention pools (-> plain means)
#   transformer  the multimodal fusion transformer over state / sentence /
#                word / image-cell tokens (-> the pooled vector alone)
#   state        the previously commanded pressures as an input (the
#                [state] token and the state code in the latent). The output
#                stays a pressure CHANGE added to the last command, so only
#                the network's view of that command is removed.
#   adapters     the trainable MiniLM adapters (sentence code + word tokens):
#                frozen at their random initialisation, so the frozen MiniLM
#                reaches the policy through a fixed random projection
ABLATIONS = ("film", "attention", "transformer", "state", "adapters")
# the switches VLAFusionEncoder takes (the others act elsewhere)
ENCODER_SWITCHES = ("film", "attention", "transformer", "state")


class ContinualVLA(nn.Module):
    def __init__(self, vocab_size, frame_history=FRAME_HISTORY,
                 state_dim=STATE_DIM, action_dim=ACTION_DIM, hidden_dim=512,
                 num_residual_blocks=4, chunk_steps=CHUNK_STEPS,
                 arch="resnet18", num_views=NUM_VIEWS,
                 feature_dim=FEATURE_DIM, embed_dim=64, ablate=()):
        super().__init__()

        self.ablate = tuple(a for a in ABLATIONS if a in ablate)
        unknown = set(ablate) - set(ABLATIONS)
        if unknown:
            raise ValueError(f"unknown ablation(s) {sorted(unknown)}; "
                             f"choose from {ABLATIONS}")
        if self.ablate and arch not in MINILM_ARCHS:
            raise ValueError("ablations are defined for the MiniLM archs")
        self.arch = arch
        self.num_views = num_views
        self.frame_history = frame_history
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_steps = chunk_steps
        self.vocab_size = vocab_size
        self.feature_dim = feature_dim
        self.embed_dim = embed_dim
        # vision + state code + sentence code (no state code if ablated)
        self.latent_dim = (2 if "state" in self.ablate else 3) * feature_dim

        # Filled from the FIRST task's training set and never touched again.
        # Restandardising at task 2 would change what the head's output means
        # and quietly invalidate every latent already in the buffer.
        self.register_buffer("state_mean", torch.zeros(state_dim))
        self.register_buffer("state_std", torch.ones(state_dim))
        self.register_buffer("action_mean", torch.zeros(action_dim))
        self.register_buffer("action_std", torch.ones(action_dim))
        self.register_buffer("normalisation_locked", torch.zeros((),
                                                                 dtype=torch.bool))

        in_channels = 3 * num_views * frame_history
        self.language_conditioned = arch in LANGUAGE_CONDITIONED
        if arch in MINILM_ARCHS:
            self.visual_encoder = VLAFusionEncoder(
                in_channels, lang_dim=feature_dim, output_dim=feature_dim,
                **{a: a not in self.ablate for a in ENCODER_SWITCHES})
        elif self.language_conditioned:
            self.visual_encoder = FiLMResNetEncoder(
                in_channels, lang_dim=feature_dim, output_dim=feature_dim)
        else:
            self.visual_encoder = VISUAL_ENCODERS[arch](in_channels,
                                                        feature_dim)
        self.state_encoder = (nn.Identity() if "state" in self.ablate
                              else base.MLPEncoder(state_dim, feature_dim))
        if arch in MINILM_ARCHS:
            # raw text in, no project vocabulary: vocab_size is unused here
            self.language_encoder = MiniLMLanguageEncoder(
                output_dim=feature_dim, adapters="adapters" not in self.ablate)
        else:
            self.language_encoder = LanguageEncoder(vocab_size, feature_dim,
                                                    embed_dim=embed_dim)

        self.fusion = nn.Sequential(
            nn.Linear(self.latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.residual_blocks = nn.ModuleList(
            [base.ResidualMLPBlock(hidden_dim)
             for _ in range(num_residual_blocks)])
        self.action_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, action_dim * chunk_steps),
        )

        # minilm_aux only: "where is the object the instruction names",
        # read from the vision latent - the part language has already
        # conditioned. Trained by an auxiliary loss (train_continual.py
        # --aux-weight) so the instruction HAS to change the visual features;
        # the policy itself never reads this output.
        self.has_grounding_head = arch == "minilm_aux"
        if self.has_grounding_head:
            self.grounding_head = nn.Sequential(
                nn.Linear(feature_dim, 128), nn.GELU(), nn.Linear(128, 2))

    # -- normalisation -------------------------------------------------------

    def set_normalisation(self, states, actions, force=False):
        """Fix the input/output scales from the first task's data.

        Silently ignored afterwards unless `force`, so a later stage cannot
        move the goalposts under the replay buffer.
        """

        if bool(self.normalisation_locked) and not force:
            return False
        for name, values in (("state", states), ("action", actions)):
            v = torch.as_tensor(np.asarray(values), dtype=torch.float32)
            getattr(self, f"{name}_mean").copy_(v.mean(0))
            getattr(self, f"{name}_std").copy_(v.std(0).clamp(min=1e-6))
        self.normalisation_locked.fill_(True)
        return True

    # -- the two halves ------------------------------------------------------

    def encode(self, images, state, token_ids, token_mask):
        """-> the replay-layer latent [B, latent_dim].

        images     [B, 3*views*frames, H, W] in [0, 1]
        state      [B, 7] the pressures commanded last step, bar
        token_ids  [B, T] int64, token_mask [B, T] float32
        """

        if "state" in self.ablate:
            language, words, word_mask = self.language_encoder(token_ids,
                                                               token_mask)
            vision = self.visual_encoder(images, language, words, word_mask,
                                         None)
            return torch.cat([vision, language], dim=1)
        state_code = self.state_encoder((state - self.state_mean)
                                        / self.state_std)
        if self.arch in MINILM_ARCHS:
            language, words, word_mask = self.language_encoder(token_ids,
                                                               token_mask)
            vision = self.visual_encoder(images, language, words, word_mask,
                                         state_code)
            return torch.cat([vision, state_code, language], dim=1)

        language = self.language_encoder(token_ids, token_mask)
        # the whole point of `film`: the instruction reaches the visual
        # features while they still have spatial structure
        vision = (self.visual_encoder(images, language)
                  if self.language_conditioned
                  else self.visual_encoder(images))
        return torch.cat([
            vision,
            state_code,
            language,
        ], dim=1)

    def head(self, latent):
        """Replay layer -> [B, chunk_steps, 7] pressure CHANGES in bar."""

        x = self.fusion(latent)
        for block in self.residual_blocks:
            x = block(x)
        out = self.action_head(x).view(-1, self.chunk_steps, self.action_dim)
        return out * self.action_std + self.action_mean

    def forward(self, images, state, token_ids, token_mask):
        return self.head(self.encode(images, state, token_ids, token_mask))

    def target_xy(self, latent):
        """Grounding head: the named object's start xy / GROUNDING_SCALE."""
        return self.grounding_head(latent[:, :self.feature_dim])

    def pressure_chunk(self, images, state, token_ids, token_mask):
        """The next `chunk_steps` absolute pressure commands, bar."""

        delta = self.forward(images, state, token_ids, token_mask)
        return state.unsqueeze(1) + torch.cumsum(delta, dim=1)

    # -- continual-learning surgery -----------------------------------------

    def encoder_modules(self):
        extra = ((self.grounding_head,) if self.has_grounding_head else ())
        return (self.visual_encoder, self.state_encoder,
                self.language_encoder) + extra

    def trunk_modules(self):
        return (self.fusion, *self.residual_blocks, self.action_head)

    def encoder_parameters(self):
        return [p for m in self.encoder_modules() for p in m.parameters()]

    def trunk_parameters(self):
        return [p for m in self.trunk_modules() for p in m.parameters()]

    def freeze_encoders(self, frozen=True):
        """Stop the encoders learning, and hold their BatchNorm statistics.

        The eval() is not cosmetic: ResNet18's BatchNorm layers update their
        running mean and variance in train() mode even with requires_grad
        off, which would shift every stored latent's meaning while the buffer
        sat there assuming otherwise.
        """

        for module in self.encoder_modules():
            for p in module.parameters():
                p.requires_grad_(not frozen)
        self._encoders_frozen = frozen
        return self

    @property
    def encoders_frozen(self):
        return bool(getattr(self, "_encoders_frozen", False))

    def train(self, mode=True):
        super().train(mode)
        if mode and self.encoders_frozen:
            for module in self.encoder_modules():
                module.eval()
        return self

    def parameter_report(self):
        rows = {}
        for name in ("visual_encoder", "state_encoder", "language_encoder",
                     "fusion", "residual_blocks", "action_head"):
            rows[name] = sum(p.numel()
                             for p in getattr(self, name).parameters())
        rows["total"] = sum(p.numel() for p in self.parameters())
        rows["encoders"] = sum(p.numel() for p in self.encoder_parameters())
        rows["trunk"] = sum(p.numel() for p in self.trunk_parameters())
        return rows


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

def save_checkpoint(path, model, vocab, extra=None):
    payload = {
        "model_state_dict": model.state_dict(),
        "vocabulary": vocab.state(),
        "frame_history": model.frame_history,
        "frame_stride": FRAME_STRIDE,
        "arch": model.arch,
        "num_views": model.num_views,
        "chunk_steps": model.chunk_steps,
        "state_dim": model.state_dim,
        "action_dim": model.action_dim,
        "feature_dim": model.feature_dim,
        "embed_dim": model.embed_dim,
        "latent_dim": model.latent_dim,
        "max_tokens": MAX_TOKENS,
        "encoders_frozen": model.encoders_frozen,
        "ablate": list(model.ablate),
        "convention": ("inputs: camera frames (oldest first) + previous 7 "
                       "pressures [A1 A2 A3 B1 B2 B3 grip] + tokenised "
                       "instruction; output: the pressure CHANGE per step in "
                       "bar, so next pressure = previous + output"),
    }
    if extra:
        payload.update(extra)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path, device):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"checkpoint not found: {path}\n"
            "run generate_demos.py then train_continual.py")

    ckpt = torch.load(path, map_location=device, weights_only=False)
    vocab = Vocabulary.restore(ckpt["vocabulary"])
    model = ContinualVLA(
        vocab_size=len(vocab),
        frame_history=int(ckpt.get("frame_history", FRAME_HISTORY)),
        state_dim=int(ckpt.get("state_dim", STATE_DIM)),
        action_dim=int(ckpt.get("action_dim", ACTION_DIM)),
        chunk_steps=int(ckpt.get("chunk_steps", CHUNK_STEPS)),
        arch=ckpt.get("arch", "resnet18"),
        num_views=int(ckpt.get("num_views", NUM_VIEWS)),
        feature_dim=int(ckpt.get("feature_dim", FEATURE_DIM)),
        embed_dim=int(ckpt.get("embed_dim", 64)),
        ablate=tuple(ckpt.get("ablate", ())))
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, vocab, ckpt


if __name__ == "__main__":
    import tasks

    vocab = Vocabulary.standard()
    model = ContinualVLA(len(vocab))
    ids, mask = vocab.encode_batch([tasks.instruction("cube"),
                                    tasks.instruction("can")])
    images = torch.zeros(2, 3 * NUM_VIEWS * FRAME_HISTORY, IMAGE_SIZE,
                         IMAGE_SIZE)
    latent = model.encode(images, torch.zeros(2, STATE_DIM), ids, mask)
    out = model.head(latent)

    print(f"vocabulary        {len(vocab)} words: "
          f"{' '.join(vocab.words)}")
    print(f"latent            {tuple(latent.shape)}  "
          f"(vision | state | language, {FEATURE_DIM} each)")
    print(f"output            {tuple(out.shape)}  [batch, chunk, pressures]")
    print()
    for name, n in model.parameter_report().items():
        print(f"  {name:18s} {n:>12,}")
