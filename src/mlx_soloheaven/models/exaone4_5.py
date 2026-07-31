"""EXAONE-4.5 (``model_type: exaone4_5``) for MLX — text tower.

LGAI-EXAONE ships EXAONE-4.5 as ``Exaone4_5_ForConditionalGeneration``: a
multimodal wrapper whose ``config.json`` nests

    text_config   -> model_type "exaone4_5_text", architectures ["Exaone4ForCausalLM"]
    vision_config -> model_type "exaone4_5_vision" (Qwen2.5-VL-style ViT)

Neither ``exaone4_5`` nor ``exaone4_5_text`` exists in mlx-lm, so the checkpoint
cannot be loaded or converted out of the box.

The key finding that makes this module small: **the text tower is bit-for-bit
EXAONE-4.0's architecture**, which mlx-lm already implements in
``mlx_lm/models/exaone4.py``. Verified against
``transformers/models/exaone4/modeling_exaone4.py`` (v5.5.4) and against the
published tensor shapes:

  * QK-norm: ``RMSNorm(head_dim)`` on q/k after the head reshape           (match)
  * hybrid attention: ``layer_types`` "LLLG", sliding_window 4096          (match)
  * **global NoPE**: RoPE is applied only on sliding layers — transformers
    guards with ``if self.sliding_window is None or self.is_sliding``, mlx-lm
    with ``use_rope = is_local is None or is_local``                       (match)
  * sandwich post-norm: ``x + post_attention_layernorm(attn(x))`` then
    ``h + post_feedforward_layernorm(mlp(h))``                             (match)
  * head_dim 128 (q_proj [5120,5120] / k_proj [1024,5120], q_norm [128])   (match)
  * untied lm_head [153600, 5120]                                         (match)

So the only real differences from ``exaone4`` are packaging, and this module
handles exactly those three:

  1. **Config shape** — the fields live under ``text_config``, ``rope_theta`` is
     buried inside ``rope_scaling``, ``head_dim`` is implicit, and
     ``tie_word_embeddings`` sits at the top level. :func:`flatten_config`
     normalizes all of that into a flat ``exaone4``-shaped config.
  2. **Weight prefix** — tensors are under ``model.language_model.*``;
     :meth:`Model.sanitize` rewrites them to ``model.*``.
  3. **Extra towers** — ``model.visual.*`` (ViT) and ``mtp.*`` (a 1-layer
     multi-token-prediction head) are dropped by :meth:`Model.sanitize`.

SCOPE: text-only. The vision tower is *not* implemented — images/video are out
of reach for this module, and serving one requires mlx-vlm, which has no EXAONE
support at all. The MTP head is dropped here but is a self-contained
speculative-decoding drafter that a follow-up can convert separately (its layout
mirrors the DeepSeek/GLM style: ``fc`` over concat[norm(embed), norm(hidden)],
one transformer layer, ``norm``, then the target's shared ``lm_head``).

Register with :func:`mlx_soloheaven.models.register_extra_architectures` so
mlx-lm's importlib dispatch finds it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import mlx.core as mx

from mlx_lm.models.exaone4 import Model as _Exaone4Model
from mlx_lm.models.exaone4 import ModelArgs as _Exaone4Args

MODEL_TYPE = "exaone4_5"

#: Weight-key prefix of the text tower in the original checkpoint.
TEXT_WEIGHT_PREFIXES = ("model.language_model.", "language_model.")
#: Sub-towers this module does not implement (see SCOPE in the module docstring).
DROPPED_WEIGHT_PREFIXES = ("model.visual.", "visual.", "mtp.")


def _rope_config(cfg: Dict[str, Any]) -> Dict[str, Any] | None:
    """Return the rope scaling dict under either transformers-4 or -5 naming."""
    return cfg.get("rope_scaling") or cfg.get("rope_parameters") or None


def _require_rope_theta(flat: Dict[str, Any], rope: Dict[str, Any] | None) -> float:
    """RoPE base, which exaone4_5 buries inside ``rope_scaling``.

    Deliberately raises rather than defaulting: mlx-lm needs this as
    ``initialize_rope``'s ``base``, and a wrong base does not fail — it quietly
    produces a model that degrades with position. Every real EXAONE config
    supplies it (flat ``rope_theta`` for 4.0, ``rope_scaling.rope_theta`` for
    4.5), so reaching this error means the config is malformed.
    """
    theta = flat.get("rope_theta")
    if theta is None:
        theta = (rope or {}).get("rope_theta")
    if theta is None:
        raise ValueError(
            "exaone4_5 config has no rope_theta — looked at text_config."
            "rope_theta and text_config.rope_scaling.rope_theta. Refusing to "
            "guess a RoPE base, which would silently degrade long-context "
            "quality instead of failing."
        )
    return float(theta)


def _derive_sliding_window_pattern(cfg: Dict[str, Any]) -> str | None:
    """Return the L/G pattern string mlx-lm's ``exaone4`` expects.

    ``exaone4`` indexes the pattern with ``i % len(pattern)``, so the compact
    "LLLG" and a full 64-char string are equivalent *when* the compact form
    actually tiles ``layer_types``. We prefer the compact form for readability
    but fall back to the fully expanded string if the checkpoint's
    ``layer_types`` is irregular — being wrong here would silently apply RoPE to
    the wrong layers, which is not the kind of bug that shows up as an error.
    """
    layer_types = cfg.get("layer_types")
    compact = cfg.get("sliding_window_pattern")
    if isinstance(compact, int):  # some configs express it as a stride
        compact = "L" * (compact - 1) + "G" if compact > 0 else None

    if not layer_types:
        return compact

    expanded = "".join(
        "L" if t == "sliding_attention" else "G" for t in layer_types
    )
    if compact and all(
        compact[i % len(compact)] == expanded[i] for i in range(len(expanded))
    ):
        return compact
    return expanded


def flatten_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize an ``exaone4_5`` config into a flat ``exaone4``-shaped one.

    Accepts either the original nested config or an already-flattened one, so
    it is safe to call on a converted checkpoint.
    """
    text = config.get("text_config")
    flat: Dict[str, Any] = dict(text) if text else dict(config)

    # rope_theta lives inside rope_scaling for exaone4_5, and mlx-lm's
    # initialize_rope() takes it as the separate `base` argument.
    rope = _rope_config(flat) or _rope_config(config)
    flat["rope_theta"] = _require_rope_theta(flat, rope)
    flat["rope_scaling"] = rope

    hidden = flat["hidden_size"]
    heads = flat["num_attention_heads"]
    if flat.get("head_dim") is None:
        flat["head_dim"] = hidden // heads

    flat["sliding_window"] = flat.get("sliding_window")
    flat["sliding_window_pattern"] = _derive_sliding_window_pattern(flat)

    # These are top-level in the multimodal wrapper, not in text_config.
    if text is not None:
        flat["tie_word_embeddings"] = bool(config.get("tie_word_embeddings", False))
        flat["vocab_size"] = flat.get("vocab_size") or config.get("vocab_size")
    flat.setdefault("tie_word_embeddings", False)

    flat["model_type"] = config.get("model_type", MODEL_TYPE)
    return flat


@dataclass
class ModelArgs(_Exaone4Args):
    """``exaone4`` args, constructed from a nested ``exaone4_5`` config."""

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> "ModelArgs":
        flat = flatten_config(params)
        fields = cls.__dataclass_fields__
        return cls(**{k: v for k, v in flat.items() if k in fields})


class Model(_Exaone4Model):
    """EXAONE-4.5 text tower. Identical compute to mlx-lm's ``exaone4``."""

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        out: Dict[str, mx.array] = {}
        for key, value in weights.items():
            if key.startswith(DROPPED_WEIGHT_PREFIXES):
                continue
            for prefix in TEXT_WEIGHT_PREFIXES:
                if key.startswith(prefix):
                    key = "model." + key[len(prefix) :]
                    break
            out[key] = value
        return out
