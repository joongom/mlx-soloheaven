"""EXAONE-4.5 (`model_type: exaone4_5`) architecture support tests.

These run entirely offline against a tiny synthetic checkpoint — no download,
no GPU-sized allocation. What they pin down:

  * mlx-lm's importlib dispatch resolves `exaone4_5` to our module
  * the nested `text_config` / buried `rope_theta` / implicit `head_dim` /
    top-level `tie_word_embeddings` all flatten correctly
  * `sanitize()` rewrites the `model.language_model.*` prefix and drops the
    vision tower + MTP head, so `load_weights(strict=True)` passes
  * **global NoPE**: RoPE must be applied on sliding layers ONLY. This is the
    one thing that would produce plausible-but-wrong output rather than an
    exception, so it is asserted per layer.
"""

from __future__ import annotations

import importlib

import mlx.core as mx
import pytest

from mlx_soloheaven.models import register_extra_architectures

register_extra_architectures()

exaone4_5 = importlib.import_module("mlx_lm.models.exaone4_5")


# Tiny stand-in for LGAI-EXAONE/EXAONE-4.5-33B's config.json: same shape and
# same key placement, ~1/1000th the size.
HIDDEN, HEADS, KV_HEADS, HEAD_DIM = 64, 4, 2, 16
LAYERS, VOCAB, INTERMEDIATE = 8, 128, 96

TINY_CONFIG = {
    "architectures": ["Exaone4_5_ForConditionalGeneration"],
    "model_type": "exaone4_5",
    "tie_word_embeddings": False,
    "vocab_size": VOCAB,
    "image_token_id": 67,
    "text_config": {
        "model_type": "exaone4_5_text",
        "architectures": ["Exaone4ForCausalLM"],
        "hidden_size": HIDDEN,
        "num_hidden_layers": LAYERS,
        "intermediate_size": INTERMEDIATE,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "rms_norm_eps": 1e-5,
        "vocab_size": VOCAB,
        "max_position_embeddings": 4096,
        "sliding_window": 32,
        "sliding_window_pattern": "LLLG",
        "layer_types": (["sliding_attention"] * 3 + ["full_attention"]) * 2,
        "eos_token_id": 53,
        "rope_scaling": {
            "factor": 16.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
            "rope_theta": 1000000.0,
            "rope_type": "llama3",
        },
    },
    "vision_config": {"model_type": "exaone4_5_vision", "hidden_size": 32, "depth": 2},
}


def _original_layout_weights() -> dict[str, mx.array]:
    """Weights keyed exactly as the published checkpoint keys them."""
    weights = {
        "lm_head.weight": mx.random.normal((VOCAB, HIDDEN)),
        "model.language_model.embed_tokens.weight": mx.random.normal(
            (VOCAB, HIDDEN)
        ),
        "model.language_model.norm.weight": mx.ones((HIDDEN,)),
        # Towers this module does not implement:
        "model.visual.patch_embed.proj.weight": mx.zeros((8, 3, 2, 14, 14)),
        "model.visual.blocks.0.attn.qkv.weight": mx.zeros((24, 8)),
        "model.visual.merger.ln_q.weight": mx.zeros((8,)),
        "mtp.fc.weight": mx.zeros((HIDDEN, 2 * HIDDEN)),
        "mtp.norm.weight": mx.ones((HIDDEN,)),
        "mtp.layers.0.self_attn.q_proj.weight": mx.zeros((HIDDEN, HIDDEN)),
    }
    for i in range(LAYERS):
        p = f"model.language_model.layers.{i}."
        weights.update(
            {
                p + "self_attn.q_proj.weight": mx.random.normal(
                    (HEADS * HEAD_DIM, HIDDEN)
                ),
                p + "self_attn.k_proj.weight": mx.random.normal(
                    (KV_HEADS * HEAD_DIM, HIDDEN)
                ),
                p + "self_attn.v_proj.weight": mx.random.normal(
                    (KV_HEADS * HEAD_DIM, HIDDEN)
                ),
                p + "self_attn.o_proj.weight": mx.random.normal(
                    (HIDDEN, HEADS * HEAD_DIM)
                ),
                p + "self_attn.q_norm.weight": mx.ones((HEAD_DIM,)),
                p + "self_attn.k_norm.weight": mx.ones((HEAD_DIM,)),
                p + "mlp.gate_proj.weight": mx.random.normal((INTERMEDIATE, HIDDEN)),
                p + "mlp.up_proj.weight": mx.random.normal((INTERMEDIATE, HIDDEN)),
                p + "mlp.down_proj.weight": mx.random.normal((HIDDEN, INTERMEDIATE)),
                p + "post_attention_layernorm.weight": mx.ones((HIDDEN,)),
                p + "post_feedforward_layernorm.weight": mx.ones((HIDDEN,)),
            }
        )
    return weights


@pytest.fixture(scope="module")
def loaded_model():
    args = exaone4_5.ModelArgs.from_dict(TINY_CONFIG)
    model = exaone4_5.Model(args)
    model.load_weights(list(model.sanitize(_original_layout_weights()).items()))
    mx.eval(model.parameters())
    return model


# ---------------------------------------------------------------------------
# registration / dispatch
# ---------------------------------------------------------------------------

def test_mlx_lm_dispatch_resolves_exaone4_5():
    from mlx_lm.utils import _get_classes

    model_class, args_class = _get_classes(TINY_CONFIG)
    assert model_class is exaone4_5.Model
    assert args_class is exaone4_5.ModelArgs


def test_registration_is_idempotent():
    first = register_extra_architectures()
    assert register_extra_architectures() == first


# ---------------------------------------------------------------------------
# config flattening
# ---------------------------------------------------------------------------

def test_flatten_config_lifts_nested_text_config():
    args = exaone4_5.ModelArgs.from_dict(TINY_CONFIG)
    assert args.hidden_size == HIDDEN
    assert args.num_hidden_layers == LAYERS
    assert args.num_key_value_heads == KV_HEADS
    assert args.sliding_window == 32


def test_flatten_config_digs_rope_theta_out_of_rope_scaling():
    # exaone4_5 has no top-level rope_theta; mlx-lm needs it as `base`.
    assert "rope_theta" not in TINY_CONFIG["text_config"]
    assert exaone4_5.ModelArgs.from_dict(TINY_CONFIG).rope_theta == 1000000.0


def test_flatten_config_infers_head_dim():
    assert "head_dim" not in TINY_CONFIG["text_config"]
    assert exaone4_5.ModelArgs.from_dict(TINY_CONFIG).head_dim == HIDDEN // HEADS


def test_flatten_config_takes_tie_word_embeddings_from_top_level():
    # It lives on the multimodal wrapper, not on text_config.
    assert exaone4_5.ModelArgs.from_dict(TINY_CONFIG).tie_word_embeddings is False


def test_flatten_config_accepts_already_flat_config():
    flat = exaone4_5.flatten_config(TINY_CONFIG)
    assert exaone4_5.flatten_config(flat) == flat


def test_sliding_window_pattern_keeps_compact_form_when_it_tiles():
    assert exaone4_5.flatten_config(TINY_CONFIG)["sliding_window_pattern"] == "LLLG"


def test_sliding_window_pattern_expands_when_layer_types_are_irregular():
    config = {**TINY_CONFIG, "text_config": {**TINY_CONFIG["text_config"]}}
    config["text_config"]["layer_types"] = ["sliding_attention"] * 7 + [
        "full_attention"
    ]
    # "LLLG" would wrongly mark layers 3 as global; the expanded form must win.
    assert exaone4_5.flatten_config(config)["sliding_window_pattern"] == "LLLLLLLG"


# ---------------------------------------------------------------------------
# weight sanitizing
# ---------------------------------------------------------------------------

def test_sanitize_rewrites_prefix_and_drops_unimplemented_towers():
    sanitized = exaone4_5.Model(
        exaone4_5.ModelArgs.from_dict(TINY_CONFIG)
    ).sanitize(_original_layout_weights())

    assert "model.embed_tokens.weight" in sanitized
    assert "model.layers.0.self_attn.q_proj.weight" in sanitized
    assert "lm_head.weight" in sanitized  # already unprefixed upstream
    assert not any(k.startswith(("model.visual.", "mtp.")) for k in sanitized)


def test_sanitize_is_idempotent_on_converted_checkpoints():
    model = exaone4_5.Model(exaone4_5.ModelArgs.from_dict(TINY_CONFIG))
    once = model.sanitize(_original_layout_weights())
    assert model.sanitize(once).keys() == once.keys()


def test_weights_load_strictly(loaded_model):
    # Covered by the fixture, which uses load_weights' default strict=True:
    # any leftover or missing key would have raised.
    assert loaded_model is not None


# ---------------------------------------------------------------------------
# architecture semantics
# ---------------------------------------------------------------------------

def test_rope_is_applied_on_sliding_layers_only(loaded_model):
    """Global NoPE — transformers guards RoPE with `sliding_window is None or
    is_sliding`. Getting this wrong degrades quality silently, never loudly."""
    is_local = [layer.self_attn.is_local for layer in loaded_model.layers]
    use_rope = [layer.self_attn.use_rope for layer in loaded_model.layers]

    assert is_local == [True, True, True, False] * 2
    assert use_rope == is_local


def test_hybrid_cache_pairs_rotating_with_sliding_layers(loaded_model):
    kinds = [type(c).__name__ for c in loaded_model.make_cache()]
    assert kinds == ["RotatingKVCache"] * 3 + ["KVCache"] + [
        "RotatingKVCache"
    ] * 3 + ["KVCache"]


def test_prefill_then_decode(loaded_model):
    cache = loaded_model.make_cache()

    prefill = loaded_model(mx.array([[1, 2, 3, 4, 5]]), cache=cache)
    mx.eval(prefill)
    assert prefill.shape == (1, 5, VOCAB)
    assert bool(mx.all(mx.isfinite(prefill)))

    step = loaded_model(mx.array([[6]]), cache=cache)
    mx.eval(step)
    assert step.shape == (1, 1, VOCAB)
    assert bool(mx.all(mx.isfinite(step)))


def test_decode_past_the_sliding_window(loaded_model):
    """The 32-token window must wrap without corrupting the global layers."""
    cache = loaded_model.make_cache()
    loaded_model(mx.arange(40).reshape(1, 40) % VOCAB, cache=cache)
    step = loaded_model(mx.array([[7]]), cache=cache)
    mx.eval(step)
    assert bool(mx.all(mx.isfinite(step)))


def test_missing_rope_theta_raises_instead_of_guessing():
    """A wrong RoPE base degrades quality silently, so refuse to invent one."""
    config = {**TINY_CONFIG, "text_config": {**TINY_CONFIG["text_config"]}}
    config["text_config"]["rope_scaling"] = {
        **config["text_config"]["rope_scaling"]
    }
    del config["text_config"]["rope_scaling"]["rope_theta"]

    with pytest.raises(ValueError, match="rope_theta"):
        exaone4_5.ModelArgs.from_dict(config)


def test_integer_sliding_window_pattern_matches_transformers_rule():
    """transformers accepts `sliding_window_pattern: int`; stride 4 means every
    4th layer is full attention -- the same thing as "LLLG"."""
    config = {**TINY_CONFIG, "text_config": {**TINY_CONFIG["text_config"]}}
    config["text_config"]["sliding_window_pattern"] = 4
    del config["text_config"]["layer_types"]

    assert exaone4_5.flatten_config(config)["sliding_window_pattern"] == "LLLG"
