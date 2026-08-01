"""DeepSeek-V4 config mapping (step 1 of docs/specs/deepseek-v4-mlx-port.md).

Offline. Pins the argument surface against the published 0731 config so the
model code written on top of it (steps 2-6) cannot be built on a misread field.
The values here were read from deepseek-ai/DeepSeek-V4-Flash-0731/config.json.
"""

from __future__ import annotations

import pytest

from mlx_soloheaven.models.deepseek_v4 import (
    COMPRESS_DENSE,
    COMPRESS_WITH_INDEXER,
    ModelArgs,
)

# Verbatim subset of the published config.json, including the trailing
# compress_ratios entries that belong to the MTP/DSpark blocks (46 for 43
# layers) — the case that must be normalized rather than indexed blindly.
V4_CONFIG = {
    "model_type": "deepseek_v4",
    "vocab_size": 129280,
    "hidden_size": 4096,
    "num_hidden_layers": 43,
    "num_attention_heads": 64,
    "num_key_value_heads": 1,
    "head_dim": 512,
    "q_lora_rank": 1024,
    "o_lora_rank": 1024,
    "o_groups": 8,
    "qk_rope_head_dim": 64,
    "moe_intermediate_size": 2048,
    "n_routed_experts": 256,
    "n_shared_experts": 1,
    "num_experts_per_tok": 6,
    "routed_scaling_factor": 1.5,
    "norm_topk_prob": True,
    "topk_method": "noaux_tc",
    "scoring_func": "sqrtsoftplus",
    "index_head_dim": 128,
    "index_n_heads": 64,
    "index_topk": 512,
    "sliding_window": 128,
    "num_hash_layers": 3,
    "hc_eps": 1e-06,
    "hc_mult": 4,
    "hc_sinkhorn_iters": 20,
    "swiglu_limit": 10.0,
    "compress_rope_theta": 160000,
    "max_position_embeddings": 1048576,
    "rms_norm_eps": 1e-06,
    "rope_theta": 10000,
    "rope_scaling": {
        "beta_fast": 32,
        "beta_slow": 1,
        "factor": 16,
        "original_max_position_embeddings": 65536,
        "type": "yarn",
    },
    "num_nextn_predict_layers": 1,
    "compress_ratios": [0, 0] + [4, 128] * 20 + [4, 0, 0, 0],
}


@pytest.fixture(scope="module")
def args():
    return ModelArgs.from_dict(V4_CONFIG)


def test_core_dimensions(args):
    assert args.model_type == "deepseek_v4"
    assert (args.hidden_size, args.num_hidden_layers) == (4096, 43)
    assert args.num_attention_heads == 64
    # MLA: ONE kv head of width 512 is why V4's KV is tiny.
    assert args.num_key_value_heads == 1
    assert args.head_dim == 512


def test_v4_only_fields_are_not_silently_defaulted(args):
    # Each of these is a real architectural delta vs deepseek_v32; a typo in the
    # field name would leave the dataclass default in place and change the model
    # without any error.
    assert args.o_lora_rank == 1024
    assert args.o_groups == 8
    assert args.scoring_func == "sqrtsoftplus"
    assert args.num_hash_layers == 3
    assert args.hc_mult == 4
    assert args.hc_sinkhorn_iters == 20
    assert args.swiglu_limit == 10.0
    assert args.sliding_window == 128
    assert args.compress_rope_theta == 160000


def test_moe_and_indexer_fields_match_v32_naming(args):
    assert args.n_routed_experts == 256
    assert args.n_shared_experts == 1
    assert args.num_experts_per_tok == 6
    assert args.routed_scaling_factor == 1.5
    assert args.topk_method == "noaux_tc"
    assert (args.index_n_heads, args.index_head_dim, args.index_topk) == (64, 128, 512)


def test_compress_ratios_are_trimmed_to_the_layer_count(args):
    # Published config lists 46 entries for 43 layers; the tail belongs to the
    # MTP/DSpark blocks and must not shift the per-layer mapping.
    assert len(V4_CONFIG["compress_ratios"]) == 46
    assert len(args.compress_ratios) == 43


def test_compress_ratios_are_padded_when_short():
    cfg = {**V4_CONFIG, "compress_ratios": [0, 4]}
    a = ModelArgs.from_dict(cfg)
    assert len(a.compress_ratios) == 43
    assert a.compress_ratios[42] == COMPRESS_DENSE


def test_layer_kinds(args):
    ratios = args.compress_ratios
    assert ratios[0] == COMPRESS_DENSE and ratios[1] == COMPRESS_DENSE
    assert sum(1 for r in ratios if r == COMPRESS_WITH_INDEXER) == 21
    assert sum(1 for r in ratios if r == 128) == 20
    assert args.layer_has_indexer(2) is True
    assert args.layer_has_indexer(3) is False


def test_hash_routing_covers_exactly_the_first_n_layers(args):
    routed = [i for i in range(args.num_hidden_layers) if args.layer_routes_by_hash(i)]
    assert routed == [0, 1, 2]


def test_rope_scaling_is_yarn(args):
    assert args.rope_scaling["type"] == "yarn"
    assert args.rope_scaling["factor"] == 16
    assert args.rope_scaling["original_max_position_embeddings"] == 65536


def test_from_dict_tolerates_a_text_config_wrapper():
    a = ModelArgs.from_dict({"model_type": "deepseek_v4", "text_config": V4_CONFIG})
    assert a.hidden_size == 4096
    assert a.num_hidden_layers == 43


def test_unknown_config_keys_are_ignored():
    a = ModelArgs.from_dict({**V4_CONFIG, "some_future_key": 123})
    assert a.hidden_size == 4096
