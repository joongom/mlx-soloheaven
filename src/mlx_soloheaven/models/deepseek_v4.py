"""DeepSeek-V4 (``model_type: deepseek_v4``) for MLX — step 1: config.

Full plan and evidence: ``docs/specs/deepseek-v4-mlx-port.md``.

Short version: mlx-lm has no ``deepseek_v4``, and neither does vLLM-Metal (which
"uses MLX as the compute backend" and ships zero model implementations), so an
MLX implementation is the only way to run this model inside our engine — which
is in turn the only way SoloHeaven's KV/prefix machinery applies to it at all.

The port base is mlx-lm's ``deepseek_v32``: it already implements MLA, the DSA
``Indexer``, ``noaux_tc`` MoE routing, YaRN, and — crucially —
fp8 block dequantization in ``sanitize()``, so it can read DeepSeek's official
checkpoints. Most of V4's ``config.json`` field names are already v32's.

THIS FILE IS NOT YET A WORKING MODEL. It currently pins down the config
mapping only: the argument surface, which V4 keys are new, and which v32 fields
they line up with. Model/forward come next (steps 2-6 of the spec). It is
deliberately NOT registered in ``models/__init__.py`` yet — registering an
incomplete architecture would make mlx-lm dispatch to it and fail at load
instead of saying "model type not supported", which is a worse error.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

MODEL_TYPE = "deepseek_v4"

#: Layer-type marker used by ``compress_ratios``: 0 = dense (plain RoPE, no KV
#: compression), 4 = compressed + DSA Indexer, 128 = compressed only.
COMPRESS_DENSE = 0
COMPRESS_WITH_INDEXER = 4


@dataclass
class ModelArgs:
    """V4's config, in mlx-lm field naming where v32 already has an equivalent.

    Fields carried over from ``deepseek_v32.ModelArgs`` keep their v32 names so
    the eventual model code can stay close to that file. Fields with no v32
    equivalent are grouped under "V4 additions" and each one is a real
    architectural delta — see the spec's delta table.
    """

    model_type: str = MODEL_TYPE

    # --- shared with deepseek_v32 -----------------------------------------
    vocab_size: int = 129280
    hidden_size: int = 4096
    num_hidden_layers: int = 43
    num_attention_heads: int = 64
    num_key_value_heads: int = 1
    moe_intermediate_size: int = 2048
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    num_experts_per_tok: int = 6
    routed_scaling_factor: float = 1.5
    norm_topk_prob: bool = True
    topk_method: str = "noaux_tc"
    q_lora_rank: int = 1024
    qk_rope_head_dim: int = 64
    index_head_dim: int = 128
    index_n_heads: int = 64
    index_topk: int = 512
    max_position_embeddings: int = 1048576
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict[str, Any]] = None
    attention_bias: bool = False

    # --- V4 additions ------------------------------------------------------
    #: MLA latent width. V4 keeps ONE kv head (num_key_value_heads == 1) of this
    #: width, which is why its KV is tiny; v32 splits kv_lora_rank + rope dims.
    head_dim: int = 512
    #: Grouped low-rank output projection (einsum bsgd,grd->bsgr), no v32 analog.
    o_lora_rank: int = 1024
    o_groups: int = 8
    #: Per-layer KV compression: 0 dense, 4 compressed+Indexer, 128 compressed.
    #: Length must equal num_hidden_layers (the published config carries a few
    #: trailing zeros for the MTP/DSpark blocks — see _normalize_compress_ratios).
    compress_ratios: List[int] = field(default_factory=list)
    compress_rope_theta: float = 160000.0
    #: Sliding-window ring; carries the uncompressed recent context.
    sliding_window: int = 128
    #: Routing score. V4 uses sqrt(softplus(x)) where v32 uses sigmoid.
    scoring_func: str = "sqrtsoftplus"
    #: First N layers route by a ``tid2eid[vocab, topk]`` LOOKUP TABLE keyed on
    #: the input token id — no gate computation at inference.
    num_hash_layers: int = 3
    #: Hyper-Connections: multiple residual streams reduced per block.
    hc_mult: int = 4
    hc_eps: float = 1e-6
    hc_sinkhorn_iters: int = 20
    #: Clipped SwiGLU.
    swiglu_limit: float = 10.0
    #: Multi-token prediction / DSpark. Not implemented; recorded so the config
    #: round-trips and a later step can pick them up.
    num_nextn_predict_layers: int = 0
    dspark_block_size: int = 0
    dspark_target_layer_ids: List[int] = field(default_factory=list)
    dspark_markov_rank: int = 0

    # ----------------------------------------------------------------------

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> "ModelArgs":
        fields = cls.__dataclass_fields__
        cfg = dict(params)
        # V4 nests nothing, but be tolerant of a text_config wrapper the way
        # exaone4_5 needed, so a repackaged checkpoint does not silently load
        # defaults for every field.
        if isinstance(cfg.get("text_config"), dict):
            nested = dict(cfg["text_config"])
            nested.update({k: v for k, v in cfg.items() if k != "text_config"})
            cfg = nested
        known = {k: v for k, v in cfg.items() if k in fields}
        args = cls(**known)
        args.compress_ratios = _normalize_compress_ratios(
            cfg.get("compress_ratios") or [], args.num_hidden_layers
        )
        return args

    # -- derived ------------------------------------------------------------

    def layer_compress_ratio(self, layer_idx: int) -> int:
        if not self.compress_ratios:
            return COMPRESS_DENSE
        return self.compress_ratios[layer_idx]

    def layer_has_indexer(self, layer_idx: int) -> bool:
        return self.layer_compress_ratio(layer_idx) == COMPRESS_WITH_INDEXER

    def layer_routes_by_hash(self, layer_idx: int) -> bool:
        return layer_idx < self.num_hash_layers


def _normalize_compress_ratios(ratios: List[int], num_layers: int) -> List[int]:
    """Trim/pad ``compress_ratios`` to exactly ``num_layers`` entries.

    The published 0731 config lists 46 entries for 43 layers — the tail belongs
    to the MTP/DSpark blocks. Indexing it by layer id without trimming would be
    silently fine here but wrong for any config whose extra entries are not
    trailing, so normalize explicitly and keep the layer mapping honest.
    """
    out = list(ratios[:num_layers])
    if len(out) < num_layers:
        out += [COMPRESS_DENSE] * (num_layers - len(out))
    return out


# ---------------------------------------------------------------------------
# Step 2a — pure functions, ported verbatim from DeepSeek's inference/model.py
#
# These are separated out and unit-tested first on purpose: they are index math
# and routing rules, so an off-by-one here does not raise, it silently attends
# to the wrong positions or routes to the wrong experts. Everything built on top
# would then be debugged against a foundation that was already wrong.
# ---------------------------------------------------------------------------

import mlx.core as mx  # noqa: E402  (kept next to the code that uses it)

#: Sentinel used by DeepSeek's sparse_attn kernel for "no position" — gathered
#: slots equal to this are excluded from the softmax.
MASKED_INDEX = -1


def window_topk_indices(
    window_size: int, seqlen: int, start_pos: int
) -> mx.array:
    """Sliding-window slot indices, per query position.

    Port of ``get_window_topk_idxs``. Returns ``[seqlen, k]`` (prefill) or
    ``[1, window_size]`` (decode) WITHOUT the batch axis — callers broadcast.

    The three branches are the ring's three regimes: fully wrapped (rotate the
    slot order so the oldest slot comes first), partially filled (pad the unused
    tail with MASKED_INDEX), and the prefill triangle.
    """
    if start_pos >= window_size - 1:
        sp = start_pos % window_size
        row = mx.concatenate(
            [mx.arange(sp + 1, window_size), mx.arange(0, sp + 1)], axis=0
        )
        return row.astype(mx.int32)[None, :]
    if start_pos > 0:
        row = mx.concatenate(
            [
                mx.arange(start_pos + 1),
                mx.full((window_size - start_pos - 1,), MASKED_INDEX),
            ],
            axis=0,
        )
        return row.astype(mx.int32)[None, :]
    base = mx.arange(seqlen)[:, None]
    width = min(seqlen, window_size)
    matrix = mx.maximum(base - window_size + 1, 0) + mx.arange(width)
    return mx.where(matrix > base, MASKED_INDEX, matrix).astype(mx.int32)


def compress_topk_indices(
    ratio: int, seqlen: int, start_pos: int, offset: int
) -> mx.array:
    """Compressed-KV slot indices, per query position.

    Port of ``get_compress_topk_idxs``. ``offset`` is where the compressed
    region starts inside the shared per-layer buffer (``window_size``), since
    DeepSeek stores ring and compressed KV in ONE buffer.

    A compressed slot only becomes visible once its whole ``ratio``-token group
    has been consumed, which is what the ``>=`` mask encodes.
    """
    if start_pos > 0:
        row = mx.arange((start_pos + 1) // ratio) + offset
        return row.astype(mx.int32)[None, :]
    width = seqlen // ratio
    matrix = mx.broadcast_to(mx.arange(width)[None, :], (seqlen, width))
    visible = (mx.arange(1, seqlen + 1) // ratio)[:, None]
    return mx.where(matrix >= visible, MASKED_INDEX, matrix + offset).astype(mx.int32)


def sqrtsoftplus(x: mx.array) -> mx.array:
    """V4's routing score: ``sqrt(softplus(x))`` (v32 uses sigmoid).

    softplus is computed in the numerically stable form
    ``log1p(exp(-|x|)) + max(x, 0)`` so large logits do not overflow before the
    sqrt — the naive ``log(1+exp(x))`` saturates to inf and the sqrt then
    propagates it into every routing decision.
    """
    return mx.sqrt(mx.log1p(mx.exp(-mx.abs(x))) + mx.maximum(x, 0))


def route(
    scores: mx.array,
    topk: int,
    route_scale: float,
    bias: mx.array | None = None,
) -> tuple[mx.array, mx.array]:
    """noaux_tc top-k selection. Returns ``(weights, indices)``.

    The correction ``bias`` shifts scores for SELECTION only — the returned
    weights are gathered from the UNBIASED scores. Folding the bias into the
    weights instead is a silent quality regression, not an error.
    """
    biased = scores if bias is None else scores + bias
    indices = mx.argpartition(-biased, kth=topk - 1, axis=-1)[..., :topk]
    weights = mx.take_along_axis(scores, indices, axis=-1)
    weights = weights / weights.sum(axis=-1, keepdims=True)
    return weights * route_scale, indices


def clipped_swiglu(gate: mx.array, up: mx.array, limit: float) -> mx.array:
    """SwiGLU with V4's asymmetric clipping.

    Note the asymmetry, straight from the reference: ``up`` is clamped on BOTH
    sides, ``gate`` only from above. Clamping gate symmetrically would cut off
    the negative tail that silu needs.
    """
    if limit > 0:
        up = mx.clip(up, -limit, limit)
        gate = mx.minimum(gate, limit)
    return mx.sigmoid(gate) * gate * up
