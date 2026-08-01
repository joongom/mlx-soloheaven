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


# ---------------------------------------------------------------------------
# Step 2b — the modules, ported from DeepSeek's inference/model.py with all
# state moved OUT of the modules into explicit cache objects (mlx-lm style).
#
# Numerics deltas vs the reference, both deliberate:
#   * The fp8/fp4 activation-quant simulations (act_quant/fp4_act_quant) are
#     omitted — we keep activations in the compute dtype. The QAT'd weights
#     tolerate higher-precision activations; oracle agreement (spec step 8)
#     is what verifies this.
#   * The Hadamard rotation in the Indexer exists only to condition activations
#     for fp4 quantization. It is orthogonal, so with quantization omitted it
#     cancels out of every dot product — omitted as well.
# ---------------------------------------------------------------------------

import mlx.nn as nn  # noqa: E402
from mlx_lm.models.cache import _BaseCache  # noqa: E402
from mlx_lm.models.switch_layers import SwitchGLU, SwitchLinear  # noqa: E402


def yarn_freqs(
    dim: int,
    base: float,
    original_seq_len: int,
    factor: float,
    beta_fast: float,
    beta_slow: float,
) -> mx.array:
    """Per-pair angular frequencies ``[dim/2]``, YaRN-scaled when
    ``original_seq_len > 0`` (port of precompute_freqs_cis minus the position
    outer product)."""
    import math

    freqs = 1.0 / (base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
    if original_seq_len > 0:
        def correction_dim(num_rotations: float) -> float:
            return (
                dim
                * math.log(original_seq_len / (num_rotations * 2 * math.pi))
                / (2 * math.log(base))
            )

        low = max(math.floor(correction_dim(beta_fast)), 0)
        high = min(math.ceil(correction_dim(beta_slow)), dim - 1)
        if low == high:
            high += 0.001
        ramp = mx.clip(
            (mx.arange(dim // 2, dtype=mx.float32) - low) / (high - low), 0, 1
        )
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    return freqs


def rope_cos_sin(freqs: mx.array, positions: mx.array) -> tuple[mx.array, mx.array]:
    """``cos``/``sin`` tables ``[P, dim/2]`` (fp32) for the given positions."""
    angles = positions.astype(mx.float32)[:, None] * freqs[None, :]
    return mx.cos(angles), mx.sin(angles)


def apply_interleaved_rope(
    x: mx.array, cos: mx.array, sin: mx.array, inverse: bool = False
) -> mx.array:
    """Rotate interleaved pairs ``(x[2i], x[2i+1])`` — the reference's
    complex-multiply convention. ``inverse=True`` multiplies by the conjugate
    (the post-attention de-rotation of the output's rope dims).

    ``cos``/``sin`` must be broadcastable to ``x``'s shape minus the pair axis.
    Computed in fp32, cast back.
    """
    dtype = x.dtype
    p = x.astype(mx.float32).reshape(*x.shape[:-1], -1, 2)
    e, o = p[..., 0], p[..., 1]
    if inverse:
        sin = -sin
    out = mx.stack([e * cos - o * sin, e * sin + o * cos], axis=-1)
    return out.reshape(x.shape).astype(dtype)


# --- caches ----------------------------------------------------------------


class CompressorState:
    """Decode-phase state + output cache of one Compressor.

    ``kv_state``/``score_state`` are the partial-group accumulators
    (``[B, coff*ratio, coff*head_dim]`` fp32; scores init -inf). ``cache``
    holds the compressed slots ``[B, cap, head_dim]`` with ``n`` valid.
    """

    GROWTH = 256

    def __init__(self) -> None:
        self.kv_state: Optional[mx.array] = None
        self.score_state: Optional[mx.array] = None
        self.cache: Optional[mx.array] = None
        self.n = 0

    def reset(self, batch: int, ratio: int, coff: int, head_dim: int) -> None:
        shape = (batch, coff * ratio, coff * head_dim)
        self.kv_state = mx.zeros(shape, dtype=mx.float32)
        self.score_state = mx.full(shape, -mx.inf, dtype=mx.float32)
        self.cache = None
        self.n = 0

    def valid(self) -> Optional[mx.array]:
        return None if self.cache is None or self.n == 0 else self.cache[:, : self.n]

    def append(self, groups: mx.array) -> None:
        b, g, d = groups.shape
        needed = self.n + g
        if self.cache is None or self.cache.shape[1] < needed:
            cap = ((needed + self.GROWTH - 1) // self.GROWTH) * self.GROWTH
            new = mx.zeros((b, cap, d), dtype=groups.dtype)
            if self.cache is not None and self.n:
                new[:, : self.n] = self.cache[:, : self.n]
            self.cache = new
        self.cache[:, self.n : needed] = groups
        self.n = needed


class DeepSeekV4Cache(_BaseCache):
    """Per-layer session state: the 128-slot sliding ring + the compressor
    states. The ring slot for absolute position ``p`` is ``p % window``; the
    compressed region is append-only — so the whole session is
    ``(arrays, offset)``, which is what makes prefix reuse expressible."""

    def __init__(self, window: int, has_compressor: bool, has_indexer: bool):
        self.window = window
        self.ring: Optional[mx.array] = None
        self.offset = 0
        self.comp = CompressorState() if has_compressor else None
        self.idx = CompressorState() if has_indexer else None

    def is_trimmable(self) -> bool:
        return False  # exact trim/rollback lands with engine integration

    @property
    def state(self):
        # Defensive copies: the live arrays are updated IN PLACE by later
        # steps, and a snapshot that aliases them silently corrupts.
        empty = mx.zeros((0,))

        def snap(a: Optional[mx.array]) -> mx.array:
            return empty if a is None else mx.array(a)

        parts = [snap(self.ring)]
        for cs in (self.comp, self.idx):
            if cs is not None:
                parts += [snap(cs.kv_state), snap(cs.score_state), snap(cs.valid())]
        return tuple(parts)

    @state.setter
    def state(self, v):
        it = iter(v)

        def take() -> Optional[mx.array]:
            a = next(it)
            # Copy again: restoring one snapshot into two caches must not make
            # them alias each other through the shared tuple.
            return None if a.size == 0 else mx.array(a)

        self.ring = take()
        for cs in (self.comp, self.idx):
            if cs is not None:
                cs.kv_state = take()
                cs.score_state = take()
                cs.cache = take()
                cs.n = 0 if cs.cache is None else cs.cache.shape[1]

    @property
    def meta_state(self):
        return (str(self.offset),)

    @meta_state.setter
    def meta_state(self, v):
        self.offset = int(v[0])


# --- modules ---------------------------------------------------------------


class Compressor(nn.Module):
    """Learned gated pooling of ``ratio`` consecutive tokens into one KV slot.

    ``ratio == 4`` uses overlapping windows (``coff = 2``): the projection
    emits two half-spaces — dims ``[:head_dim]`` score the *previous* group's
    overlap, dims ``[head_dim:]`` the current group.
    """

    def __init__(self, dim: int, head_dim: int, ratio: int, rope_dim: int, eps: float):
        super().__init__()
        self.head_dim = head_dim
        self.ratio = ratio
        self.rope_dim = rope_dim
        self.overlap = ratio == COMPRESS_WITH_INDEXER
        coff = self.coff = 2 if self.overlap else 1
        self.ape = mx.zeros((ratio, coff * head_dim), dtype=mx.float32)
        self.wkv = nn.Linear(dim, coff * head_dim, bias=False)
        self.wgate = nn.Linear(dim, coff * head_dim, bias=False)
        self.norm = nn.RMSNorm(head_dim, eps)

    def _finish(
        self, groups: mx.array, freqs: mx.array, first_group: int, dtype
    ) -> mx.array:
        """RMS-norm the pooled groups and rope their tail dims at each group's
        START position — done by scaling the freqs by ``ratio`` so contiguous
        group indices land on positions ``g * ratio``."""
        groups = self.norm(groups.astype(dtype))
        g = groups.shape[1]
        cos, sin = rope_cos_sin(
            freqs * self.ratio, mx.arange(first_group, first_group + g)
        )
        head, tail = groups[..., : -self.rope_dim], groups[..., -self.rope_dim :]
        return mx.concatenate(
            [head, apply_interleaved_rope(tail, cos, sin)], axis=-1
        )

    def __call__(
        self, x: mx.array, state: CompressorState, freqs: mx.array, start_pos: int
    ) -> None:
        b, s, _ = x.shape
        ratio, d, coff = self.ratio, self.head_dim, self.coff
        kv = self.wkv(x).astype(mx.float32)
        score = self.wgate(x).astype(mx.float32)

        if start_pos == 0:
            state.reset(b, ratio, coff, d)
            remainder = s % ratio
            cutoff = s - remainder
            off = ratio if self.overlap else 0
            if self.overlap and cutoff >= ratio:
                state.kv_state[:, :ratio] = kv[:, cutoff - ratio : cutoff]
                state.score_state[:, :ratio] = score[:, cutoff - ratio : cutoff] + self.ape
            if remainder > 0:
                state.kv_state[:, off : off + remainder] = kv[:, cutoff:]
                state.score_state[:, off : off + remainder] = (
                    score[:, cutoff:] + self.ape[:remainder]
                )
            if cutoff == 0:
                return
            kv_g = kv[:, :cutoff].reshape(b, -1, ratio, coff * d)
            score_g = score[:, :cutoff].reshape(b, -1, ratio, coff * d) + self.ape
            if self.overlap:
                kv_g = self._overlap_transform(kv_g, 0.0)
                score_g = self._overlap_transform(score_g, -mx.inf)
            groups = (kv_g * mx.softmax(score_g, axis=2)).sum(axis=2)
            state.append(self._finish(groups, freqs, 0, x.dtype))
            return

        assert s == 1, "continuation prefill is not implemented yet"
        p = start_pos
        kv1, score1 = kv[:, 0], score[:, 0] + self.ape[p % ratio]
        slot = (ratio if self.overlap else 0) + p % ratio
        state.kv_state[:, slot] = kv1
        state.score_state[:, slot] = score1
        if (p + 1) % ratio:
            return
        if self.overlap:
            ks = mx.concatenate(
                [state.kv_state[:, :ratio, :d], state.kv_state[:, ratio:, d:]], axis=1
            )
            ss = mx.concatenate(
                [state.score_state[:, :ratio, :d], state.score_state[:, ratio:, d:]],
                axis=1,
            )
            group = (ks * mx.softmax(ss, axis=1)).sum(axis=1, keepdims=True)
            state.kv_state[:, :ratio] = state.kv_state[:, ratio:]
            state.score_state[:, :ratio] = state.score_state[:, ratio:]
        else:
            group = (state.kv_state * mx.softmax(state.score_state, axis=1)).sum(
                axis=1, keepdims=True
            )
        first_group = (p + 1) // ratio - 1
        state.append(self._finish(group, freqs, first_group, x.dtype))

    def _overlap_transform(self, t: mx.array, fill: float) -> mx.array:
        """[b,g,r,2d] -> [b,g,2r,d]: rows [:r] carry the PREVIOUS group's
        first-half dims (group 0 gets ``fill``), rows [r:] the current group's
        second-half dims."""
        b, g, r, _ = t.shape
        d = self.head_dim
        prev = mx.full((b, g, r, d), fill, dtype=t.dtype)
        if g > 1:
            prev[:, 1:] = t[:, :-1, :, :d]
        return mx.concatenate([prev, t[..., d:]], axis=2)


class Indexer(nn.Module):
    """DSA: scores the compressed slots and returns the top-k slot indices
    (−1-masked where not yet visible) for the ratio-4 layers."""

    def __init__(self, args: "ModelArgs"):
        super().__init__()
        self.n_heads = args.index_n_heads
        self.head_dim = args.index_head_dim
        self.rope_dim = args.qk_rope_head_dim
        self.topk = args.index_topk
        self.ratio = COMPRESS_WITH_INDEXER
        self.wq_b = nn.Linear(args.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.weights_proj = nn.Linear(args.hidden_size, self.n_heads, bias=False)
        self.compressor = Compressor(
            args.hidden_size, self.head_dim, self.ratio, self.rope_dim, args.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        qr: mx.array,
        state: CompressorState,
        freqs: mx.array,
        start_pos: int,
    ) -> Optional[mx.array]:
        b, s, _ = x.shape
        rd = self.rope_dim
        q = self.wq_b(qr).reshape(b, s, self.n_heads, self.head_dim)
        cos, sin = rope_cos_sin(freqs, mx.arange(start_pos, start_pos + s))
        q_tail = apply_interleaved_rope(q[..., -rd:], cos[:, None], sin[:, None])
        q = mx.concatenate([q[..., :-rd], q_tail], axis=-1)

        self.compressor(x, state, freqs, start_pos)
        kvc = state.valid()
        if kvc is None:
            return None
        w = self.weights_proj(x).astype(mx.float32) * (
            self.head_dim**-0.5 * self.n_heads**-0.5
        )
        scores = mx.einsum(
            "bshd,bgd->bshg", q.astype(mx.float32), kvc.astype(mx.float32)
        )
        scores = (mx.maximum(scores, 0) * w[..., None]).sum(axis=2)  # [B,S,G]
        g = kvc.shape[1]
        if start_pos == 0:
            visible = (mx.arange(1, s + 1) // self.ratio)[:, None]
            scores = mx.where(mx.arange(g)[None, :] >= visible, -mx.inf, scores)
        k = min(self.topk, g)
        idxs = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k].astype(mx.int32)
        if start_pos == 0:
            idxs = mx.where(idxs >= visible, MASKED_INDEX, idxs)
        return idxs


def sparse_attend(
    q: mx.array,
    parts: list[tuple[mx.array, mx.array]],
    attn_sink: mx.array,
    scale: float,
) -> mx.array:
    """Attention over gathered slots with the sink in the denominator.

    ``parts`` are ``(kv [B,N,D], idxs [B,S,K])`` pairs; ``idxs == -1`` means
    "no position". K = V = the gathered latent (that is MLA here). The sink
    joins as ``exp(sink - max)`` in the denominator ONLY — it is not an
    attended value. Softmax math in fp32.
    """
    scores, values = [], []
    for kv, idxs in parts:
        safe = mx.maximum(idxs, 0)[..., None]
        gathered = mx.take_along_axis(kv[:, None], safe, axis=2)  # [B,S,K,D]
        s = mx.einsum(
            "bshd,bskd->bshk", q.astype(mx.float32), gathered.astype(mx.float32)
        ) * scale
        s = mx.where((idxs == MASKED_INDEX)[:, :, None, :], -mx.inf, s)
        scores.append(s)
        values.append(gathered.astype(mx.float32))
    s = mx.concatenate(scores, axis=-1)
    v = mx.concatenate(values, axis=2)
    m = mx.max(s, axis=-1)  # [B,S,H]
    m = mx.maximum(m, attn_sink[None, None, :])
    p = mx.exp(s - m[..., None])
    denom = p.sum(axis=-1) + mx.exp(attn_sink[None, None, :] - m)
    o = mx.einsum("bshk,bskd->bshd", p, v) / denom[..., None]
    return o.astype(q.dtype)


class Attention(nn.Module):
    """MLA with ONE 512-wide kv latent (K = V), sliding-window ring, optional
    compression/Indexer, grouped low-rank O projection, and the post-attention
    inverse rope on the output's tail dims."""

    def __init__(self, args: "ModelArgs", layer_id: int):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.head_dim = args.head_dim
        self.rope_dim = args.qk_rope_head_dim
        self.window = args.sliding_window
        self.ratio = args.layer_compress_ratio(layer_id)
        self.n_groups = args.o_groups
        self.eps = args.rms_norm_eps
        self.scale = self.head_dim**-0.5

        self.attn_sink = mx.zeros((self.n_heads,), dtype=mx.float32)
        self.wq_a = nn.Linear(args.hidden_size, args.q_lora_rank, bias=False)
        self.q_norm = nn.RMSNorm(args.q_lora_rank, self.eps)
        self.wq_b = nn.Linear(args.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wkv = nn.Linear(args.hidden_size, self.head_dim, bias=False)
        self.kv_norm = nn.RMSNorm(self.head_dim, self.eps)
        self.wo_a = SwitchLinear(
            self.n_heads * self.head_dim // self.n_groups,
            args.o_lora_rank,
            self.n_groups,
            bias=False,
        )
        self.wo_b = nn.Linear(self.n_groups * args.o_lora_rank, args.hidden_size, bias=False)

        if self.ratio:
            self.compressor = Compressor(
                args.hidden_size, self.head_dim, self.ratio, self.rope_dim, self.eps
            )
            if args.layer_has_indexer(layer_id):
                self.indexer = Indexer(args)
        # Dense (ratio 0) layers use plain rope; compressed layers use
        # compress_rope_theta WITH YaRN — per the reference's per-layer freqs.
        rs = args.rope_scaling or {}
        if self.ratio:
            base, orig = args.compress_rope_theta, rs.get("original_max_position_embeddings", 0)
        else:
            base, orig = args.rope_theta, 0
        self._freqs = yarn_freqs(
            self.rope_dim,
            base,
            orig,
            rs.get("factor", 1.0),
            rs.get("beta_fast", 32.0),
            rs.get("beta_slow", 1.0),
        )

    def __call__(self, x: mx.array, cache: DeepSeekV4Cache) -> mx.array:
        b, s, _ = x.shape
        rd, win = self.rope_dim, self.window
        start = cache.offset
        assert start == 0 or s == 1, "continuation prefill is not implemented yet"
        cos, sin = rope_cos_sin(self._freqs, mx.arange(start, start + s))

        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).reshape(b, s, self.n_heads, self.head_dim)
        qf = q.astype(mx.float32)
        q = (qf * mx.rsqrt(qf.square().mean(-1, keepdims=True) + self.eps)).astype(q.dtype)
        q_tail = apply_interleaved_rope(q[..., -rd:], cos[:, None], sin[:, None])
        q = mx.concatenate([q[..., :-rd], q_tail], axis=-1)

        kv = self.kv_norm(self.wkv(x))
        kv_tail = apply_interleaved_rope(kv[..., -rd:], cos, sin)
        kv = mx.concatenate([kv[..., :-rd], kv_tail], axis=-1)

        if cache.ring is None:
            cache.ring = mx.zeros((b, win, self.head_dim), dtype=x.dtype)

        if start == 0:
            if s <= win:
                cache.ring[:, :s] = kv
            else:
                slots = mx.arange(s - win, s) % win
                cache.ring[:, slots] = kv[:, -win:]
            win_idx = window_topk_indices(win, s, 0)
            parts = [(kv, mx.broadcast_to(win_idx[None], (b, s, win_idx.shape[1])))]
            if self.ratio:
                if "indexer" in self:
                    cidx = self.indexer(x, qr, cache.idx, self._freqs, 0)
                elif s >= self.ratio:
                    cidx = compress_topk_indices(self.ratio, s, 0, 0)
                else:
                    cidx = None
                self.compressor(x, cache.comp, self._freqs, 0)
                comp = cache.comp.valid()
                if comp is not None and cidx is not None and cidx.shape[-1] > 0:
                    if cidx.ndim == 2:
                        cidx = mx.broadcast_to(cidx[None], (b, *cidx.shape))
                    parts.append((comp, cidx))
        else:
            cache.ring[:, start % win] = kv[:, 0]
            win_idx = window_topk_indices(win, 1, start)
            parts = [(cache.ring, mx.broadcast_to(win_idx[None], (b, 1, win)))]
            if self.ratio:
                if "indexer" in self:
                    cidx = self.indexer(x, qr, cache.idx, self._freqs, start)
                else:
                    cidx = None
                self.compressor(x, cache.comp, self._freqs, start)
                comp = cache.comp.valid()
                if comp is not None:
                    if cidx is None:
                        cidx = compress_topk_indices(self.ratio, 1, start, 0)
                        cidx = mx.broadcast_to(cidx[None], (b, *cidx.shape))
                    if cidx.shape[-1] > 0:
                        parts.append((comp, cidx))
        cache.offset = start + s

        o = sparse_attend(q, parts, self.attn_sink, self.scale)
        o_tail = apply_interleaved_rope(o[..., -rd:], cos[:, None], sin[:, None], inverse=True)
        o = mx.concatenate([o[..., :-rd], o_tail], axis=-1)

        o = o.reshape(b, s, self.n_groups, -1)
        groups = mx.broadcast_to(mx.arange(self.n_groups)[None, None], (b, s, self.n_groups))
        o = self.wo_a(o[..., None, :], groups).squeeze(-2)
        return self.wo_b(o.reshape(b, s, -1))


class ClippedSwiGLU:
    """Two-arg activation for SwitchGLU: called as ``activation(up, gate)``."""

    def __init__(self, limit: float):
        self.limit = limit

    def __call__(self, up: mx.array, gate: mx.array) -> mx.array:
        dtype = up.dtype
        return clipped_swiglu(
            gate.astype(mx.float32), up.astype(mx.float32), self.limit
        ).astype(dtype)


class Expert(nn.Module):
    """The shared expert: SwiGLU FFN with V4's clipping, activation in fp32."""

    def __init__(self, dim: int, inter_dim: int, limit: float):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, inter_dim, bias=False)
        self.limit = limit

    def __call__(self, x: mx.array) -> mx.array:
        h = clipped_swiglu(
            self.w1(x).astype(mx.float32), self.w3(x).astype(mx.float32), self.limit
        )
        return self.w2(h.astype(x.dtype))


class Gate(nn.Module):
    """sqrtsoftplus scoring; hash layers select experts by ``tid2eid[token]``
    lookup, score layers by biased top-k — weights always come from the
    UNBIASED scores (see ``route``)."""

    def __init__(self, args: "ModelArgs", layer_id: int):
        super().__init__()
        self.topk = args.num_experts_per_tok
        self.route_scale = args.routed_scaling_factor
        self.hash = args.layer_routes_by_hash(layer_id)
        self.weight = mx.zeros((args.n_routed_experts, args.hidden_size))
        if self.hash:
            self.tid2eid = mx.zeros((args.vocab_size, self.topk), dtype=mx.int32)
        else:
            self.bias = mx.zeros((args.n_routed_experts,), dtype=mx.float32)

    def __call__(self, x: mx.array, input_ids: mx.array) -> tuple[mx.array, mx.array]:
        scores = sqrtsoftplus(x.astype(mx.float32) @ self.weight.T.astype(mx.float32))
        if self.hash:
            indices = self.tid2eid[input_ids]
            weights = mx.take_along_axis(scores, indices, axis=-1)
            weights = weights / weights.sum(axis=-1, keepdims=True)
            return weights * self.route_scale, indices
        return route(scores, self.topk, self.route_scale, self.bias)


class MoE(nn.Module):
    def __init__(self, args: "ModelArgs", layer_id: int):
        super().__init__()
        self.gate = Gate(args, layer_id)
        self.experts = SwitchGLU(
            args.hidden_size,
            args.moe_intermediate_size,
            args.n_routed_experts,
            activation=ClippedSwiGLU(args.swiglu_limit),
            bias=False,
        )
        self.shared_experts = Expert(
            args.hidden_size, args.moe_intermediate_size, args.swiglu_limit
        )

    def __call__(self, x: mx.array, input_ids: mx.array) -> mx.array:
        weights, indices = self.gate(x, input_ids)
        routed = self.experts(x, indices).astype(mx.float32)
        y = (routed * weights[..., None]).sum(axis=-2)
        return (y + self.shared_experts(x).astype(mx.float32)).astype(x.dtype)


# --- Hyper-Connections -----------------------------------------------------


def hc_split_sinkhorn(
    mixes: mx.array, scale: mx.array, base: mx.array, hc: int, iters: int, eps: float
) -> tuple[mx.array, mx.array, mx.array]:
    """Split the mix vector into (pre, post, comb) and Sinkhorn-normalize comb.
    Transcribed from the reference kernel: the FIRST row pass is a softmax
    (+eps); every later pass divides by (sum + eps), rows then columns."""
    pre = mx.sigmoid(mixes[..., :hc] * scale[0] + base[:hc]) + eps
    post = 2 * mx.sigmoid(mixes[..., hc : 2 * hc] * scale[1] + base[hc : 2 * hc])
    comb = (mixes[..., 2 * hc :] * scale[2] + base[2 * hc :]).reshape(
        *mixes.shape[:-1], hc, hc
    )
    comb = mx.softmax(comb, axis=-1) + eps
    comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(axis=-1, keepdims=True) + eps)
        comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
    return pre, post, comb


def _hc_mixes(h: mx.array, fn: mx.array, eps: float) -> tuple[mx.array, mx.array]:
    """Flatten the hc streams, compute the parameterless-RMS-scaled mixes."""
    b, s, hc, d = h.shape
    flat = h.reshape(b, s, hc * d).astype(mx.float32)
    r = mx.rsqrt(flat.square().mean(-1, keepdims=True) + eps)
    return flat, (flat @ fn.T) * r


class Block(nn.Module):
    def __init__(self, args: "ModelArgs", layer_id: int):
        super().__init__()
        self.hc = args.hc_mult
        self.iters = args.hc_sinkhorn_iters
        self.hc_eps = args.hc_eps
        self.eps = args.rms_norm_eps
        self.attn = Attention(args, layer_id)
        self.ffn = MoE(args, layer_id)
        self.attn_norm = nn.RMSNorm(args.hidden_size, self.eps)
        self.ffn_norm = nn.RMSNorm(args.hidden_size, self.eps)
        mix = (2 + self.hc) * self.hc
        hc_dim = self.hc * args.hidden_size
        self.hc_attn_fn = mx.zeros((mix, hc_dim), dtype=mx.float32)
        self.hc_ffn_fn = mx.zeros((mix, hc_dim), dtype=mx.float32)
        self.hc_attn_base = mx.zeros((mix,), dtype=mx.float32)
        self.hc_ffn_base = mx.zeros((mix,), dtype=mx.float32)
        self.hc_attn_scale = mx.zeros((3,), dtype=mx.float32)
        self.hc_ffn_scale = mx.zeros((3,), dtype=mx.float32)

    def _pre(self, h, fn, scale, base):
        flat, mixes = _hc_mixes(h, fn, self.eps)
        pre, post, comb = hc_split_sinkhorn(
            mixes, scale, base, self.hc, self.iters, self.hc_eps
        )
        y = (pre[..., None] * flat.reshape(h.shape)).sum(axis=2)
        return y.astype(h.dtype), post, comb

    @staticmethod
    def _post(x, residual, post, comb):
        y = post[..., None] * x[..., None, :] + mx.einsum(
            "bsjk,bsjd->bskd", comb, residual.astype(mx.float32)
        )
        return y.astype(x.dtype)

    def __call__(self, h: mx.array, input_ids: mx.array, cache) -> mx.array:
        residual = h
        x, post, comb = self._pre(h, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        x = self.attn(self.attn_norm(x), cache)
        h = self._post(x, residual, post, comb)

        residual = h
        x, post, comb = self._pre(h, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        x = self.ffn(self.ffn_norm(x), input_ids)
        return self._post(x, residual, post, comb)


class Model(nn.Module):
    """DeepSeek-V4. Module names mirror the checkpoint's reference naming
    (``layers.N.attn.wq_a`` ...) so weight loading is 1:1 after dequant; the
    ``mtp.*`` blocks are deferred."""

    def __init__(self, args: "ModelArgs"):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.hc = args.hc_mult
        self.eps = args.rms_norm_eps
        self.hc_eps = args.hc_eps
        self.embed = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [Block(args, i) for i in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, self.eps)
        self.head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        self.hc_head_fn = mx.zeros(
            (self.hc, self.hc * args.hidden_size), dtype=mx.float32
        )
        self.hc_head_base = mx.zeros((self.hc,), dtype=mx.float32)
        self.hc_head_scale = mx.zeros((1,), dtype=mx.float32)

    def make_cache(self) -> list[DeepSeekV4Cache]:
        return [
            DeepSeekV4Cache(
                self.args.sliding_window,
                has_compressor=self.args.layer_compress_ratio(i) > 0,
                has_indexer=self.args.layer_has_indexer(i),
            )
            for i in range(self.args.num_hidden_layers)
        ]

    def __call__(self, inputs: mx.array, cache=None) -> mx.array:
        if cache is None:
            cache = self.make_cache()
        h = self.embed(inputs)
        h = mx.broadcast_to(h[:, :, None, :], (*h.shape[:2], self.hc, h.shape[-1]))
        for layer, c in zip(self.layers, cache):
            h = layer(h, inputs, c)
        flat, mixes = _hc_mixes(h, self.hc_head_fn, self.eps)
        pre = mx.sigmoid(mixes * self.hc_head_scale + self.hc_head_base) + self.hc_eps
        x = (pre[..., None] * flat.reshape(h.shape)).sum(axis=2).astype(h.dtype)
        return self.head(self.norm(x))
