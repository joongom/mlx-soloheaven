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

import functools
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, NamedTuple, Optional

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

    Port of ``get_compress_topk_idxs``, generalized to CONTINUATION chunks
    (``start_pos > 0`` with ``seqlen > 1``, which the reference never sees —
    it only prefills from zero or decodes one token). ``offset`` is where the
    compressed region starts inside a shared buffer (0 when the compressed KV
    is its own buffer).

    A compressed slot only becomes visible once its whole ``ratio``-token
    group has been consumed, which is what the ``>=`` mask encodes; at the
    reference's two regimes this reduces exactly to the reference output
    (checked in tests against the transcription).
    """
    width = (start_pos + seqlen) // ratio
    matrix = mx.broadcast_to(mx.arange(width)[None, :], (seqlen, width))
    visible = (mx.arange(start_pos + 1, start_pos + seqlen + 1) // ratio)[:, None]
    return mx.where(matrix >= visible, MASKED_INDEX, matrix + offset).astype(mx.int32)


def continuation_window_indices(
    window_size: int, seqlen: int, start_pos: int
) -> mx.array:
    """Window indices for a continuation chunk, into the VIRTUAL buffer
    ``concat([ring (window_size slots), chunk (seqlen)], axis=1)``.

    Query row ``i`` (absolute position ``p = start_pos + i``) attends to
    positions ``p-window+1 .. p``: in-chunk positions map past the ring
    (``window_size + (q - start_pos)``), older positions live in the ring at
    slot ``q % window_size`` (guaranteed present — the ring holds the last
    ``window_size`` positions), and negative positions are masked.
    """
    p = mx.arange(start_pos, start_pos + seqlen)[:, None]
    q = p - window_size + 1 + mx.arange(window_size)[None, :]
    ring_slot = mx.maximum(q, 0) % window_size
    idx = mx.where(q >= start_pos, q - start_pos + window_size, ring_slot)
    return mx.where(q < 0, MASKED_INDEX, idx).astype(mx.int32)


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

        # Continuation (start_pos > 0, any seqlen): finish the partially
        # filled group from state, bulk-pool the full groups in the chunk, and
        # stash the remainder back into state. The single-token decode step is
        # the seqlen == 1 special case of this path.
        off = ratio if self.overlap else 0
        groups: list[mx.array] = []
        pos = 0
        filled = start_pos % ratio
        if filled:
            n = min(s, ratio - filled)
            state.kv_state[:, off + filled : off + filled + n] = kv[:, :n]
            state.score_state[:, off + filled : off + filled + n] = (
                score[:, :n] + self.ape[filled : filled + n]
            )
            pos = n
            if filled + n == ratio:
                groups.append(self._pool_state(state))
        nfull = (s - pos) // ratio
        if nfull:
            kv_g = kv[:, pos : pos + nfull * ratio].reshape(b, nfull, ratio, coff * d)
            sc_g = (
                score[:, pos : pos + nfull * ratio].reshape(b, nfull, ratio, coff * d)
                + self.ape
            )
            if self.overlap:
                # Chain the overlap halves: group j's first rows come from
                # group j-1's raw kv; group 0's come from the last COMPLETED
                # group, which state[:ratio] holds by invariant (prefill and
                # _pool_state both maintain it).
                prev_kv = mx.concatenate(
                    [state.kv_state[:, None, :ratio], kv_g[:, :-1]], axis=1
                )[..., :d]
                prev_sc = mx.concatenate(
                    [state.score_state[:, None, :ratio], sc_g[:, :-1]], axis=1
                )[..., :d]
                kv_o = mx.concatenate([prev_kv, kv_g[..., d:]], axis=2)
                sc_o = mx.concatenate([prev_sc, sc_g[..., d:]], axis=2)
                groups.append((kv_o * mx.softmax(sc_o, axis=2)).sum(axis=2))
                state.kv_state[:, :ratio] = kv_g[:, -1]
                state.score_state[:, :ratio] = sc_g[:, -1]
            else:
                groups.append((kv_g * mx.softmax(sc_g, axis=2)).sum(axis=2))
            pos += nfull * ratio
        r = s - pos
        if r:  # start of a new partial group (start_pos + pos is a boundary)
            state.kv_state[:, off : off + r] = kv[:, pos:]
            state.score_state[:, off : off + r] = score[:, pos:] + self.ape[:r]
        if groups:
            state.append(
                self._finish(
                    mx.concatenate(groups, axis=1), freqs, start_pos // ratio, x.dtype
                )
            )

    def decode_step_math(
        self,
        proj: tuple[mx.array, mx.array],
        out_dtype,
        kv_state: mx.array,
        score_state: mx.array,
        buf: mx.array,
        n: mx.array,
        offset: mx.array,
        freqs: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        """BRANCHLESS single-token compressor step for the compiled decode
        path. ``offset``/``n`` are traced int32 scalars, so nothing here may
        branch on their VALUES in Python — group completion is a ``where``
        mask, and the output row is always written (with the OLD value when
        the group is incomplete). This keeps one trace valid for every token;
        a Python branch on offset would retrace per position.

        Returns (kv_state, score_state, buf, new_n), all functionally updated.
        """
        ratio, d = self.ratio, self.head_dim
        off_base = ratio if self.overlap else 0
        filled = offset % ratio
        kv1 = proj[0].astype(mx.float32)[:, 0]
        sc1 = proj[1].astype(mx.float32)[:, 0] + mx.take(self.ape, filled, axis=0)
        slot = off_base + filled
        kv_state[:, slot] = kv1
        score_state[:, slot] = sc1
        complete = ((offset + 1) % ratio) == 0
        if self.overlap:
            ks = mx.concatenate(
                [kv_state[:, :ratio, :d], kv_state[:, ratio:, d:]], axis=1
            )
            ss = mx.concatenate(
                [score_state[:, :ratio, :d], score_state[:, ratio:, d:]], axis=1
            )
            pooled = (ks * mx.softmax(ss, axis=1)).sum(axis=1, keepdims=True)
            kv_state = mx.concatenate(
                [mx.where(complete, kv_state[:, ratio:], kv_state[:, :ratio]),
                 kv_state[:, ratio:]],
                axis=1,
            )
            score_state = mx.concatenate(
                [mx.where(complete, score_state[:, ratio:], score_state[:, :ratio]),
                 score_state[:, ratio:]],
                axis=1,
            )
        else:
            pooled = (kv_state * mx.softmax(score_state, axis=1)).sum(
                axis=1, keepdims=True
            )
        g = self.norm(pooled.astype(out_dtype))
        gpos = ((offset + 1) // ratio - 1).reshape(1)
        cos, sin = rope_cos_sin(freqs * ratio, gpos)
        tail = apply_interleaved_rope(g[..., -self.rope_dim :], cos, sin)
        g = mx.concatenate([g[..., : -self.rope_dim], tail], axis=-1)
        old = mx.take(buf, n, axis=1)
        buf[:, n] = mx.where(complete, g[:, 0], old)
        return kv_state, score_state, buf, n + complete.astype(n.dtype)

    def _pool_state(self, state: CompressorState) -> mx.array:
        """Pool the completed group held in ``state`` into one slot and (for
        overlap) shift the current group into the previous-group region."""
        ratio, d = self.ratio, self.head_dim
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
        return group

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
        # Per-row causal visibility: row i (absolute position start_pos + i)
        # sees only groups completed by then. At single-token decode this
        # masks nothing (g == (start_pos + 1) // ratio), matching the
        # reference's unmasked decode branch.
        visible = ((mx.arange(s) + start_pos + 1) // self.ratio)[:, None]
        scores = mx.where(mx.arange(g)[None, :] >= visible, -mx.inf, scores)
        k = min(self.topk, g)
        idxs = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k].astype(mx.int32)
        return mx.where(idxs >= visible, MASKED_INDEX, idxs)

    def decode_step_math(
        self, proj, w_raw, out_dtype, qr, kv_state, score_state, buf, n, offset, freqs
    ):
        """Branchless single-token DSA step for the compiled decode path.
        Scores the FULL capacity buffer (invisible slots masked) so the trace
        is capacity-keyed, not group-count-keyed. ``proj``/``w_raw`` come from
        the layer's stacked x-projection."""
        kv_state, score_state, buf, n2 = self.compressor.decode_step_math(
            proj, out_dtype, kv_state, score_state, buf, n, offset, freqs
        )
        rd = self.rope_dim
        q = self.wq_b(qr).reshape(1, 1, self.n_heads, self.head_dim)
        cos, sin = rope_cos_sin(freqs, offset.reshape(1))
        tail = apply_interleaved_rope(q[..., -rd:], cos[:, None], sin[:, None])
        q = mx.concatenate([q[..., :-rd], tail], axis=-1)
        w = w_raw.astype(mx.float32) * (self.head_dim**-0.5 * self.n_heads**-0.5)
        cap = buf.shape[1]
        scores = mx.einsum(
            "bshd,bgd->bshg", q.astype(mx.float32), buf.astype(mx.float32)
        )
        scores = (mx.maximum(scores, 0) * w[..., None]).sum(axis=2)  # [1,1,cap]
        scores = mx.where(mx.arange(cap) >= n2, -mx.inf, scores)
        k = min(self.topk, cap)
        idxs = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k].astype(mx.int32)
        idxs = mx.where(idxs >= n2, MASKED_INDEX, idxs)
        return idxs, kv_state, score_state, buf, n2


# --- fused decode attention (Metal) ----------------------------------------
#
# Eager sparse_attend issues ~25 ops per layer per decoded token (gathers,
# fp32 einsums, softmax pieces); the dispatch gaps between them are pure tax
# (see the decode-speed section of the port spec — ds4 solves the same
# problem with a fused kernel). This kernel does the whole thing in ONE
# dispatch per layer: one threadgroup per head, online softmax with the sink
# in the denominator, MASKED_INDEX slots skipped, both parts (ring buffer +
# compressed cache) walked inside the kernel. Decode only (b=1, s=1);
# everything else takes the eager path, which stays the canonical reference
# the kernel is differential-tested against.
#
# The structure follows DeepSeek's MIT-licensed sparse_attn_kernel
# (inference/kernel.py) — see LICENSE notes in the README acknowledgments.

_DECODE_TG = 128  # threads per threadgroup (one threadgroup per head)
_DECODE_MAX_D = 512
_DECODE_MAX_K = 1024

_DECODE_KERNEL_SRC = """
    uint h = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    const int D = params[0];
    const int K1 = params[1];
    const int K2 = params[2];
    const int K = K1 + K2;
    const int TG = 128;

    threadgroup float qh[512];
    threadgroup float sc[1024];
    threadgroup float red[128];

    for (int d = tid; d < D; d += TG) qh[d] = float(q[h * D + d]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // scores (masked slots stay -INFINITY)
    for (int j = tid; j < K; j += TG) {
        float s = -INFINITY;
        if (j < K1) {
            int idx = idxs1[j];
            if (idx >= 0) {
                float acc = 0.0f;
                for (int d = 0; d < D; ++d) acc += qh[d] * float(kv1[idx * D + d]);
                s = acc * scale[0];
            }
        } else {
            int idx = idxs2[j - K1];
            if (idx >= 0) {
                float acc = 0.0f;
                for (int d = 0; d < D; ++d) acc += qh[d] * float(kv2[idx * D + d]);
                s = acc * scale[0];
            }
        }
        sc[j] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // max over scores, then include the sink
    float lm = -INFINITY;
    for (int j = tid; j < K; j += TG) lm = max(lm, sc[j]);
    red[tid] = lm;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int off = TG / 2; off > 0; off >>= 1) {
        if (tid < (uint)off) red[tid] = max(red[tid], red[tid + off]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float m = max(red[0], sink[h]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // exp + sum; the sink joins the DENOMINATOR only
    float ls = 0.0f;
    for (int j = tid; j < K; j += TG) {
        float p = exp(sc[j] - m);
        sc[j] = p;
        ls += p;
    }
    red[tid] = ls;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int off = TG / 2; off > 0; off >>= 1) {
        if (tid < (uint)off) red[tid] += red[tid + off];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float denom = red[0] + exp(sink[h] - m);

    // weighted values (p == 0 covers masked slots, so idx is never read there)
    for (int d = tid; d < D; d += TG) {
        float acc = 0.0f;
        for (int j = 0; j < K1; ++j) {
            float p = sc[j];
            if (p > 0.0f) acc += p * float(kv1[idxs1[j] * D + d]);
        }
        for (int j = 0; j < K2; ++j) {
            float p = sc[K1 + j];
            if (p > 0.0f) acc += p * float(kv2[idxs2[j] * D + d]);
        }
        out[h * D + d] = acc / denom;
    }
"""

#: v2 "attention core": absorbs the decode glue that the per-op cost model
#: (docs/benchmarks/deepseek-v4.md) says dominates — in-kernel rope table
#: from (offset x freqs), parameterless per-head q-RMS, q/kv rope, window
#: index generation, plain-comp visibility indices, online-softmax attention
#: with the sink, and the inverse rope on the output tail. Emits the roped
#: kv row for the (python-side) ring write. One dispatch replaces ~30 ops.
_ATTN_CORE_SRC = """
    // grid: one threadgroup (128 threads) per head; head h.
    uint h_ = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    const int TG = 512;   // 64 heads x 128 = 8K threads underfilled the chip
    const int D = params[0];        // head_dim (512)
    const int RD = params[1];       // rope dim (64)
    const int WIN = params[2];      // sliding window
    const int KC = params[3];       // comp part width (0 = none)
    const int PLAIN = params[4];    // 1: cidx generated in-kernel from n
    const float scale = fscal[0];
    const float eps = fscal[1];
    const int offset = ioff[0];
    const int NCOMP = ioff[1];      // valid comp groups (traced; plain mask)

    threadgroup float qh[512];
    threadgroup float kvr[512];     // roped kv row (fresh token)
    threadgroup float cs[64];       // cos/sin for RD/2 pairs at pos=offset
    threadgroup float sc[2176];
    threadgroup float red[128];

    // rope table for position = offset
    for (int i = tid; i < RD / 2; i += TG) {
        float ang = float(offset) * freqs[i];
        cs[2 * i] = cos(ang);
        cs[2 * i + 1] = sin(ang);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // roped kv row (computed redundantly per TG; 512 dims, trivial)
    for (int i = tid; i < D; i += TG) kvr[i] = float(kv[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int p = tid; p < RD / 2; p += TG) {
        int i0 = D - RD + 2 * p;
        float c = cs[2 * p], s = cs[2 * p + 1];
        float e = kvr[i0], o = kvr[i0 + 1];
        kvr[i0] = e * c - o * s;
        kvr[i0 + 1] = e * s + o * c;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (h_ == 0) {
        // The roped KV row IS this token's ring entry, so store it here rather
        // than paying a separate dispatch (dsv4_ring_store_k, 43/token). Safe:
        // the window loop below never reads slot offset%WIN — qpos == offset
        // takes kvr directly — so no threadgroup can see a half-written slot.
        // RING_WRITE is a no-op in the mx.fast twin, whose `ring` is a const
        // input; the native wrapper defines it as the actual store.
        for (int i = tid; i < D; i += TG) {
            kv_out[i] = T(kvr[i]);
            RING_WRITE(offset % WIN, i, T(kvr[i]));
        }
    }

    // q: load head h, parameterless RMS over D, rope the tail
    float ss = 0.0f;
    for (int i = tid; i < D; i += TG) {
        float v = float(q[h_ * D + i]);
        qh[i] = v;
        ss += v * v;
    }
    ss = simd_sum(ss);
    if ((tid & 31u) == 0) red[tid / 32] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float t = 0.0f;
        for (int i = 0; i < TG / 32; ++i) t += red[i];
        red[0] = rsqrt(t / D + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rn = red[0];
    for (int i = tid; i < D; i += TG) qh[i] *= rn;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int p = tid; p < RD / 2; p += TG) {
        int i0 = D - RD + 2 * p;
        float c = cs[2 * p], s = cs[2 * p + 1];
        float e = qh[i0], o = qh[i0 + 1];
        qh[i0] = e * c - o * s;
        qh[i0 + 1] = e * s + o * c;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // scores: window part (ring slots computed inline; slot==offset%WIN is
    // the fresh token -> use kvr) then comp part.
    // Only [jlo, jhi) can hold valid entries: window slots below jlo have
    // qpos < 0, and BOTH comp modes expose a contiguous valid prefix of
    // min(NCOMP, KC) groups (plain masks past NCOMP; idx_topk emits its
    // winners as a -1-padded prefix and NCOMP is the n2 it saw). The window
    // and comp regions are contiguous, so the loops run O(context) instead
    // of O(window capacity).
    // One SIMDGROUP per slot, lanes split the D dims: a single thread
    // serially walking a 512-dim row exposes full DRAM latency per element
    // (~138 us/dispatch, Stage 3k bisection); 32 coalesced lanes + simd_sum
    // hide it.
    const int K = WIN + KC;
    const int jlo = (offset + 1 < WIN) ? (WIN - 1 - offset) : 0;
    const int jhi = WIN + min(NCOMP, KC);
    uint sg_ = simdgroup_index_in_threadgroup;
    uint lane_ = thread_index_in_simdgroup;
    for (int j = jlo + (int)sg_; j < jhi; j += TG / 32) {
        float s = -INFINITY;
        if (j < WIN) {
            int qpos = offset - WIN + 1 + j;
            float a = 0.0f;
            if (qpos == offset) {
                for (int i = lane_; i < D; i += 32) a += qh[i] * kvr[i];
            } else {
                int slot = qpos % WIN;
                for (int i = lane_; i < D; i += 32) a += qh[i] * float(ring[slot * D + i]);
            }
            s = simd_sum(a) * scale;
        } else {
            int idx = PLAIN ? (j - WIN) : cidx[j - WIN];
            if (idx >= 0) {
                float a = 0.0f;
                for (int i = lane_; i < D; i += 32) a += qh[i] * float(comp[idx * D + i]);
                s = simd_sum(a) * scale;
            }
        }
        if (lane_ == 0) sc[j] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float lm = -INFINITY;
    for (int j = jlo + (int)tid; j < jhi; j += TG) lm = max(lm, sc[j]);
    lm = simd_max(lm);
    if ((tid & 31u) == 0) red[tid / 32] = lm;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float m = sink[h_];
        for (int i = 0; i < TG / 32; ++i) m = max(m, red[i]);
        red[0] = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float m = red[0];
    float ls = 0.0f;
    for (int j = jlo + (int)tid; j < jhi; j += TG) {
        float p = exp(sc[j] - m);
        sc[j] = p;
        ls += p;
    }
    ls = simd_sum(ls);
    if ((tid & 31u) == 0) red[tid / 32] = ls;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float t = exp(sink[h_] - m);
        for (int i = 0; i < TG / 32; ++i) t += red[i];
        red[0] = t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float denom = red[0];

    // values, pairwise so the inverse rope on the tail applies in-register
    for (int p = tid; p < D / 2; p += TG) {
        int i0 = 2 * p;
        float a0 = 0.0f, a1 = 0.0f;
        for (int j = jlo; j < jhi; ++j) {
            float pj = sc[j];
            if (pj <= 0.0f) continue;
            if (j < WIN) {
                int qpos = offset - WIN + 1 + j;
                if (qpos == offset) { a0 += pj * kvr[i0]; a1 += pj * kvr[i0 + 1]; }
                else {
                    int slot = qpos % WIN;
                    a0 += pj * float(ring[slot * D + i0]);
                    a1 += pj * float(ring[slot * D + i0 + 1]);
                }
            } else {
                int idx = PLAIN ? (j - WIN) : cidx[j - WIN];
                a0 += pj * float(comp[idx * D + i0]);
                a1 += pj * float(comp[idx * D + i0 + 1]);
            }
        }
        a0 /= denom;
        a1 /= denom;
        if (i0 >= D - RD) {                    // inverse rope (conjugate)
            int pr = (i0 - (D - RD)) / 2;
            float c = cs[2 * pr], s = cs[2 * pr + 1];
            float e = a0, o = a1;
            a0 = e * c + o * s;
            a1 = o * c - e * s;
        }
        out[h_ * D + i0] = T(a0);
        out[h_ * D + i0 + 1] = T(a1);
    }
"""

#: Fused compressor decode step: the whole branchless state machine —
#: state-slot write (+ape), completion mask, overlap/plain pooling, RMS-norm,
#: group rope, state shift — as ONE dispatch. Called 61x per token
#: (comp x41 + indexer-comp x21 - dense); replaces ~12 small ops each.
#: State is [coff*ratio, coff*d] (tiny), so a single threadgroup is
#: appropriate here — there is no large GEMV inside to starve (the
#: projections arrive precomputed from the stacked matmul).
_COMP_STEP_SRC = """
    uint tid = thread_position_in_threadgroup.x;
    const int TG = 1024;  // single threadgroup; widest allowed hides the state-copy latency
    const int ratio = params[0];
    const int d = params[1];
    const int coff = params[2];     // 2 = overlap, 1 = plain
    const int RD = params[3];
    const int offset = ioff[0];
    const float eps = feps[0];
    const int cd = coff * d;
    const int rows = coff * ratio;
    const int filled = offset % ratio;
    const int slot = (coff == 2 ? ratio : 0) + filled;
    const bool complete = ((offset + 1) % ratio) == 0;

    threadgroup float pooled[512];
    threadgroup float wsum[32];  // TG/32 simdgroup partials

    // State is updated IN PLACE: only the fresh slot row is written (the
    // Stage 3o bisection measured the old full-state double-buffer copy at
    // 2.5 ms/token). The pooling below never reads state row `slot` (it
    // redirects to the fresh kv_row/sc_row), so this write races nothing;
    // the overlap-head shift happens after the pooling barrier.
    for (int i = tid; i < cd; i += TG) {
        kv_st[slot * cd + i] = float(kv_row[i]);
        sc_st[slot * cd + i] = float(sc_row[i]) + ape[filled * cd + i];
    }

    // pooled[i] = softmax-weighted sum over rows of W (pre-shift view).
    // ONLINE softmax (single pass with running rescale) — the two-pass form
    // walked the row-strided state twice per element (Stage 3o bisection:
    // 2.3 ms/token). The mn > -inf guard skips empty (-inf score) rows so
    // the -inf - -inf = NaN intermediate never forms; row `slot` is always
    // fresh, so the final m is finite.
    for (int i = tid; i < d; i += TG) {
        float m = -INFINITY, den = 0.0f, acc = 0.0f;
        for (int r = 0; r < rows; ++r) {
            int col = (coff == 2) ? ((r < ratio) ? i : i + d) : i;
            float scv = (r == slot) ? float(sc_row[col]) + ape[filled * cd + col]
                                    : sc_st[r * cd + col];
            float kvv = (r == slot) ? float(kv_row[col]) : kv_st[r * cd + col];
            float mn = max(m, scv);
            if (mn > -INFINITY) {
                float rs = (m == -INFINITY) ? 0.0f : exp(m - mn);
                float e = exp(scv - mn);
                den = den * rs + e;
                acc = acc * rs + e * kvv;
                m = mn;
            }
        }
        pooled[i] = acc / den;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

    // overlap-head shift on completion: head rows [0, ratio) take the tail
    // rows [ratio, 2*ratio), which include this token's freshly written slot.
    // Reads and writes are disjoint row ranges, and the barrier above ordered
    // the shift after every pooling read of the old head.
    if (coff == 2 && complete) {
        for (int idx = tid; idx < ratio * cd; idx += TG) {
            kv_st[idx] = kv_st[ratio * cd + idx];
            sc_st[idx] = sc_st[ratio * cd + idx];
        }
    }

    // weighted RMS norm over d, rope the tail at the group-start position
    float ssq = 0.0f;
    for (int i = tid; i < d; i += TG) ssq += pooled[i] * pooled[i];
    ssq = simd_sum(ssq);
    if ((tid & 31u) == 0) wsum[tid / 32] = ssq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float t = 0.0f;
        for (int i = 0; i < TG / 32; ++i) t += wsum[i];
        wsum[0] = rsqrt(t / d + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rn = wsum[0];
    int gpos = (offset + 1) / ratio - 1;
    for (int i = tid; i < d; i += TG) {
        float v = pooled[i] * rn * float(nw[i]);
        if (i >= d - RD) {
            int p2 = (i - (d - RD)) / 2;
            bool hi = ((i - (d - RD)) & 1) == 1;
            float ang = float(gpos) * float(freqs[p2]) * float(ratio);
            float c = cos(ang), s = sin(ang);
            int i0 = d - RD + 2 * p2;
            float e = pooled[i0] * rn * float(nw[i0]);
            float o = pooled[i0 + 1] * rn * float(nw[i0 + 1]);
            v = hi ? (e * s + o * c) : (e * c - o * s);
        }
        row_out[i] = T(complete ? v : float(old_row[i]));
    }
"""

_attn_core_kernel = None


def _get_attn_core_kernel():
    global _attn_core_kernel
    if _attn_core_kernel is None:
        _attn_core_kernel = mx.fast.metal_kernel(
            name="sh_dsv4_attn_core",
            input_names=["q", "kv", "ring", "comp", "cidx", "sink", "freqs",
                         "params", "fscal", "ioff"],
            output_names=["out", "kv_out"],
            source=_ATTN_CORE_SRC,
            header="#define RING_WRITE(s, i, v) ((void)0)\n",
        )
    return _attn_core_kernel


@functools.lru_cache(maxsize=256)
def _attn_core_params(d, rd, win, kc, plain):
    return mx.array([d, rd, win, kc, plain], dtype=mx.int32)


@functools.lru_cache(maxsize=64)
def _attn_core_fscal(scale: float, eps: float):
    return mx.array([scale, eps], dtype=mx.float32)


_decode_kernel = None


@functools.lru_cache(maxsize=1024)
def _kernel_params(d: int, k1: int, k2: int) -> mx.array:
    """Host->device upload of the tiny params array happens once per shape,
    not once per layer per token (86 uploads/token otherwise)."""
    return mx.array([d, k1, k2], dtype=mx.int32)


@functools.lru_cache(maxsize=64)
def _kernel_scale(scale: float) -> mx.array:
    return mx.array([scale], dtype=mx.float32)


@functools.lru_cache(maxsize=8)
def _kernel_dummy_part(d: int, dtype_name: str):
    kv = mx.zeros((1, 1, d), dtype=getattr(mx, dtype_name))
    idx = mx.full((1, 1, 1), MASKED_INDEX, dtype=mx.int32)
    return kv, idx


def _get_decode_kernel():
    global _decode_kernel
    if _decode_kernel is None:
        _decode_kernel = mx.fast.metal_kernel(
            name="sh_dsv4_sparse_decode",
            input_names=["q", "kv1", "idxs1", "kv2", "idxs2", "sink", "scale", "params"],
            output_names=["out"],
            source=_DECODE_KERNEL_SRC,
        )
    return _decode_kernel


def _sparse_attend_decode_metal(
    q: mx.array,
    parts: list[tuple[mx.array, mx.array]],
    attn_sink: mx.array,
    scale: float,
) -> mx.array:
    h, d = q.shape[2], q.shape[3]
    kv1, i1 = parts[0]
    if len(parts) == 2:
        kv2, i2 = parts[1]
    else:  # dense layer: dummy second part, one fully-masked slot
        kv2, i2 = _kernel_dummy_part(d, str(q.dtype).split(".")[-1])
    out = _get_decode_kernel()(
        inputs=[
            q.reshape(h, d),
            kv1.reshape(-1, d),
            i1.reshape(-1),
            kv2.reshape(-1, d),
            i2.reshape(-1),
            attn_sink,
            _kernel_scale(scale),
            _kernel_params(d, i1.size, i2.size),
        ],
        grid=(h * _DECODE_TG, 1, 1),
        threadgroup=(_DECODE_TG, 1, 1),
        output_shapes=[(h, d)],
        output_dtypes=[mx.float32],
    )[0]
    return out.reshape(1, 1, h, d).astype(q.dtype)


def _decode_kernel_usable(q: mx.array, parts) -> bool:
    if os.environ.get("SOLOHEAVEN_METAL_KERNELS", "1") == "0":
        return False
    if not mx.metal.is_available():
        return False
    if q.shape[0] != 1 or q.shape[1] != 1 or len(parts) > 2:
        return False
    if q.shape[3] > _DECODE_MAX_D:
        return False
    k = sum(p[1].shape[-1] for p in parts)
    return k + (1 if len(parts) == 1 else 0) <= _DECODE_MAX_K


def sparse_attend(
    q: mx.array,
    parts: list[tuple[mx.array, mx.array]],
    attn_sink: mx.array,
    scale: float,
) -> mx.array:
    if _decode_kernel_usable(q, parts):
        return _sparse_attend_decode_metal(q, parts, attn_sink, scale)
    return _sparse_attend_eager(q, parts, attn_sink, scale)


# --- whole-layer compiled decode (spec: decode-speed path 2) ----------------
#
# One mx.compile'd function per (layer, cache-capacity signature) covering the
# ENTIRE decode step — HC pre, attention with cache updates, MoE, HC post —
# so a token costs ~43 compiled-graph dispatches instead of thousands of eager
# ones. State is threaded FUNCTIONALLY: cache arrays go in, updated arrays
# come out, and the Python side commits them after all layers succeed. The
# traced `offset`/`n` scalars keep one trace valid for every token; traces are
# keyed only by the compressed-cache capacities (which step in 256-group
# increments). Kill switch: SOLOHEAVEN_COMPILE_DECODE=0.


_COMPILED_DECODE_BROKEN = False


def _COMPILED_DECODE_ENABLED() -> bool:
    return (
        not _COMPILED_DECODE_BROKEN
        and os.environ.get("SOLOHEAVEN_COMPILE_DECODE", "1") != "0"
    )


# Native Metal replay decode (external command-buffer loop, ~1.85x the
# compiled path — docs/benchmarks/deepseek-v4.md Stage 3). Opt-in; any
# failure falls back to the compiled/eager path for the process lifetime.
_NATIVE_DECODE_BROKEN = False


#: Names this switch has had. A stale export is silent — the server starts,
#: serves correctly, and is simply 2x slower — which is exactly how it went
#: unnoticed for a whole session. Say so once, loudly, instead.
_NATIVE_FLAG = "SOLOHEAVEN_NATIVE_DECODE"
_NATIVE_FLAG_FORMER = ("SOLOHEAVEN_DSV4_NATIVE",)


def _warn_stale_native_flag() -> None:
    """Warn once if a RETIRED name for the native switch is exported."""
    global _WARNED_STALE_FLAG
    if _WARNED_STALE_FLAG or os.environ.get(_NATIVE_FLAG):
        return
    stale = [n for n in _NATIVE_FLAG_FORMER if os.environ.get(n)]
    if stale:
        import logging

        _WARNED_STALE_FLAG = True
        logging.getLogger(__name__).warning(
            "[deepseek_v4] %s is set but was RENAMED to %s — native decode is "
            "OFF and this build is running the ~2x slower compiled path. "
            "Export %s=1 instead.",
            ", ".join(stale), _NATIVE_FLAG, _NATIVE_FLAG)


_WARNED_STALE_FLAG = False


def _NATIVE_DECODE_ENABLED() -> bool:
    _warn_stale_native_flag()
    return (
        not _NATIVE_DECODE_BROKEN
        and os.environ.get(_NATIVE_FLAG, "0") == "1"
    )


@functools.lru_cache(maxsize=4096)
def _scalar_i32(v: int) -> mx.array:
    return mx.array(v, dtype=mx.int32)


@functools.lru_cache(maxsize=8)
def _comp_target_capacity(ratio: int) -> int:
    """Campaign Stage 2: preallocate compressed caches to the max-context
    capacity so decode buffers keep ONE address and ONE shape for a whole
    session — the precondition for command-buffer replay (and it removes
    the 256-group trace re-keying as a side effect). Default 32K context
    costs ~220 MB across all layers; override via env for longer sessions.
    """
    max_ctx = int(os.environ.get("SOLOHEAVEN_NATIVE_MAX_CONTEXT", "32768"))
    g = CompressorState.GROWTH
    return max(g, ((max_ctx // ratio + g - 1) // g) * g)


def _ensure_comp_capacity(cs: CompressorState, comp: "Compressor", dtype) -> None:
    """Decode-side capacity management, kept OUTSIDE the compiled function:
    make sure the buffers exist at the SESSION-STABLE capacity (see
    _comp_target_capacity); growth beyond it stays possible but re-keys the
    trace and breaks replay for that session."""
    if cs.kv_state is None:
        cs.reset(1, comp.ratio, comp.coff, comp.head_dim)
    target = _comp_target_capacity(comp.ratio)
    if cs.cache is None:
        cs.cache = mx.zeros((1, target, comp.head_dim), dtype=dtype)
        return
    # Reaching `target` must NOT wait until the buffer is already full.
    # Prefill grows the cache itself (CompressorState.append, rounded to
    # GROWTH), so a short first prompt leaves a 256-slot buffer here — and an
    # "only grow when cs.n >= shape[1]" rule never lifts it to `target`. The
    # compiled path survives that (it re-checks every step); the native one
    # does not: it registers these buffers ONCE per decoder and afterwards
    # only advances `n`, so a long turn wrote ~18 groups PAST a 256-slot
    # indexer cache, corrupting the neighbouring MLX allocation (garbage
    # output, then a 500 on the next prefill). Grow eagerly instead.
    if cs.cache.shape[1] >= max(target, cs.n + 1):
        return
    new_cap = max(target, cs.n + CompressorState.GROWTH)
    grown = mx.zeros((1, new_cap, comp.head_dim), dtype=cs.cache.dtype)
    grown[:, : cs.n] = cs.cache[:, : cs.n]
    cs.cache = grown


def _pack_decode_state(c: DeepSeekV4Cache, layer) -> tuple:
    attn = layer.attn
    if c.ring is None:  # decode at offset>0 without prefill: restored state
        c.ring = mx.zeros((1, attn.window, attn.head_dim), dtype=mx.bfloat16)
    arrays = [c.ring]
    if c.comp is not None:
        _ensure_comp_capacity(c.comp, attn.compressor, c.ring.dtype)
        arrays += [c.comp.kv_state, c.comp.score_state, c.comp.cache,
                   _scalar_i32(c.comp.n)]
    if c.idx is not None:
        _ensure_comp_capacity(c.idx, attn.indexer.compressor, c.ring.dtype)
        arrays += [c.idx.kv_state, c.idx.score_state, c.idx.cache,
                   _scalar_i32(c.idx.n)]
    return tuple(arrays)


def _unpack_decode_state(
    c: DeepSeekV4Cache, outs: tuple, start: int, ratio: int
) -> None:
    """Commit a layer's functionally-updated state. The python-side group
    counts advance WITHOUT reading the traced n back (that would force a
    device sync per layer): completion is derivable from the python offset —
    a group closes exactly when (start+1) % ratio == 0. The indexer's
    compressor shares the schedule (indexer layers are ratio 4)."""
    c.ring = outs[0]
    completed = 1 if (start + 1) % ratio == 0 else 0
    i = 1
    for cs in (c.comp, c.idx):
        if cs is None:
            continue
        cs.kv_state, cs.score_state, cs.cache = outs[i], outs[i + 1], outs[i + 2]
        cs.n += completed
        i += 4  # outs[i+3] is the traced n — intentionally unread


def _decode_window_indices_scalar(win: int, offset: mx.array) -> mx.array:
    """[1, 1, win] indices into concat([ring, kv1]) for one query at
    ``offset`` — the scalar-offset form of continuation_window_indices."""
    qpos = offset - win + 1 + mx.arange(win)
    slot = mx.maximum(qpos, 0) % win
    idx = mx.where(qpos >= offset, win + (qpos - offset), slot)
    return mx.where(qpos < 0, MASKED_INDEX, idx).astype(mx.int32)[None, None]


def _sparse_attend_eager(
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
            # Cold prefill: window indices point into the chunk itself.
            win_idx = window_topk_indices(win, s, 0)
            parts = [(kv, mx.broadcast_to(win_idx[None], (b, s, win_idx.shape[1])))]
        else:
            # Continuation (chunked prefill or decode): attend over the
            # virtual buffer [ring | chunk]. The ring still holds the state
            # from BEFORE this chunk — it is written below, after the
            # indices are established (rows never reference their own
            # in-chunk positions through the ring half).
            buffer = mx.concatenate([cache.ring, kv], axis=1)
            win_idx = continuation_window_indices(win, s, start)
            parts = [(buffer, mx.broadcast_to(win_idx[None], (b, *win_idx.shape)))]
        if self.ratio:
            if "indexer" in self:
                cidx = self.indexer(x, qr, cache.idx, self._freqs, start)
            else:
                cidx = None
            self.compressor(x, cache.comp, self._freqs, start)
            comp = cache.comp.valid()
            if comp is not None:
                if cidx is None:
                    m = compress_topk_indices(self.ratio, s, start, 0)
                    cidx = mx.broadcast_to(m[None], (b, *m.shape))
                if cidx.shape[-1] > 0:
                    parts.append((comp, cidx))

        if s >= win:
            slots = mx.arange(start + s - win, start + s) % win
            cache.ring[:, slots] = kv[:, -win:]
        else:
            slots = mx.arange(start, start + s) % win
            cache.ring[:, slots] = kv
        cache.offset = start + s

        o = sparse_attend(q, parts, self.attn_sink, self.scale)
        o_tail = apply_interleaved_rope(o[..., -rd:], cos[:, None], sin[:, None], inverse=True)
        o = mx.concatenate([o[..., :-rd], o_tail], axis=-1)

        o = o.reshape(b, s, self.n_groups, -1)
        groups = mx.broadcast_to(mx.arange(self.n_groups)[None, None], (b, s, self.n_groups))
        o = self.wo_a(o[..., None, :], groups).squeeze(-2)
        return self.wo_b(o.reshape(b, s, -1))

    def _x_stack(self):
        """The lazily-built stacked x-projection: every per-layer projection OF
        X (wq_a, wkv, the compressors' wkv/wgate, the indexer's weights_proj)
        concatenated along the output axis. Affine-quantized rows are
        independent along that axis, so this is a pure concat — no
        reconversion, identical numerics. Shared by the compiled decode path
        (_x_projections) and the native replay plan (one qmv instead of up to
        7 dispatches per layer)."""
        st = getattr(self, "_xstack", None)
        if st is not None:
            return st
        mods = [self.wq_a, self.wkv]
        if self.ratio:
            mods += [self.compressor.wkv, self.compressor.wgate]
            if "indexer" in self:
                mods += [
                    self.indexer.weights_proj,
                    self.indexer.compressor.wkv,
                    self.indexer.compressor.wgate,
                ]
        if isinstance(mods[0], nn.QuantizedLinear):
            gs, bits = mods[0].group_size, mods[0].bits
            assert all(m.group_size == gs and m.bits == bits for m in mods)
            st = (
                "q",
                mx.concatenate([m.weight for m in mods], axis=0),
                mx.concatenate([m.scales for m in mods], axis=0),
                mx.concatenate([m.biases for m in mods], axis=0),
                gs,
                bits,
                [m.scales.shape[0] for m in mods],
            )
        else:
            st = ("p", mx.concatenate([m.weight for m in mods], axis=0),
                  [m.weight.shape[0] for m in mods])
        # materialize now: the native replay registers these arrays in its
        # buffer table via DLPack, which needs real buffers, and identity must
        # stay stable across plan rebuilds (hence cached on the module).
        mx.eval(*st[1:4] if st[0] == "q" else (st[1],))
        self._xstack = st
        return st

    def _x_projections(self, x):
        """All per-layer projections OF X as ONE matmul (decode path); see
        _x_stack for the stacking contract."""
        st = self._x_stack()
        if st[0] == "q":
            _, qw, sc, bs, gs, bits, sizes = st
            out = mx.quantized_matmul(
                x, qw, sc, bs, transpose=True, group_size=gs, bits=bits
            )
        else:
            _, w, sizes = st
            out = x @ w.T
        splits = []
        acc = 0
        for sz in sizes[:-1]:
            acc += sz
            splits.append(acc)
        return mx.split(out, splits, axis=-1)

    def _attn_core_call(self, q_raw, kv_normed, ring, comp, cidx, plain, ncomp, offset):
        """Invoke the fused v2 attention-core kernel. All inputs traced; the
        outputs come back already de-rotated (o) and roped (kv row)."""
        h, d, rd, win = self.n_heads, self.head_dim, self.rope_dim, self.window
        if comp is None:
            comp, cidx = _kernel_dummy_part(d, str(q_raw.dtype).split(".")[-1])
            kc = 0
        elif plain:
            kc = comp.shape[1]
            # PLAIN mode generates indices in-kernel; cidx is an unread dummy
            cidx = _kernel_dummy_part(d, str(q_raw.dtype).split(".")[-1])[1]
        else:
            kc = cidx.shape[-1]
        out, kv_out = _get_attn_core_kernel()(
            inputs=[
                q_raw.reshape(-1),
                kv_normed.reshape(-1),
                ring.reshape(-1),
                comp.reshape(-1, d).reshape(-1),
                cidx.reshape(-1),
                self.attn_sink,
                self._freqs,
                _attn_core_params(d, rd, win, kc, 1 if plain else 0),
                _attn_core_fscal(self.scale, self.eps),
                mx.stack([offset.astype(mx.int32), ncomp.astype(mx.int32)]),
            ],
            template=[("T", q_raw.dtype)],
            grid=(h * 512, 1, 1),
            threadgroup=(512, 1, 1),
            output_shapes=[(h * d,), (d,)],
            output_dtypes=[q_raw.dtype, ring.dtype],
        )
        return out, kv_out

    def _attn_core_usable(self, kc: int, q_dtype, ring_dtype) -> bool:
        return (
            os.environ.get("SOLOHEAVEN_METAL_KERNELS", "1") != "0"
            and mx.metal.is_available()
            and self.head_dim <= 512
            and self.window + kc <= 2176
            and q_dtype == ring_dtype  # single template T covers both outputs
        )

    def decode_step_math(self, x, ring, state_arrays, offset):
        """Branchless single-token attention for the compiled decode path.

        ``state_arrays``: () for dense layers, (ckv, csc, cbuf, cn) for
        plain-compressed, plus (ikv, isc, ibuf, in_) for indexer layers.
        Returns (out, new_ring, *new_state_arrays). Same math as __call__'s
        continuation branch, with offset as a traced scalar. The glue
        (ropes, q-RMS, window indices, de-rotation) lives inside the v2
        kernel when usable; the expanded math below is the reference path.
        """
        xp = self._x_projections(x)
        qr = self.q_norm(xp[0])
        q_raw = self.wq_b(qr)
        kvn = self.kv_norm(xp[1])

        new_state: tuple = ()
        comp = cidx = None
        plain = False
        ncomp = offset  # dummy scalar when unused
        if self.ratio:
            if "indexer" in self:
                ckv, csc, cbuf, cn, ikv, isc, ibuf, in_ = state_arrays
                cidx, ikv, isc, ibuf, in_ = self.indexer.decode_step_math(
                    (xp[5], xp[6]), xp[4], x.dtype, qr,
                    ikv, isc, ibuf, in_, offset, self._freqs,
                )
            else:
                ckv, csc, cbuf, cn = state_arrays
            ckv, csc, cbuf, cn = self.compressor.decode_step_math(
                (xp[2], xp[3]), x.dtype, ckv, csc, cbuf, cn, offset, self._freqs
            )
            comp = cbuf
            plain = "indexer" not in self
            ncomp = cn
            if plain:
                new_state = (ckv, csc, cbuf, cn)
            else:
                new_state = (ckv, csc, cbuf, cn, ikv, isc, ibuf, in_)

        kc = 0 if comp is None else (comp.shape[1] if plain else cidx.shape[-1])
        if self._attn_core_usable(kc, q_raw.dtype, ring.dtype):
            b = 1
            o_flat, kv_roped = self._attn_core_call(
                q_raw, kvn, ring, comp, cidx, plain, ncomp, offset
            )
            new_ring = ring
            new_ring[:, offset % self.window] = kv_roped[None]
            o = o_flat.reshape(b, 1, self.n_groups, -1)
            groups = mx.broadcast_to(
                mx.arange(self.n_groups)[None, None], (b, 1, self.n_groups)
            )
            o = self.wo_a(o[..., None, :], groups).squeeze(-2)
            return (self.wo_b(o.reshape(b, 1, -1)), new_ring, *new_state)
        return self._decode_attn_reference(
            x, q_raw, kvn, ring, comp, cidx, plain, ncomp, offset, new_state
        )

    def _decode_attn_reference(
        self, x, q_raw, kvn, ring, comp, cidx, plain, ncomp, offset, new_state
    ):
        """The expanded decode attention math — canonical reference for the
        v2 kernel (cross-checked by the decode-consistency suite)."""
        rd, win = self.rope_dim, self.window
        pos = offset.reshape(1)
        cos, sin = rope_cos_sin(self._freqs, pos)

        q = q_raw.reshape(1, 1, self.n_heads, self.head_dim)
        qf = q.astype(mx.float32)
        q = (qf * mx.rsqrt(qf.square().mean(-1, keepdims=True) + self.eps)).astype(q.dtype)
        q_tail = apply_interleaved_rope(q[..., -rd:], cos[:, None], sin[:, None])
        q = mx.concatenate([q[..., :-rd], q_tail], axis=-1)

        kv_tail = apply_interleaved_rope(kvn[..., -rd:], cos, sin)
        kv = mx.concatenate([kvn[..., :-rd], kv_tail], axis=-1)

        buffer = mx.concatenate([ring, kv], axis=1)
        parts = [(buffer, _decode_window_indices_scalar(win, offset))]
        if comp is not None:
            if plain:
                cap = comp.shape[1]
                base = mx.arange(cap, dtype=mx.int32)[None, None]
                cidx = mx.where(base >= ncomp, MASKED_INDEX, base)
            parts.append((comp, cidx))

        new_ring = ring
        new_ring[:, offset % win] = kv[:, 0]

        o = sparse_attend(q, parts, self.attn_sink, self.scale)
        o_tail = apply_interleaved_rope(
            o[..., -rd:], cos[:, None], sin[:, None], inverse=True
        )
        o = mx.concatenate([o[..., :-rd], o_tail], axis=-1)
        o = o.reshape(1, 1, self.n_groups, -1)
        groups = mx.broadcast_to(
            mx.arange(self.n_groups)[None, None], (1, 1, self.n_groups)
        )
        o = self.wo_a(o[..., None, :], groups).squeeze(-2)
        return (self.wo_b(o.reshape(1, 1, -1)), new_ring, *new_state)


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


# --- gate + rms_norm kernels (REPLAY path) ----------------------------------
#
# dsv4_gate: scores = sqrtsoftplus(x @ weight^T) over n_routed experts, then
# noaux_tc top-k on (scores + bias) with weights gathered from the UNBIASED
# scores, normalized and route-scaled. Score layers only (hash layers index by
# tid2eid — a trivial lookup done Python-side). One threadgroup: each simdgroup
# scores a strip of experts, then thread 0 runs the tiny top-k selection.
# dsv4_rms: plain RMSNorm with a weight, [d] -> [d].

_GATE_SRC = """
    uint tid = thread_position_in_threadgroup.x;
    const int TG = 256;
    const int n_exp = params[0];
    const int dim = params[1];
    const int topk = params[2];
    const float route_scale = feps[0];

    threadgroup float xs[4096];
    for (int i = tid; i < dim; i += TG) xs[i] = float(x[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // each simdgroup handles experts e = sg, sg+8, ...
    uint sg = tid / 32, lane = tid % 32;
    for (int e = sg; e < n_exp; e += TG / 32) {
        float a = 0.0f;
        for (int i = lane; i < dim; i += 32) a += float(weight[e * dim + i]) * xs[i];
        a = simd_sum(a);
        if (lane == 0) {
            float sp = log(1.0f + exp(-fabs(a))) + max(a, 0.0f);  // softplus
            scores[e] = sqrt(sp);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        // top-k over (scores + bias); weights from UNBIASED scores.
        float wsum = 0.0f;
        bool taken[512];
        for (int e = 0; e < n_exp; ++e) taken[e] = false;
        for (int k = 0; k < topk; ++k) {
            int best = -1;
            float bestv = -INFINITY;
            for (int e = 0; e < n_exp; ++e) {
                if (taken[e]) continue;
                float v = scores[e] + bias[e];
                if (v > bestv) { bestv = v; best = e; }
            }
            taken[best] = true;
            out_idx[k] = best;
            out_w[k] = scores[best];
            wsum += scores[best];
        }
        for (int k = 0; k < topk; ++k) out_w[k] = out_w[k] / wsum * route_scale;
    }
"""

# Split gate for the native path: the single-threadgroup _GATE_SRC scores all
# n_exp experts with ONE threadgroup, which cannot hide the weight-fetch latency
# and cost ~1.75 ms/layer on the real model (256 experts x 4096). dsv4_gate_score_k
# scores ONE expert per threadgroup (grid = n_exp) so the whole chip hides latency;
# dsv4_gate_topk_k then does the tiny noaux_tc top-k over the scores. Same math as
# _GATE_SRC, just parallelized.
_GATE_SCORE_SRC = """
    uint e = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    const int TG = 256;
    const int n_exp = params[0];
    const int dim = params[1];
    if ((int)e >= n_exp) return;
    threadgroup float red[8];
    float a = 0.0f;
    for (int i = tid; i < dim; i += TG) a += float(weight[e * dim + i]) * float(x[i]);
    a = simd_sum(a);
    if ((tid & 31u) == 0) red[tid / 32] = a;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float t = 0.0f;
        for (int i = 0; i < TG / 32; ++i) t += red[i];
        float sp = log(1.0f + exp(-fabs(t))) + max(t, 0.0f);
        scores[e] = sqrt(sp);
    }
"""

_GATE_TOPK_SRC = """
    uint tid = thread_position_in_threadgroup.x;
    const int TG = 256;
    const int n_exp = params[0];
    const int topk = params[2];
    const float route_scale = feps[0];

    // THREADGROUP-PARALLEL top-k: topk rounds of a cooperative argmax over the
    // biased scores. A single-thread selection loop (even over staged
    // threadgroup memory) cost ~0.27 ms/dispatch — one GPU thread running
    // topk*n_exp serial iterations is the bottleneck itself, not the memory.
    // Ties pick the lowest expert index (strict > in the reduction), matching
    // the serial scan's order.
    threadgroup float sc[512];      // unbiased scores (weights come from these)
    threadgroup float sb[512];      // biased scores; -inf once taken
    threadgroup float bv[256];
    threadgroup int bi[256];
    threadgroup float wsel[64];
    threadgroup int isel[64];
    for (int e = tid; e < n_exp; e += TG) {
        float s = scores[e];
        sc[e] = s;
        sb[e] = s + bias[e];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int k = 0; k < topk; ++k) {
        float mv = -INFINITY;
        int mi = -1;
        for (int e = tid; e < n_exp; e += TG) {
            if (sb[e] > mv) { mv = sb[e]; mi = e; }
        }
        bv[tid] = mv;
        bi[tid] = mi;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int s = TG / 2; s > 0; s >>= 1) {
            if ((int)tid < s && bv[tid + s] > bv[tid]) {
                bv[tid] = bv[tid + s];
                bi[tid] = bi[tid + s];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0) {
            isel[k] = bi[0];
            wsel[k] = sc[bi[0]];
            sb[bi[0]] = -INFINITY;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        float wsum = 0.0f;
        for (int k = 0; k < topk; ++k) wsum += wsel[k];
        for (int k = 0; k < topk; ++k) {
            out_idx[k] = isel[k];
            out_w[k] = wsel[k] / wsum * route_scale;
        }
    }
"""

# Hash-routing gate (the first `num_hash_layers` layers): the topk experts come
# straight from tid2eid[token] (no top-k search, no bias), and the weights are
# the UNBIASED sqrtsoftplus scores at exactly those experts, normalized and
# route-scaled. Mirrors Gate.__call__'s hash branch. Token id arrives in ioff[0].
_GATE_HASH_SRC = """
    uint tid = thread_position_in_threadgroup.x;
    const int TG = 256;
    const int dim = params[1];
    const int topk = params[2];
    const float route_scale = feps[0];
    const int token = ioff[0];

    threadgroup float xs[4096];
    for (int i = tid; i < dim; i += TG) xs[i] = float(x[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // one simdgroup per selected expert: score = sqrtsoftplus(x . weight[e])
    threadgroup float sc[64];
    uint sg = tid / 32, lane = tid % 32;
    for (int k = sg; k < topk; k += TG / 32) {
        int e = tid2eid[token * topk + k];
        float a = 0.0f;
        for (int i = lane; i < dim; i += 32) a += float(weight[e * dim + i]) * xs[i];
        a = simd_sum(a);
        if (lane == 0) {
            float sp = log(1.0f + exp(-fabs(a))) + max(a, 0.0f);
            sc[k] = sqrt(sp);
            out_idx[k] = e;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float wsum = 0.0f;
        for (int k = 0; k < topk; ++k) wsum += sc[k];
        for (int k = 0; k < topk; ++k) out_w[k] = sc[k] / wsum * route_scale;
    }
"""

_RMS_SRC = """
    uint tid = thread_position_in_threadgroup.x;
    const int TG = 256;
    const int d = params[0];
    const float eps = feps[0];
    threadgroup float red[8];
    threadgroup float rn[1];
    float acc = 0.0f;
    for (int i = tid; i < d; i += TG) { float v = float(x[i]); acc += v * v; }
    acc = simd_sum(acc);
    if ((tid & 31u) == 0) red[tid / 32] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float t = 0.0f;
        for (int i = 0; i < TG / 32; ++i) t += red[i];
        rn[0] = rsqrt(t / d + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float r = rn[0];
    for (int i = tid; i < d; i += TG) y[i] = T(float(x[i]) * r * float(w[i]));
"""

# DSA indexer scoring: score[g] = sum_h relu(q_roped[h] . buf[g]) * w[h] over
# n_idx_heads, then top-k over the visible groups. Split into a per-group score
# kernel (grid = cap threadgroups, buf[g] staged; q read from device and roped
# inline) and a tiny top-k kernel, so neither needs to stage the full q
# (n_idx_heads*idx_head_dim can be 32 KB). Follows Indexer.decode_step_math.
_RMS2_SRC = """
    // TWO independent RMS norms in one dispatch (threadgroup 0 -> a, 1 -> b).
    // q_norm and kv_norm both read slices of the stacked x-projection and feed
    // different consumers, so they have no ordering between them; pairing them
    // halves the 4-per-layer rms dispatches. Native-only (the compiled path
    // uses mx.fast.rms_norm), so no twin signature to keep in step.
    uint tid = thread_position_in_threadgroup.x;
    uint which = threadgroup_position_in_grid.x;
    const int TG = 256;
    const int d = params[which == 0 ? 0 : 1];
    const float eps = feps[0];
    threadgroup float red[8];
    threadgroup float rn[1];
    const device bfloat* x = (which == 0) ? xa : xb;
    const device bfloat* w = (which == 0) ? wa : wb;
    device bfloat* y = (which == 0) ? ya : yb;
    float acc = 0.0f;
    for (int i = tid; i < d; i += TG) { float v = float(x[i]); acc += v * v; }
    acc = simd_sum(acc);
    if ((tid & 31u) == 0) red[tid / 32] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float t = 0.0f;
        for (int i = 0; i < TG / 32; ++i) t += red[i];
        rn[0] = rsqrt(t / d + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float r = rn[0];
    for (int i = tid; i < d; i += TG) y[i] = T(float(x[i]) * r * float(w[i]));
"""

_IDX_SCORE_SRC = """
    uint g = threadgroup_position_in_grid.x;   // group index
    uint tid = thread_position_in_threadgroup.x;
    uint sg = tid / 32, lane = tid % 32;
    const int TG = 256;
    const int n_h = params[0];
    const int hd = params[1];       // idx_head_dim
    const int rd = params[2];       // rope_dim
    const int cap = params[3];
    const int n2 = ioff[1];
    const int offset = ioff[0];
    const float wscale = fscal[0];
    if ((int)g >= cap) return;

    threadgroup float bg[128];      // buf[g]
    threadgroup float cs[64];       // cos/sin for the rope tail
    threadgroup float red[8];
    for (int i = tid; i < hd; i += TG) bg[i] = float(buf[g * hd + i]);
    for (int p = tid; p < rd / 2; p += TG) {
        float ang = float(offset) * float(freqs[p]);
        cs[2 * p] = cos(ang); cs[2 * p + 1] = sin(ang);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if ((int)g >= n2) { if (tid == 0) scores[g] = -INFINITY; return; }

    float acc = 0.0f;
    for (int h = sg; h < n_h; h += TG / 32) {
        float d = 0.0f;
        for (int i = lane; i < hd; i += 32) {
            float qv = float(q[h * hd + i]);
            if (i >= hd - rd) {                 // rope the tail pairwise
                int pr = (i - (hd - rd)) / 2;
                bool hi = ((i - (hd - rd)) & 1) == 1;
                int i0 = hd - rd + 2 * pr;
                float e = float(q[h * hd + i0]), o = float(q[h * hd + i0 + 1]);
                float c = cs[2 * pr], s = cs[2 * pr + 1];
                qv = hi ? (e * s + o * c) : (e * c - o * s);
            }
            d += qv * bg[i];
        }
        d = simd_sum(d);
        if (lane == 0) acc += max(d, 0.0f) * float(w[h]) * wscale;
    }
    if (lane == 0) red[sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float t = 0.0f;
        for (int i = 0; i < TG / 32; ++i) t += red[i];
        scores[g] = t;
    }
"""

_IDX_TOPK_SRC = """
    // top-k over scores[cap]; scores at indices >= n2 are already -inf. Emit the
    // selected group indices (order is irrelevant — attn_core softmaxes over the
    // whole set), -1 for slots past the valid count (matching the reference mask).
    uint tid = thread_position_in_threadgroup.x;
    const int cap = params[0];   // scratch capacity; the scan is bounded by n2
    (void)cap;
    const int topk = params[1];
    const int n2 = ioff[1];
    // Common case (context shorter than index_topk groups): every valid group is
    // selected, so emit 0..n2-1 directly across all threads. This skips the
    // O(cap*topk) single-thread selection sort that otherwise dominates decode
    // (~19 ms/layer at cap=256, topk=512 — see docs/benchmarks/deepseek-v4.md).
    if (n2 <= topk) {
        for (int k = tid; k < topk; k += 256) out_idx[k] = (k < n2) ? k : -1;
        return;
    }
    // n2 > topk (context past index_topk*ratio tokens): a real selection.
    //
    // This used to be a single-thread selection sort over the whole CAPACITY:
    // O(cap*topk) = 8192*512 = 4.2M serial iterations per layer per token, and
    // `bool taken[cap]` overran its 1024-entry declaration once cap grew past
    // that. It cost 335 of the 380 ms/token this path spent at a 2.5k context
    // while every other kernel stayed flat (Stage 4m). Now: a threshold search
    // on an order-preserving integer key, which the whole threadgroup shares.
    threadgroup int red[8];              // TG/32 simdgroup partials
    threadgroup atomic_int cnt;

    // Largest key with count(key >= lo) >= topk — i.e. the key of the topk-th
    // largest score. sh_order_key is monotonic in the float, so 32 halvings of
    // the uint range land on it EXACTLY: no tolerance, no iteration budget.
    uint lo = 0u, hi = 0xFFFFFFFFu;
    while (lo < hi) {
        uint span = hi - lo;             // (hi-lo+1)/2 would overflow at full range
        uint mid = lo + (span >> 1) + (span & 1u);
        int c = 0;
        for (int i = tid; i < n2; i += 256)
            if (sh_order_key(scores[i]) >= mid) ++c;
        c = simd_sum(c);
        if ((tid & 31u) == 0) red[tid / 32] = c;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        int tot = 0;
        for (int j = 0; j < 8; ++j) tot += red[j];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // lo/hi are identical in every thread, so the loop stays uniform and
        // the barriers above are reached by all of them.
        if (tot >= topk) lo = mid; else hi = mid - 1u;
    }

    if (tid == 0) atomic_store_explicit(&cnt, 0, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Strictly-above-threshold groups: fewer than topk of them by construction
    // (count(>= lo+1) < topk), and their order within the set does not matter —
    // attn_core softmaxes over the whole selection.
    for (int i = tid; i < n2; i += 256)
        if (sh_order_key(scores[i]) > lo) {
            int slot = atomic_fetch_add_explicit(&cnt, 1, memory_order_relaxed);
            if (slot < topk) out_idx[slot] = i;
        }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        // Top up from the tie group AT the threshold, lowest index first, so
        // the selected SET is deterministic even when scores collide.
        int c = atomic_load_explicit(&cnt, memory_order_relaxed);
        for (int i = 0; i < n2 && c < topk; ++i)
            if (sh_order_key(scores[i]) == lo) out_idx[c++] = i;
        // out_idx[0] stays the argmax (the indexer test asserts it).
        int bp = 0;
        for (int k = 1; k < topk; ++k) {
            int a = out_idx[k], b = out_idx[bp];
            if (scores[a] > scores[b] || (scores[a] == scores[b] && a < b)) bp = k;
        }
        if (bp) { int t = out_idx[0]; out_idx[0] = out_idx[bp]; out_idx[bp] = t; }
    }
"""

_idx_kernels = None


def _get_idx_kernels():
    global _idx_kernels
    if _idx_kernels is None:
        score = mx.fast.metal_kernel(
            name="sh_dsv4_idx_score_k",
            input_names=["q", "buf", "w", "freqs", "params", "fscal", "ioff"],
            output_names=["scores"],
            source=_IDX_SCORE_SRC,
        )
        topk = mx.fast.metal_kernel(
            name="sh_dsv4_idx_topk_k",
            input_names=["scores", "params", "ioff"],
            output_names=["out_idx"],
            source=_IDX_TOPK_SRC,
            header=_STATIC_DEFINES,
        )
        _idx_kernels = (score, topk)
    return _idx_kernels


_EMBED_SRC = """
    // dequantize embed row token_id (8-bit gs64) and replicate into hc streams:
    // h[s*hidden + i] = dequant(weight[token][i]) for every stream s.
    uint tid = thread_position_in_grid.x;
    const int hidden = params[0];
    const int hc = params[1];
    const int token = ioff[0];
    if ((int)tid >= hidden) return;
    // QD_VPW values per uint32, so hidden/QD_VPW packed words per row.
    uint word = weight[(uint)token * (hidden / QD_VPW) + tid / QD_VPW];
    float sc = float(scales[(uint)token * (hidden / QD_GS) + tid / QD_GS]);
    float bi = float(biases[(uint)token * (hidden / QD_GS) + tid / QD_GS]);
    float v = float((word >> (QD_BITS * (tid % QD_VPW))) & QD_MASK) * sc + bi;
    for (int s = 0; s < hc; ++s) h[s * hidden + tid] = T(v);
"""

_HC_HEAD_SRC = """
    // x[i] = sum_hc pre[hc] * flat[hc, i], pre = sigmoid(mixes*scale+base)+eps
    // mixes[r] = (fn[r] . flat) * rms(flat), fn is [hc, hc*hidden].
    uint tid = thread_position_in_threadgroup.x;
    const int TG = 256;
    const int hc = params[0];
    const int d = params[1];
    const int hcd = hc * d;
    const float eps = feps[0];
    const float hc_eps = feps[1];
    threadgroup float red[8];
    threadgroup float mixes[8];
    threadgroup float rms_s[1];

    float acc = 0.0f;
    for (int i = tid; i < hcd; i += TG) { float v = float(h[i]); acc += v * v; }
    acc = simd_sum(acc);
    if ((tid & 31u) == 0) red[tid / 32] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) { float t = 0.0f; for (int i = 0; i < TG/32; ++i) t += red[i]; rms_s[0] = rsqrt(t / hcd + eps); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rms = rms_s[0];
    for (int r = tid / 32; r < hc; r += TG / 32) {
        float a = 0.0f;
        for (int i = tid % 32; i < hcd; i += 32) a += float(fn[r * hcd + i]) * float(h[i]);
        a = simd_sum(a);
        if ((tid & 31u) == 0) mixes[r] = a * rms;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0)
        for (int r = 0; r < hc; ++r)
            mixes[r] = 1.0f / (1.0f + exp(-(mixes[r] * scale[0] + base[r]))) + hc_eps;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = tid; i < d; i += TG) {
        float a = 0.0f;
        for (int r = 0; r < hc; ++r) a += mixes[r] * float(h[r * d + i]);
        y[i] = T(a);
    }
"""

_SWIGLU_SRC = """
    // clipped SwiGLU elementwise: out[i] = silu(min(g,lim)) * clamp(u,-lim,lim)
    uint tid = thread_position_in_grid.x;
    const int n = params[0];
    const float limit = feps[0];
    if ((int)tid >= n) return;
    float g = float(gate[tid]);
    float u = float(up[tid]);
    if (limit > 0.0f) { u = clamp(u, -limit, limit); g = min(g, limit); }
    out[tid] = T((g / (1.0f + exp(-g))) * u);
"""

_ADD_SRC = """
    // out[i] = a[i] + b[i], written in the compute dtype T
    uint tid = thread_position_in_grid.x;
    const int n = params[0];
    if ((int)tid >= n) return;
    out[tid] = T(float(a[tid]) + float(b[tid]));
"""

_WO_A_SRC = """
    // Grouped 8-bit affine qmv for the o_groups low-rank O projection:
    //   out[gi*o_lora + j] = sum_i deq(w[gi, j, i]) * x[gi*gin + i]
    // replacing the o_groups (8) separate library qmv dispatches with ONE.
    // One simdgroup per output row (gi, j); weight is uint32-packed 8-bit gs64
    // (4 values/word, one scale+bias per 64 = 16 words), affine w = q*sc + bi.
    // Pairing two rows per simdgroup (as done in moe_w2, where it is worth
    // -1.6 ms) measured 3.45 -> 3.73 ms HERE — see Stage 4k: this kernel's
    // activation is only 8 KB per group and already L1-resident, so the
    // halved loads bought nothing and the doubled live weight stream cost
    // occupancy. Do not re-pair it without a new measurement.
    uint sg_id = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint row = threadgroup_position_in_grid.x * 8 + sg_id;
    const int g = params[0];
    const int gin = params[1];
    const int o_lora = params[2];
    uint gi = row / o_lora;
    if (gi >= (uint)g) return;
    uint j = row % o_lora;
    const int words = gin / QD_VPW;
    const uint wbase = ((uint)gi * o_lora + j) * words;
    const uint sbase = ((uint)gi * o_lora + j) * (gin / QD_GS);
    const uint xbase = gi * gin;
    float a = 0.0f;
    for (int w = lane; w < words; w += 32) {
        uint p = weight[wbase + w];
        float sc = float(scales[sbase + w / QD_WPG]);
        float bi = float(biases[sbase + w / QD_WPG]);
        float aw = 0.0f, sw = 0.0f;
        for (int k = 0; k < QD_VPW; ++k) {
            float xk = float(x[xbase + w * QD_VPW + k]);
            aw += float((p >> (QD_BITS * k)) & QD_MASK) * xk;
            sw += xk;
        }
        a += aw * sc + sw * bi;
    }
    a = simd_sum(a);
    if (lane == 0) out[row] = T(a);
"""

_SH13_SRC = """
    // Fused shared-expert w1/w3 + clipped SwiGLU: one simdgroup per inter
    // row j computes both 8-bit gs64 affine dots against x and writes
    // silu(min(g,lim)) * clamp(u,-lim,lim) — replacing two library qmv
    // dispatches and the elementwise swiglu per MoE layer.
    uint sg_id = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint row = threadgroup_position_in_grid.x * 8 + sg_id;
    const int hidden = params[0];
    const int inter = params[1];
    const float limit = feps[0];
    if (row >= (uint)inter) return;
    const int words = hidden / QD_VPW;
    const uint wbase = row * (uint)words;
    const uint sbase = row * (uint)(hidden / QD_GS);
    float a1 = 0.0f, a3 = 0.0f;
    for (int w = lane; w < words; w += 32) {
        uint p1 = w1[wbase + w];
        uint p3 = w3[wbase + w];
        float sc1 = float(s1[sbase + w / QD_WPG]);
        float bi1 = float(b1[sbase + w / QD_WPG]);
        float sc3 = float(s3[sbase + w / QD_WPG]);
        float bi3 = float(b3[sbase + w / QD_WPG]);
        float aw1 = 0.0f, aw3 = 0.0f, sx = 0.0f;
        #pragma unroll
        for (int k = 0; k < QD_VPW; ++k) {
            float xk = float(x[w * QD_VPW + k]);
            aw1 += float((p1 >> (QD_BITS * k)) & QD_MASK) * xk;
            aw3 += float((p3 >> (QD_BITS * k)) & QD_MASK) * xk;
            sx += xk;
        }
        a1 += aw1 * sc1 + sx * bi1;
        a3 += aw3 * sc3 + sx * bi3;
    }
    a1 = simd_sum(a1);
    a3 = simd_sum(a3);
    if (lane == 0) {
        float g = a1, u = a3;
        if (limit > 0.0f) { u = clamp(u, -limit, limit); g = min(g, limit); }
        out[row] = T((g / (1.0f + exp(-g))) * u);
    }
"""

_RING_STORE_SRC = """
    // ring[(offset % win) * D + i] = src[i]  — the post-attention KV write.
    uint tid = thread_position_in_threadgroup.x;
    const int TG = 256;
    const int D = params[0];
    const int win = params[1];
    const int slot = ioff[0] % win;
    for (int i = tid; i < D; i += TG) ring[slot * D + i] = src[i];
"""

@functools.lru_cache(maxsize=8)
def _get_misc_kernels(dense_bits: int, dense_gs: int):
    """Small shared kernels; `embed` unpacks DENSE weights, so this is
    compiled per dense recipe like the moe pair."""
    store = mx.fast.metal_kernel(
        name="sh_dsv4_ring_store_k",
        input_names=["src", "params", "ioff"],
        output_names=["ring"],
        source=_RING_STORE_SRC,
    )
    swiglu = mx.fast.metal_kernel(
        name="sh_dsv4_swiglu_k",
        input_names=["gate", "up", "params", "feps"],
        output_names=["out"],
        source=_SWIGLU_SRC,
    )
    add = mx.fast.metal_kernel(
        name="sh_dsv4_add_k",
        input_names=["a", "b", "params"],
        output_names=["out"],
        source=_ADD_SRC,
    )
    embed = mx.fast.metal_kernel(
        name="sh_dsv4_embed_k",
        input_names=["weight", "scales", "biases", "params", "ioff"],
        output_names=["h"],
        source=_EMBED_SRC,
        header=_STATIC_DEFINES + pack_defines("QD", dense_bits, dense_gs),
    )
    hc_head = mx.fast.metal_kernel(
        name="sh_dsv4_hc_head_k",
        input_names=["h", "fn", "scale", "base", "params", "feps"],
        output_names=["y"],
        source=_HC_HEAD_SRC,
    )
    return (store, swiglu, add, embed, hc_head)


_gate_kernels = None


def _get_gate_kernels():
    global _gate_kernels
    if _gate_kernels is None:
        gate = mx.fast.metal_kernel(
            name="sh_dsv4_gate_k",
            input_names=["x", "weight", "bias", "params", "feps"],
            output_names=["scores", "out_idx", "out_w"],
            source=_GATE_SRC,
        )
        rms = mx.fast.metal_kernel(
            name="sh_dsv4_rms_k",
            input_names=["x", "w", "params", "feps"],
            output_names=["y"],
            source=_RMS_SRC,
        )
        _gate_kernels = (gate, rms)
    return _gate_kernels


# --- HC pre/post kernels (REPLAY path only) ---------------------------------
#
# hc_pre = rms(flat) -> mixes = (flat @ fn^T) * rms -> sinkhorn split ->
#          y = sum_hc pre[hc] * flat[hc]. hc_post = post*x + comb@residual.
# A single-dispatch kernel each. These are NOT wired into the mx.compile decode
# path — there, one threadgroup doing the [mix, hc*d] GEMV starved the chip and
# regressed decode (docs/benchmarks/deepseek-v4.md). In the external replay
# loop there is no per-op floor to amortize, so the single dispatch is the
# right shape. Exposed for the native runtime and diff-tested against
# _hc_pre_math / _hc_post_math.

_HC_MIX_SRC = """
    // One threadgroup per mix row: the raw dot fn[r] . h, reduced over the whole
    // threadgroup. This is the HC input mixing GEMV — ~1.5 MB of fn read that
    // starved a single threadgroup in the fused hc_pre (it was 50%+ of decode);
    // splitting it out to (mix) threadgroups lets the chip hide the memory
    // latency. rms and everything downstream stay in dsv4_hc_pre_k, which just
    // reads these raw dots.
    uint tid = thread_position_in_threadgroup.x;
    const int TG = 1024;  // 24 threadgroups underfill the chip; shorten each thread's chain instead
    const int hcn = params[0];
    const int d = params[1];
    const int hcd = hcn * d;
    uint r = threadgroup_position_in_grid.x;
    threadgroup float red[32];  // TG/32 simdgroup partials
    threadgroup float redq[32];
    // Threadgroup 0 also reduces sum(h^2) into mixes[nmix] — hc_pre's rms
    // input. It already streams every element of h for its own dot, so the
    // squares are ~free here, and hc_pre is then freed from making its OWN
    // full pass over h. That pass is what forced hc_pre into ONE threadgroup
    // (~1/64 of the chip's bandwidth); see Stage 4l. Reduction shape is
    // identical to the pass it replaces, so the rms stays bit-identical.
    const int nmix = (2 + hcn) * hcn;
    float a = 0.0f, q = 0.0f;
    for (int i = tid; i < hcd; i += TG) {
        float hv = float(h[i]);
        a += float(fn[r * hcd + i]) * hv;
        if (r == 0) q += hv * hv;
    }
    a = simd_sum(a);
    if ((tid & 31u) == 0) red[tid / 32] = a;
    if (r == 0) {                          // r is uniform across the group
        q = simd_sum(q);
        if ((tid & 31u) == 0) redq[tid / 32] = q;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float t = 0.0f;
        for (int i = 0; i < TG / 32; ++i) t += red[i];
        mixes[r] = t;
        if (r == 0) {
            float s = 0.0f;
            for (int i = 0; i < TG / 32; ++i) s += redq[i];
            mixes[nmix] = s;
        }
    }
"""

_HC_PRE_SRC = """
    uint tid = thread_position_in_threadgroup.x;
    const int TG = 1024;
    // HC_PRE_SPLIT threadgroups over the output dim d. This kernel used to be
    // a SINGLE threadgroup because it had to reduce sum(h^2) over all of h
    // before it could do anything else — and one threadgroup gets ~1/64 of
    // this chip's bandwidth. dsv4_hc_mix_k now hands that sum over (it
    // already streams all of h), so the only thing left is the per-output
    // reduction, which splits cleanly. The gates and Sinkhorn are recomputed
    // in every part: they are 16 floats of simdgroup-local work, far cheaper
    // than another dispatch and its barrier.
    const int NSPLIT = HC_PRE_SPLIT;
    const int hcn = params[0];
    const int d = params[1];
    const int hcd = hcn * d;
    const float rms_eps = feps[0];
    const float hc_eps = feps[1];
    const int part = int(threadgroup_position_in_grid.x);

    threadgroup float pc[64];      // pre[hc], post[hc], comb[hc*hc]

    float rms = rsqrt(mixes[(2 + hcn) * hcn] / hcd + rms_eps);

    // Gates + Sinkhorn on ONE simdgroup: lane l < hcn*hcn holds comb element
    // (l/hcn, l%hcn); row/col sums are gathered lane-by-lane with simd_shuffle
    // in the SAME serial order as the barriered version's inner loops, and
    // every mul/add keeps its two-step shape, so results stay BIT-identical.
    // The barriered version paid 2 threadgroup barriers per Sinkhorn iteration
    // (~40 total, 20 iters) to sync a hcn x hcn matrix across a 1024-thread
    // group; a simdgroup is implicitly synchronous, so this pays none.
    if (tid < 32) {
        int l = tid;
        int r = l / hcn, c = l % hcn;
        float cb = 0.0f;
        if (l < hcn * hcn) {
            float mt = float(mixes[2 * hcn + l]) * rms;
            cb = mt * scale[2] + base[2 * hcn + l];
        }
        float m = simd_shuffle(cb, r * hcn);                 // row softmax (+ hc_eps)
        for (int k = 1; k < hcn; ++k) m = max(m, simd_shuffle(cb, r * hcn + k));
        float e = exp(cb - m);
        float s = 0.0f;
        for (int k = 0; k < hcn; ++k) s += simd_shuffle(e, r * hcn + k);
        cb = e / s + hc_eps;
        s = 0.0f;                                            // first column normalize
        for (int k = 0; k < hcn; ++k) s += simd_shuffle(cb, c + hcn * k);
        cb /= (s + hc_eps);
        for (int it = 0; it < iters[0] - 1; ++it) {
            s = 0.0f;                                        // row normalize
            for (int k = 0; k < hcn; ++k) s += simd_shuffle(cb, r * hcn + k);
            cb /= (s + hc_eps);
            s = 0.0f;                                        // column normalize
            for (int k = 0; k < hcn; ++k) s += simd_shuffle(cb, c + hcn * k);
            cb /= (s + hc_eps);
        }
        // every part computes identical values; only part 0 publishes the
        // shared post/comb outputs that dsv4_hc_post_k reads next.
        if (l < hcn * hcn) {
            pc[2 * hcn + l] = cb;
            if (part == 0) comb[l] = cb;
        }
        if (l < hcn) {
            float mt0 = float(mixes[l]) * rms;
            float mt1 = float(mixes[hcn + l]) * rms;
            float pst = 2.0f / (1.0f + exp(-(mt1 * scale[1] + base[hcn + l])));
            pc[l] = 1.0f / (1.0f + exp(-(mt0 * scale[0] + base[l]))) + hc_eps;
            pc[hcn + l] = pst;
            if (part == 0) post[l] = pst;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int chunk = (d + NSPLIT - 1) / NSPLIT;
    const int lo = part * chunk;
    const int hi = min(lo + chunk, d);
    for (int i = lo + (int)tid; i < hi; i += TG) {
        float a = 0.0f;
        for (int j = 0; j < hcn; ++j) a += pc[j] * float(h[j * d + i]);
        y[i] = T(a);
    }
"""

_HC_POST2_SRC = """
    // hc_post with the routed+shared add fused: x = T(a + b) computed
    // in-register (the T() round matches what the standalone dsv4_add_k
    // wrote, so results are bit-identical to the two-dispatch form).
    // NOTE a is float32 (moe_w2's y_routed), b is bfloat (shared) — the
    // same asymmetry dsv4_add_k had; binding both as bfloat reinterprets
    // the float buffer and NaNs the whole block (found the hard way).
    // Same grid contract as dsv4_hc_post_k: hcn * NSPLIT threadgroups.
    uint tid = thread_position_in_threadgroup.x;
    const int TG = 256;
    const int NSPLIT = 8;
    const int hcn = params[0];
    const int d = params[1];
    uint hc_ = threadgroup_position_in_grid.x / NSPLIT;
    const int part = int(threadgroup_position_in_grid.x % NSPLIT);
    const int chunk = (d + NSPLIT - 1) / NSPLIT;
    const int lo = part * chunk;
    const int hi = min(lo + chunk, d);
    for (int i = lo + (int)tid; i < hi; i += TG) {
        float xv = float(T(float(a[i]) + float(b[i])));
        float acc = post[hc_] * xv;
        for (int j = 0; j < hcn; ++j) acc += comb[j * hcn + hc_] * float(residual[j * d + i]);
        y[hc_ * d + i] = T(acc);
    }
"""

_HC_POST_SRC = """
    // y[k, d] = post[k]*x[d] + sum_j comb[j, k] * residual[j, d]
    // NOTE the comb index order: the reference is
    //   einsum("bsjk,bsjd->bskd", comb, residual)
    // so the SOURCE stream j is comb's first axis and the OUTPUT stream k
    // its second. Reading comb[k*hcn + j] transposes the mixing matrix —
    // invisible at hc=2 (every 2x2 doubly-stochastic matrix is symmetric,
    // which is why the tiny-config tests passed) and worth 14.5% of the
    // block output at the real hc_mult=4.
    // grid = hcn * NSPLIT threadgroups: hcn alone (4 TGs) underfilled the
    // chip, so each hc row is split into NSPLIT d-slices. NSPLIT is a
    // compile-time constant — every dispatch site's grid must move with it.
    uint tid = thread_position_in_threadgroup.x;
    const int TG = 256;
    const int NSPLIT = 8;
    const int hcn = params[0];
    const int d = params[1];
    uint hc_ = threadgroup_position_in_grid.x / NSPLIT;
    const int part = int(threadgroup_position_in_grid.x % NSPLIT);
    const int chunk = (d + NSPLIT - 1) / NSPLIT;
    const int lo = part * chunk;
    const int hi = min(lo + chunk, d);
    for (int i = lo + (int)tid; i < hi; i += TG) {
        float a = post[hc_] * float(x[i]);
        for (int j = 0; j < hcn; ++j) a += comb[j * hcn + hc_] * float(residual[j * d + i]);
        y[hc_ * d + i] = T(a);
    }
"""

_hc_kernels = None


def _get_hc_kernels():
    global _hc_kernels
    if _hc_kernels is None:
        # NOTE: _HC_PRE_SRC now consumes precomputed raw mixes (fn.h dots from
        # _HC_MIX_SRC) — the twin's inputs must match the body's buffer names.
        pre = mx.fast.metal_kernel(
            name="sh_dsv4_hc_pre_k",
            input_names=["h", "mixes", "scale", "base", "params", "feps", "iters"],
            output_names=["y", "post", "comb"],
            source=_HC_PRE_SRC,
            header=_STATIC_DEFINES,
        )
        post = mx.fast.metal_kernel(
            name="sh_dsv4_hc_post_k",
            input_names=["x", "residual", "post", "comb", "params"],
            output_names=["y"],
            source=_HC_POST_SRC,
        )
        _hc_kernels = (pre, post)
    return _hc_kernels


# --- fused batch-1 MoE kernels (decode) -------------------------------------
#
# gather_qmm at batch 1 measured 14 ms/token against ~3 ms of pure bandwidth
# (docs/benchmarks/deepseek-v4.md). These two kernels do the routed-expert
# FFN for ONE token with full-chip parallelism: K1 grids (expert x inner-row)
# simdgroups computing gate/up dots + clipped SwiGLU into h; K2 grids output
# dims reducing down_proj against h across the active experts, applying the
# routing weights. 2-bit affine unpack (16 values per uint32, one 64-group
# per word) happens in-register. Quantized builds only (bits=2, gs=64).

_MOE_K1_SRC = """
    // one simdgroup per (expert_slot, inner_row); x read straight from
    // device — it is 8 KB and L2-hot across all threadgroups. Staging it in
    // 16 KB of threadgroup memory capped residency at 2 TGs/core and cost a
    // barrier; removing the stage measured -1.6 ms/token (Stage 3l probe).
    uint sg_id = simdgroup_index_in_threadgroup;
    uint sg_global = threadgroup_position_in_grid.x * 8 + sg_id;
    uint lane = thread_index_in_simdgroup;
    const int n_act = params[0];
    const int d_model = params[1];
    const int d_inner = params[2];
    const float limit = feps[0];

    // One inner row per simdgroup. Pairing two rows here (the trick that is
    // worth -1.6 ms in moe_w2) measured 4.80 -> 4.90 ms: this kernel already
    // carries TWO weight streams (gate and up), so a pair needs four live
    // streams and four accumulators, and the register cost ate the halved x
    // loads. See Stage 4k — do not re-pair without a new measurement.
    uint e_slot = sg_global / (uint)d_inner;
    uint row = sg_global % (uint)d_inner;
    if (e_slot >= (uint)n_act) return;
    int e = idxs[e_slot];
    if (e < 0) { if (lane == 0) h[e_slot * d_inner + row] = 0.0f; return; }

    const int words = d_model / QE_VPW;
    const int wpg = QE_WPG;                   // packed words per scale group
    const uint gbase = ((uint)e * (uint)d_inner + row) * (uint)words;
    const uint sbase = ((uint)e * (uint)d_inner + row) * (uint)(d_model / QE_GS);

    float acc_g = 0.0f;
    float acc_u = 0.0f;
    for (int w = lane; w < words; w += 32) {
        uint pg = gw[gbase + w];
        uint pu = uw[gbase + w];
        uint g_ = w / wpg;
        float sgv = float(gs_[sbase + g_]);
        float bgv = float(gb[sbase + g_]);
        float suv = float(us[sbase + g_]);
        float buv = float(ub[sbase + g_]);
        const device bfloat* xv = x + w * QE_VPW;
        float ag = 0.0f, au = 0.0f, sx = 0.0f;
        #pragma unroll
        for (int j = 0; j < QE_VPW; ++j) {
            float xj = float(xv[j]);
            ag += float((pg >> (QE_BITS * j)) & QE_MASK) * xj;
            au += float((pu >> (QE_BITS * j)) & QE_MASK) * xj;
            sx += xj;
        }
        acc_g += ag * sgv + sx * bgv;
        acc_u += au * suv + sx * buv;
    }
    acc_g = simd_sum(acc_g);
    acc_u = simd_sum(acc_u);
    if (lane == 0) {
        if (limit > 0.0f) {
            acc_u = clamp(acc_u, -limit, limit);
            acc_g = min(acc_g, limit);
        }
        h[e_slot * d_inner + row] = (acc_g / (1.0f + exp(-acc_g))) * acc_u;
    }
"""

#: Output dims one moe_w2 simdgroup computes. Widening this tile divides the
#: dominant h stream (see the kernel body); 1 -> 2 measured 4.65 -> 3.07 ms.
#: The shared body is compiled twice — native replay and the mx.fast twin — so
#: the value ships as a #define from this one constant.
_W2_ROWS = 4

#: Threadgroups dsv4_hc_pre_k splits the output dim over (see the kernel body).
_HC_PRE_SPLIT = 8

def pack_defines(tag: str, bits: int, gs: int) -> str:
    """Kernel constants for one weight class: how a packed uint32 is cut up."""
    vpw = 32 // bits                                  # values per packed word
    return "".join(f"#define {tag}_{k}{v}\n" for k, v in (
        ("BITS ", bits), ("VPW  ", vpw), ("MASK ", f"{(1 << bits) - 1}u"),
        ("GS   ", gs), ("WPG  ", gs // vpw)))         # packed words per group


class QuantSpec(NamedTuple):
    """The weight packing the decode kernels must be compiled for.

    The kernels index packed uint32 words arithmetically — values per word,
    words per scale group, shift and mask all follow from (bits, group_size) —
    so these CANNOT be constants: a build with a different recipe would be read
    at the right addresses with the wrong stride and return plausible garbage
    rather than failing. Two classes, because the converter quantizes them
    differently: the ROUTED experts (QuantizedSwitchLinear) and everything
    DENSE (attention, shared expert, embedding, head).
    """

    dense_bits: int
    dense_gs: int
    exp_bits: int
    exp_gs: int

    #: bits that tile a uint32 exactly. MLX also packs 3/5/6, but not on a word
    #: boundary, so the shift/mask form these kernels use does not apply.
    SUPPORTED_BITS = (2, 4, 8)

    #: what mx.quantize emits; also what MLX's library qmv is specialized for.
    SUPPORTED_GROUP_SIZES = (32, 64, 128)

    @classmethod
    def from_model(cls, model) -> "QuantSpec":
        """Derive the spec from a loaded model, or raise explaining why this
        build cannot be served by the native path."""
        seen: dict[str, set] = {"dense": set(), "exp": set()}
        dtypes: set = set()
        modes: set = set()

        def visit(path, m):
            bits, gs = getattr(m, "bits", None), getattr(m, "group_size", None)
            if bits is None or gs is None:
                return
            # By PATH, not by class: attn.wo_a is a QuantizedSwitchLinear too
            # (the o_groups low-rank O projection) but carries the DENSE
            # recipe, and sh_dsv4_wo_a_k unpacks it as dense. Only the routed
            # experts under ffn.experts feed moe_w13/moe_w2.
            kind = "exp" if ".ffn.experts." in f".{path}." else "dense"
            seen[kind].add((int(bits), int(gs)))
            modes.add(getattr(m, "mode", "affine"))
            sc = getattr(m, "scales", None)
            if sc is not None:
                dtypes.add(sc.dtype)

        model.apply_to_modules(visit)
        for kind in ("dense", "exp"):
            if not seen[kind]:
                raise ValueError(
                    f"native decode needs a quantized build: no {kind} "
                    f"quantized weights found")
            if len(seen[kind]) > 1:
                raise ValueError(
                    f"native decode needs ONE {kind} recipe, found "
                    f"{sorted(seen[kind])} — the kernels are compiled per spec")
        if modes - {"affine"}:
            raise ValueError(
                f"native decode unpacks AFFINE quantization; build uses "
                f"{sorted(modes)} — a different mode packs the same bytes "
                f"with different meaning")
        if dtypes != {mx.bfloat16}:
            raise ValueError(
                f"native decode reads scales/biases as bfloat16, build has "
                f"{sorted(str(d) for d in dtypes)}")
        (db, dg), (eb, eg) = seen["dense"].pop(), seen["exp"].pop()
        for bits in (db, eb):
            if bits not in cls.SUPPORTED_BITS:
                raise ValueError(
                    f"native decode supports {cls.SUPPORTED_BITS}-bit packing, "
                    f"build has {bits}-bit")
        for bits, gs in ((db, dg), (eb, eg)):
            if gs not in cls.SUPPORTED_GROUP_SIZES:
                raise ValueError(
                    f"group_size {gs} is not one MLX affine quantization "
                    f"produces {cls.SUPPORTED_GROUP_SIZES}")
            # No divisibility check needed: every SUPPORTED_GROUP_SIZES is a
            # multiple of the values a uint32 holds at every SUPPORTED_BITS.
        return cls(db, dg, eb, eg)

    def defines(self) -> str:
        """Both classes — what the native library needs (it compiles every
        kernel). A twin that only unpacks one class asks for that one."""
        return (pack_defines("QD", self.dense_bits, self.dense_gs)
                + pack_defines("QE", self.exp_bits, self.exp_gs))

    def qmv_kernel(self, dtype: str = "bfloat16") -> str:
        """MLX's library qmv specialization matching the DENSE recipe."""
        return f"affine_qmv_fast_{dtype}_t_gs_{self.dense_gs}_b_{self.dense_bits}_batch_0"


#: Compile-time constants the shared kernel bodies read that do NOT depend on
#: the model. The bodies are built twice — once into the native replay library,
#: once per mx.fast twin — so both get this same text.
_STATIC_DEFINES = (
    f"#define W2_ROWS {_W2_ROWS}\n"
    f"#define HC_PRE_SPLIT {_HC_PRE_SPLIT}\n"
    # Order-preserving float -> uint: a < b  <=>  key(a) < key(b), so a binary
    # search over the uint range is a binary search over the float values and
    # terminates exactly. Used by the indexer's top-k threshold search.
    "static inline uint sh_order_key(float f) {\n"
    "    uint u = as_type<uint>(f);\n"
    "    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);\n"
    "}\n"
)


def _kernel_defines(spec: "QuantSpec | None") -> str:
    """Full define block for a kernel build: model-independent tile widths plus
    the packing of THIS model, when the kernel unpacks weights."""
    return _STATIC_DEFINES + (spec.defines() if spec is not None else "")

_MOE_K2_SRC = """
    // threadgroup covers 8 output dims; h read straight from device — the
    // active experts' h is ~48 KB and L2-hot across all threadgroups.
    // Staging each expert into threadgroup memory serialized the TG behind
    // 2 barriers per expert; removing it measured -1.6 ms/token (Stage 3l).
    // W2_ROWS output dims per simdgroup. Every output dim reduces against the
    // SAME h (all active experts, ~57 KB), so h is by far the larger of the
    // two streams a row reads — its packed weights are only ~3.5 KB. Widening
    // the tile divides that dominant stream: 1 -> 2 rows measured
    // 4.65 -> 3.07 ms. The same trick LOSES in moe_w13 and wo_a, where the
    // activation is 8 KB against 2-4 KB of weights (Stage 4k) — the rule is
    // the activation:weight ratio, not the kernel.
    // W2_ROWS must divide d_model, which _moe_kernel_usable enforces.
    uint sg_id = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint dim0 = (threadgroup_position_in_grid.x * 8 + sg_id) * W2_ROWS;
    const int n_act = params[0];
    const int d_model = params[1];
    const int d_inner = params[2];
    const int words = d_inner / QE_VPW;
    const int wpg = QE_WPG;
    const uint meta = (uint)(d_inner / QE_GS);
    // dim0 is uniform across the simdgroup, so this return takes all 32 lanes
    // with it and no simd_sum is left half-populated.
    if (dim0 >= (uint)d_model) return;

    float acc[W2_ROWS];
    #pragma unroll
    for (int r = 0; r < W2_ROWS; ++r) acc[r] = 0.0f;

    for (int s = 0; s < n_act; ++s) {
        int e = idxs[s];
        if (e < 0) continue;
        const device float* hs = h + s * d_inner;
        float we = wts[s];
        const uint base = ((uint)e * (uint)d_model + dim0) * (uint)words;
        const uint sbase = ((uint)e * (uint)d_model + dim0) * meta;
        float a[W2_ROWS];
        #pragma unroll
        for (int r = 0; r < W2_ROWS; ++r) a[r] = 0.0f;
        for (int w = lane; w < words; w += 32) {
            uint p[W2_ROWS];
            #pragma unroll
            for (int r = 0; r < W2_ROWS; ++r) p[r] = dw[base + r * words + w];
            uint g_ = w / wpg;
            // scale/bias into named locals BEFORE the inner loop, exactly as
            // the one-row original did. Reading them inline in the `a[r] +=`
            // expression instead lets the compiler contract that FMA chain
            // differently: same algebra, ~4e-5 different output, which over 43
            // layers moved native ppl 3.651 -> 3.672 (Stage 4k). Keep the shape.
            float sc[W2_ROWS], bi[W2_ROWS];
            #pragma unroll
            for (int r = 0; r < W2_ROWS; ++r) {
                sc[r] = float(ds_[sbase + r * meta + g_]);
                bi[r] = float(db[sbase + r * meta + g_]);
            }
            const device float* hv = hs + w * QE_VPW;
            float aw[W2_ROWS];
            #pragma unroll
            for (int r = 0; r < W2_ROWS; ++r) aw[r] = 0.0f;
            float sw = 0.0f;
            #pragma unroll
            for (int j = 0; j < QE_VPW; ++j) {
                float hj = hv[j];
                #pragma unroll
                for (int r = 0; r < W2_ROWS; ++r)
                    aw[r] += float((p[r] >> (QE_BITS * j)) & QE_MASK) * hj;
                sw += hj;
            }
            #pragma unroll
            for (int r = 0; r < W2_ROWS; ++r) a[r] += aw[r] * sc[r] + sw * bi[r];
        }
        #pragma unroll
        for (int r = 0; r < W2_ROWS; ++r) acc[r] += we * a[r];
    }
    #pragma unroll
    for (int r = 0; r < W2_ROWS; ++r) acc[r] = simd_sum(acc[r]);
    if (lane == 0) {
        #pragma unroll
        for (int r = 0; r < W2_ROWS; ++r) y[dim0 + r] = acc[r];
    }
"""

@functools.lru_cache(maxsize=8)
def _get_moe_kernels(exp_bits: int, exp_gs: int):
    """The routed-expert kernels, compiled for ONE expert recipe. No default:
    the packing decides how the body reads every weight word, so the caller
    must say which build it holds."""
    hdr = _STATIC_DEFINES + pack_defines("QE", exp_bits, exp_gs)
    k1 = mx.fast.metal_kernel(
        name="sh_dsv4_moe_w13",
        input_names=["x", "gw", "gs_", "gb", "uw", "us", "ub", "idxs", "params", "feps"],
        output_names=["h"],
        source=_MOE_K1_SRC,
        header=hdr,
    )
    k2 = mx.fast.metal_kernel(
        name="sh_dsv4_moe_w2",
        input_names=["h", "dw", "ds_", "db", "idxs", "wts", "params"],
        output_names=["y"],
        source=_MOE_K2_SRC,
        header=hdr,
    )
    return (k1, k2)


@functools.lru_cache(maxsize=16)
def _moe_params(n_act: int, d_model: int, d_inner: int, limit: float):
    return (
        mx.array([n_act, d_model, d_inner], dtype=mx.int32),
        mx.array([limit], dtype=mx.float32),
    )


def _moe_kernel_usable(glu) -> bool:
    """Can the fused batch-1 routed-expert kernels serve this SwitchGLU?

    The bit width used to be pinned at 2 here, matching constants that no
    longer exist in the kernel bodies — they derive their packing from the
    weights now (QuantSpec), so the gate follows the same supported set. The
    remaining conditions are what the INDEXING needs: one recipe across the
    three projections, dimensions that divide the scale group, and an output
    vector the moe_w2 tile covers exactly.
    """
    from mlx_lm.models.switch_layers import QuantizedSwitchLinear

    if os.environ.get("SOLOHEAVEN_METAL_KERNELS", "1") == "0":
        return False
    if not mx.metal.is_available():
        return False
    projs = (glu.gate_proj, glu.up_proj, glu.down_proj)
    if not all(isinstance(p, QuantizedSwitchLinear) for p in projs):
        return False
    # AFFINE packing only. mxfp4 stores a shared exponent instead of a
    # scale/bias pair and packs differently, so these kernels would read it at
    # the right addresses with the wrong meaning — plausible garbage, no error.
    # Qwen3.5-122B-A10B ships mxfp4, so this is not hypothetical.
    if any(getattr(p, "mode", "affine") != "affine" for p in projs):
        return False
    recipes = {(p.bits, p.group_size) for p in projs}
    if len(recipes) != 1:
        return False
    bits, gs = recipes.pop()
    return (
        bits in QuantSpec.SUPPORTED_BITS
        and gs in QuantSpec.SUPPORTED_GROUP_SIZES
        and glu.gate_proj.input_dims % gs == 0
        and glu.down_proj.input_dims % gs == 0
        # moe_w2 emits _W2_ROWS output dims per simdgroup, so the tile must
        # divide the output vector exactly
        and glu.down_proj.output_dims % _W2_ROWS == 0
    )


def _moe_routed_kernel(glu, x, indices, weights, limit: float) -> mx.array:
    """Routed-expert output for one token via the fused kernels.
    ``x`` [1,1,D]; ``indices`` [1,1,K] int32; ``weights`` [1,1,K]. Returns
    [1,1,D] fp32."""
    k1, k2 = _get_moe_kernels(glu.gate_proj.bits, glu.gate_proj.group_size)
    d_model = glu.gate_proj.input_dims
    d_inner = glu.gate_proj.output_dims
    n_act = indices.size
    params, feps = _moe_params(n_act, d_model, d_inner, limit)
    idxs = indices.reshape(-1).astype(mx.int32)
    tg = 256
    sgs1 = n_act * d_inner
    (h,) = k1(
        inputs=[x.reshape(-1), glu.gate_proj.weight, glu.gate_proj.scales,
                glu.gate_proj.biases, glu.up_proj.weight, glu.up_proj.scales,
                glu.up_proj.biases, idxs, params, feps],
        grid=(((sgs1 + 7) // 8) * tg, 1, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(n_act * d_inner,)],
        output_dtypes=[mx.float32],
    )
    (y,) = k2(
        inputs=[h, glu.down_proj.weight, glu.down_proj.scales,
                glu.down_proj.biases, idxs, weights.reshape(-1).astype(mx.float32),
                params],
        grid=(((d_model + 8 * _W2_ROWS - 1) // (8 * _W2_ROWS)) * tg, 1, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(d_model,)],
        output_dtypes=[mx.float32],
    )
    return y.reshape(1, 1, d_model)


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

    def decode_step_math(self, x: mx.array, input_ids: mx.array) -> mx.array:
        """Single-token MoE for the compiled decode path: fused batch-1
        kernels for the routed experts on quantized builds; identical eager
        math (the canonical, differential-tested reference) otherwise."""
        weights, indices = self.gate(x, input_ids)
        if _moe_kernel_usable(self.experts):
            y = _moe_routed_kernel(
                self.experts, x, indices, weights, self.shared_experts.limit
            )
        else:
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


def _hc_pre_math(h, fn, scale, base, hc, iters, hc_eps, rms_eps):
    """Raw hc_pre: mixes + Sinkhorn + pre-reduction. Shared by the eagerly
    compiled wrapper below and the whole-layer compiled decode step (which
    must inline it — nesting compiled functions is not worth the risk)."""
    b, s, hcd, d = h.shape
    flat = h.reshape(b, s, hcd * d).astype(mx.float32)
    r = mx.rsqrt(flat.square().mean(-1, keepdims=True) + rms_eps)
    mixes = (flat @ fn.T) * r
    pre, post, comb = hc_split_sinkhorn(mixes, scale, base, hc, iters, hc_eps)
    y = (pre[..., None] * flat.reshape(h.shape)).sum(axis=2)
    return y.astype(h.dtype), post, comb


def _hc_post_math(x, residual, post, comb):
    y = post[..., None] * x[..., None, :] + mx.einsum(
        "bsjk,bsjd->bskd", comb, residual.astype(mx.float32)
    )
    return y.astype(x.dtype)


@functools.lru_cache(maxsize=None)
def _compiled_hc_pre(hc: int, iters: int, hc_eps: float, rms_eps: float):
    """Fused hc_pre: mixes + the 20-iteration Sinkhorn + the pre-reduction.

    Eager, this is ~130 tiny kernel launches per call and it runs 86 times per
    decoded token (2 per layer x 43); measured at 39 ms/token of pure launch
    overhead — 35% of decode — for arithmetic on 4x4 matrices. The function is
    pure (no cache mutation), so mx.compile is safe; it re-traces per input
    shape, which in serving is just decode [1,1,...] plus a handful of prefill
    chunk sizes. Cached per (hc, iters, eps) so tiny test configs and the real
    config coexist.
    """

    def f(h, fn, scale, base):
        b, s, hcd, d = h.shape
        flat = h.reshape(b, s, hcd * d).astype(mx.float32)
        r = mx.rsqrt(flat.square().mean(-1, keepdims=True) + rms_eps)
        mixes = (flat @ fn.T) * r
        pre, post, comb = hc_split_sinkhorn(mixes, scale, base, hc, iters, hc_eps)
        y = (pre[..., None] * flat.reshape(h.shape)).sum(axis=2)
        return y.astype(h.dtype), post, comb

    return mx.compile(f)


@functools.lru_cache(maxsize=None)
def _compiled_hc_post():
    def f(x, residual, post, comb):
        y = post[..., None] * x[..., None, :] + mx.einsum(
            "bsjk,bsjd->bskd", comb, residual.astype(mx.float32)
        )
        return y.astype(x.dtype)

    return mx.compile(f)


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
        return _compiled_hc_pre(self.hc, self.iters, self.hc_eps, self.eps)(
            h, fn, scale, base
        )

    @staticmethod
    def _post(x, residual, post, comb):
        return _compiled_hc_post()(x, residual, post, comb)

    def decode_step_math(self, h, input_ids, offset, ring, *state_arrays):
        """The whole layer for one decoded token, as a pure function — the
        body that gets mx.compile'd. HC math is inlined raw (the eager path's
        separately-compiled HC wrappers must not nest inside this trace)."""
        residual = h
        x, post, comb = _hc_pre_math(
            h, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base,
            self.hc, self.iters, self.hc_eps, self.eps,
        )
        x = self.attn_norm(x)
        x, new_ring, *new_state = self.attn.decode_step_math(
            x, ring, state_arrays, offset
        )
        h = _hc_post_math(x, residual, post, comb)

        residual = h
        x, post, comb = _hc_pre_math(
            h, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base,
            self.hc, self.iters, self.hc_eps, self.eps,
        )
        x = self.ffn.decode_step_math(self.ffn_norm(x), input_ids)
        h = _hc_post_math(x, residual, post, comb)
        return (h, new_ring, *new_state)

    def compiled_step(self):
        """Compiled decode step. mx.compile re-traces by input shape, which
        here only changes when a compressed cache crosses a 256-group capacity
        boundary; offset/n travel as traced scalars and never key the trace."""
        fn = getattr(self, "_step_fn", None)
        if fn is None:
            fn = mx.compile(self.decode_step_math)
            self._step_fn = fn
        return fn

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
        is_decode = (
            inputs.shape[0] == 1 and inputs.shape[1] == 1 and cache[0].offset > 0
        )
        if is_decode and _NATIVE_DECODE_ENABLED():
            try:
                return self._decode_native(inputs, cache)
            except Exception:  # noqa: BLE001 — fall back to compiled/eager once
                global _NATIVE_DECODE_BROKEN
                if not _NATIVE_DECODE_BROKEN:
                    _NATIVE_DECODE_BROKEN = True
                    import logging

                    logging.getLogger(__name__).exception(
                        "[deepseek_v4] native decode failed — compiled/eager fallback"
                    )
        if is_decode and _COMPILED_DECODE_ENABLED():
            try:
                return self._decode_compiled(inputs, cache)
            except Exception:  # noqa: BLE001 — fall back to the eager path once
                global _COMPILED_DECODE_BROKEN
                if not _COMPILED_DECODE_BROKEN:
                    _COMPILED_DECODE_BROKEN = True
                    import logging

                    logging.getLogger(__name__).exception(
                        "[deepseek_v4] compiled decode failed — eager fallback"
                    )
        h = self.embed(inputs)
        h = mx.broadcast_to(h[:, :, None, :], (*h.shape[:2], self.hc, h.shape[-1]))
        for layer, c in zip(self.layers, cache):
            h = layer(h, inputs, c)
        flat, mixes = _hc_mixes(h, self.hc_head_fn, self.eps)
        pre = mx.sigmoid(mixes * self.hc_head_scale + self.hc_head_base) + self.hc_eps
        x = (pre[..., None] * flat.reshape(h.shape)).sum(axis=2).astype(h.dtype)
        return self.head(self.norm(x))

    def _native_stale(self, dec, cache) -> bool:
        """The native decoder borrows the cache's arrays; it is stale the
        moment anything replaced them — a prefill batch, a state restore, a
        different session's cache list — detectable as list identity, offset
        divergence, or per-layer array identity changes (mx setitem allocates
        new buffers, so ANY non-native touch changes identity)."""
        if dec is None or dec.cache is not cache or dec.offset != cache[0].offset:
            return True
        if dec.overflowing():  # a compressed buffer is full — rebind over a grown one
            return True
        for i, c in enumerate(cache):
            if dec._arrays[f"L{i}_ring"] is not c.ring:
                return True
            if c.comp is not None and (
                dec._arrays[f"L{i}_kv_a"] is not c.comp.kv_state
                or dec._arrays[f"L{i}_buf"] is not c.comp.cache
            ):
                return True
        return False

    def _decode_native(self, inputs: mx.array, cache) -> mx.array:
        from mlx_soloheaven.native import decoder as _nd

        dec = getattr(self, "_native_dec", None)
        if self._native_stale(dec, cache):
            mx.synchronize()  # cache arrays were written on MLX's queue
            max_ctx = int(os.environ.get("SOLOHEAVEN_NATIVE_MAX_CONTEXT", "32768"))
            dec = _nd.NativeDecoder(self, max_context=max_ctx, cache=cache)
            self._native_dec = dec
        token = int(inputs.reshape(-1)[0])
        logits = dec.decode(token)
        # Copy AND evaluate before returning: the decoder reuses its logits
        # buffer next step, and a lazy copy would read whatever is there when
        # the engine's sampling graph finally evaluates.
        out = mx.array(logits).reshape(1, 1, -1)
        mx.eval(out)
        return out

    def _decode_compiled(self, inputs: mx.array, cache) -> mx.array:
        """One decoded token through the per-layer compiled steps.

        State is committed only after EVERY layer's function has been built
        and called (all lazily functional), so a trace failure anywhere falls
        back to the eager path with the caches untouched.
        """
        start = cache[0].offset
        offset = _scalar_i32(start)
        h = self.embed(inputs)
        h = mx.broadcast_to(h[:, :, None, :], (*h.shape[:2], self.hc, h.shape[-1]))

        staged = []
        for layer, c in zip(self.layers, cache):
            arrays = _pack_decode_state(c, layer)
            outs = layer.compiled_step()(h, inputs, offset, *arrays)
            h = outs[0]
            staged.append((c, outs[1:], layer.attn.ratio))
        # commit (python bookkeeping stays eval-free: completion is derivable
        # from the PYTHON offset, so n never forces a device sync)
        for c, outs, ratio in staged:
            _unpack_decode_state(c, outs, start, ratio or 1)
            c.offset = start + 1

        flat, mixes = _hc_mixes(h, self.hc_head_fn, self.eps)
        pre = mx.sigmoid(mixes * self.hc_head_scale + self.hc_head_base) + self.hc_eps
        x = (pre[..., None] * flat.reshape(h.shape)).sum(axis=2).astype(h.dtype)
        return self.head(self.norm(x))
