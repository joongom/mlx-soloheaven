"""Qwen3.5/3.6 MTP speculative decoding on the mlx-lm path.

Runs a DeepSeek-style Multi-Token-Prediction head (model_type
``qwen3_5_mtp``, e.g. mlx-community/Qwen3.6-35B-A3B-MTP-5bit) as a
speculative drafter for a qwen3_5 / qwen3_5_moe TARGET loaded via mlx-lm.
Self-contained: no mlx-vlm, no fork/patch of installed mlx-lm.

Head semantics (cross-checked against mlx-lm PR #990, mlx-vlm main's
qwen3_5_mtp drafter, vLLM qwen3_next_mtp.py; locally validated exact-greedy
on the real 35B target — see /tmp/qwen_mtp_probe/probe2.py):

    h_in = fc( concat[ pre_fc_norm_embedding(embed(y_{t+1})),
                       pre_fc_norm_hidden(h_t) ], axis=-1 )
    h    = MTPDecoderLayer(h_in)        # exactly 1 FULL-ATTENTION layer
    logits = TARGET.lm_head( head.norm(h) )

* ``h_t`` is the target's last-decoder-layer output BEFORE the final model
  norm (pre-final-norm residual). Installed mlx-lm 0.31.3 returns post-norm
  hidden from ``Qwen3_5TextModel.__call__``, so the target layer loop is
  driven manually here (verified byte-identical to ``model()`` logits —
  ``verify_layer_loop_parity``).
* The head has NO embed_tokens and NO lm_head — it borrows the target's.
* CRITICAL TRAP: the head config carries the TARGET's 40-entry
  layer_types / full_attention_interval=4. A naive ``DecoderLayer(...,
  layer_idx=0)`` would build GatedDeltaNet (linear attention) and fail to
  load. The head layer is built with ``layer_idx = full_attention_interval-1``
  to force full attention (matches the checkpoint's ``self_attn.*`` keys).
* NORM WEIGHTS: the local 5-bit head was split from an already-converted MLX
  checkpoint, so the zero-centered-RMSNorm +1 shift is ALREADY BAKED IN.
  Weights are loaded verbatim — NEVER through mlx-lm 0.31.3's qwen3_5
  ``sanitize`` (it +1-shifts ALL norms whenever any 'mtp.' key is present;
  fixed only in unmerged PR #990/#1320).

Block recursion (``block_size`` = drafts per round): for k >= 2 the head
re-feeds its OWN pre-head-norm output hidden (surrogate) — validated
locally at block_size=3 with mean 2.77/3 greedy acceptance.

Rollback (the target cache is 30x ArraysCache — recurrent, NOT trimmable —
plus 10x KVCache): rejected-draft rollback = ArraysCache state
SNAPSHOT/RESTORE (reference copies; mlx-lm's GatedDeltaNet replaces
``cache[0]/cache[1]`` functionally, never mutates in place) + KVCache trim,
then a REPLAY forward of the accepted tokens (replay logits verified == 0.0
diff vs the verify forward locally). Every restore is verified per-layer
fail-closed (commit 8491f1d philosophy): on any offset mismatch the round
raises ``MTPCacheCorruption`` -> the stream TERMINATES at the corruption
point (U6 — the target cache is unverifiable, so post-corruption plain
decoding would emit untrustworthy tokens) + ``on_cache_corruption``
invalidates the session cache; the engine surfaces an "error" terminal
frame and the next turn takes an honest MISS.

Head-cache pairing convention: head KV slot ``i`` = pair
``(h_i, embed(y_{i+1}))`` at absolute position ``i``. During decode the head
offset is ``target_offset - 1`` (the pending token's pair is written by the
next round's first draft step). At stream end ``finalize`` (1) commits the
pending pair into the head AND (2) forwards the pending token through the
TARGET — exactly what plain mlx-lm's lookahead ``generate_step`` does — so
the persisted target cache contains EVERY emitted token (the natural stop
token included) and ``target_offset == len(stored_token_ids)`` keeps the
same meaning on the MTP and non-MTP paths (the engine's chat-template
suffix builder relies on stored ids ending WITH the stop token; the next
turn's suffix starts with template glue like ``"\n"`` and never re-sends
the stop). The head therefore lands at ``target_offset - 1``: its LAST
slot would pair with the NEXT turn's first token, which is unknowable at
finalize time, so that slot is committed LAZILY at resume.
``on_finalize(final_token, final_hidden)`` surfaces the final pre-norm
hidden (h at the last target position); the engine stashes it, next
turn's reuse gate (``validate_mtp_cache_reuse``) requires ``head_offset ==
target_offset - 1`` plus a stashed hidden, and the runner entry commits
``(final_hidden, first_suffix_token)`` through the head once before
prefill. Fail-closed: missing hidden / any offset mismatch -> cold-fill
(mlx-lm issue #1292 failure class).
"""

from __future__ import annotations

import glob
import json
import logging
import os
from typing import Any, Callable, Generator, List, Optional, Sequence, Tuple

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.generate import generation_stream
from mlx_lm.models.cache import KVCache, make_prompt_cache

logger = logging.getLogger(__name__)

QWEN_MTP_MODEL_TYPE = "qwen3_5_mtp"


class MTPCacheCorruption(RuntimeError):
    """Fail-closed signal: a snapshot/restore/trim did not land exactly on
    the expected per-layer offsets (or a round invariant broke). The runner
    TERMINATES the stream at the corruption point (U6) and fires
    ``on_cache_corruption`` so the caller invalidates the session cache."""


# ---------------------------------------------------------------------------
# Head module + loader
# ---------------------------------------------------------------------------


class QwenMTPHead(nn.Module):
    """The MTP head module tree, named to match the checkpoint keys exactly:

    fc.{weight[,scales,biases]}, pre_fc_norm_embedding.weight,
    pre_fc_norm_hidden.weight, norm.weight, layers.N.self_attn.*,
    layers.N.mlp.{gate,switch_mlp.*,shared_expert.*,shared_expert_gate},
    layers.N.{input,post_attention}_layernorm.
    """

    def __init__(self, args, num_layers: int = 1):
        super().__init__()
        # Imported lazily by the loader; taking the class via args' module
        # would be circular — import here keeps mocks importable without mlx-lm.
        from mlx_lm.models.qwen3_5 import DecoderLayer

        h = args.hidden_size
        self.pre_fc_norm_hidden = nn.RMSNorm(h, eps=args.rms_norm_eps)
        self.pre_fc_norm_embedding = nn.RMSNorm(h, eps=args.rms_norm_eps)
        self.fc = nn.Linear(2 * h, h, bias=False)
        # Force FULL attention: layer_idx = full_attention_interval - 1 makes
        # DecoderLayer.is_linear False (the head checkpoint's single layer is
        # self_attn, NOT GatedDeltaNet, despite the copied target config).
        fa_idx = args.full_attention_interval - 1
        self.layers = [DecoderLayer(args, layer_idx=fa_idx) for _ in range(num_layers)]
        self.norm = nn.RMSNorm(h, eps=args.rms_norm_eps)


def read_model_type(model_path: str) -> Optional[str]:
    """Peek a model directory's config.json model_type (None if unreadable)."""
    try:
        cfg_path = os.path.join(os.path.expanduser(model_path), "config.json")
        with open(cfg_path) as f:
            return json.load(f).get("model_type")
    except Exception:  # noqa: BLE001 — caller treats None as "unknown"
        return None


def load_qwen_mtp_head(model_path: str) -> Tuple[QwenMTPHead, dict]:
    """Load a qwen3_5_mtp head checkpoint. Returns ``(head, info)``.

    * Weights are loaded VERBATIM with ``strict=True`` (all tensors must be
      consumed; no missing params) — never through mlx-lm's qwen3_5
      ``sanitize`` (0.31.3 +1-shifts all norms when 'mtp.' keys are present).
    * Pre-quantized checkpoints are rebuilt via ``nn.quantize`` using the
      config's quantization section, quantizing exactly the modules that
      have ``.scales`` tensors in the checkpoint.
    """
    from mlx_lm.models.qwen3_5 import TextModelArgs

    path = os.path.expanduser(model_path)
    with open(os.path.join(path, "config.json")) as f:
        cfg = json.load(f)
    model_type = cfg.get("model_type")
    if model_type != QWEN_MTP_MODEL_TYPE:
        raise ValueError(
            f"load_qwen_mtp_head: not a {QWEN_MTP_MODEL_TYPE} checkpoint "
            f"(model_type={model_type!r}) at {path}"
        )
    text_cfg = cfg.get("text_config", cfg)
    args = TextModelArgs.from_dict(text_cfg)
    num_layers = int(text_cfg.get("mtp_num_hidden_layers", 1) or 1)
    block_size = int(cfg.get("block_size", text_cfg.get("block_size", 3)) or 3)

    head = QwenMTPHead(args, num_layers=num_layers)

    weight_files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    if not weight_files:
        raise FileNotFoundError(f"no *.safetensors under {path}")
    weights: dict = {}
    for wf in weight_files:
        weights.update(mx.load(wf))

    quant = cfg.get("quantization")
    if quant:
        nn.quantize(
            head,
            group_size=int(quant.get("group_size", 64)),
            bits=int(quant.get("bits", 5)),
            mode=quant.get("mode", "affine"),
            # Per-path overrides (if present in the config) win; otherwise a
            # module is quantized iff the checkpoint carries scales for it.
            class_predicate=lambda p, m: (
                quant[p]
                if p in quant
                else (hasattr(m, "to_quantized") and f"{p}.scales" in weights)
            ),
        )

    # strict=True == full key accounting: every checkpoint tensor consumed,
    # every module parameter filled. NO norm +1 shift is applied (the MLX
    # conversion already baked it in — verified against the target's norms).
    head.load_weights(list(weights.items()), strict=True)
    mx.eval(head.parameters())

    info = {
        "model_type": model_type,
        "block_size": block_size,
        "num_layers": num_layers,
        "hidden_size": int(args.hidden_size),
        "vocab_size": int(args.vocab_size),
        "num_weights": len(weights),
    }
    return head, info


def assert_head_target_compat(info: dict, model: Any) -> None:
    """Fail loud at load if head and target disagree on hidden/vocab size
    (the head borrows the target's embed_tokens + lm_head)."""
    lm = getattr(model, "language_model", model)
    args = getattr(lm, "args", None)
    t_hidden = int(getattr(args, "hidden_size", -1) or -1)
    t_vocab = int(getattr(args, "vocab_size", -1) or -1)
    if t_hidden != info["hidden_size"] or t_vocab != info["vocab_size"]:
        raise ValueError(
            f"qwen3_5_mtp head incompatible with target: head "
            f"hidden/vocab={info['hidden_size']}/{info['vocab_size']} vs "
            f"target {t_hidden}/{t_vocab} — the head shares the target's "
            f"embeddings and lm_head, sizes must match exactly."
        )


# ---------------------------------------------------------------------------
# Target / head forward ops (manual layer loop — pre-final-norm hidden)
# ---------------------------------------------------------------------------


class QwenMTPOps:
    """Forward primitives for the MTP runner.

    The target loop is driven manually (mirrors mlx-lm 0.31.3
    ``Qwen3_5TextModel.__call__`` minus the final ``self.norm``) so the
    PRE-final-norm hidden can feed ``pre_fc_norm_hidden``. This depends on
    mlx-lm internals (``language_model.model.{embed_tokens,layers,norm,
    fa_idx,ssm_idx}``); construction fails fast if the tree changed, and
    ``verify_layer_loop_parity`` catches semantic drift (e.g. PR #990
    moving the final norm out of the text model) at load time.

    Tests pass a duck-typed stand-in to ``qwen_mtp_generate_step`` instead.
    """

    _REQUIRED_TM_ATTRS = ("embed_tokens", "layers", "norm", "fa_idx", "ssm_idx")

    def __init__(self, model: Any, head: Optional[QwenMTPHead]):
        lm = getattr(model, "language_model", model)
        tm = getattr(lm, "model", None)
        if tm is None or not all(hasattr(tm, a) for a in self._REQUIRED_TM_ATTRS):
            raise RuntimeError(
                "qwen_mtp: target does not expose the expected mlx-lm qwen3_5 "
                "module tree (language_model.model.{embed_tokens,layers,norm,"
                "fa_idx,ssm_idx}) — mlx-lm internals changed; refusing MTP."
            )
        args = getattr(lm, "args", None)
        self._tied = bool(getattr(args, "tie_word_embeddings", False))
        if not self._tied and not hasattr(lm, "lm_head"):
            raise RuntimeError("qwen_mtp: target exposes no lm_head")
        self.lm = lm
        self.tm = tm
        self.head = head

    # -- shared lm_head (borrowed by the head) --
    def _lm_head(self, h: mx.array) -> mx.array:
        if self._tied:
            return self.tm.embed_tokens.as_linear(h)
        return self.lm.lm_head(h)

    def target_hidden(self, toks2d: mx.array, cache: Sequence[Any]) -> mx.array:
        """Forward ``toks2d`` (1, T) through the target decoder layers;
        returns the PRE-final-norm hidden (1, T, H). Advances ``cache``."""
        from mlx_lm.models.base import create_attention_mask, create_ssm_mask

        tm = self.tm
        h = tm.embed_tokens(toks2d)
        fa_mask = create_attention_mask(h, cache[tm.fa_idx])
        ssm_mask = create_ssm_mask(h, cache[tm.ssm_idx])
        for layer, c in zip(tm.layers, cache):
            h = layer(h, mask=ssm_mask if layer.is_linear else fa_mask, cache=c)
        return h

    def target_logits(self, h: mx.array) -> mx.array:
        """Normal target output path: lm_head(final_norm(hidden))."""
        return self._lm_head(self.tm.norm(h))

    def head_hidden(
        self, hidden: mx.array, next_ids: mx.array, mtp_cache: Sequence[Any]
    ) -> mx.array:
        """One head forward over pairs (hidden_i, embed(next_ids_i)).
        Returns the PRE-head-norm output (the surrogate hidden for block
        recursion). Appends one KV slot per position to ``mtp_cache``."""
        from mlx_lm.models.base import create_attention_mask

        head = self.head
        e = head.pre_fc_norm_embedding(self.tm.embed_tokens(next_ids))
        hh = head.pre_fc_norm_hidden(hidden)
        x = head.fc(mx.concatenate([e, hh], axis=-1))
        mask = create_attention_mask(x, mtp_cache[0])
        for layer, c in zip(head.layers, mtp_cache):
            x = layer(x, mask=mask, cache=c)
        return x

    def head_logits(self, x: mx.array) -> mx.array:
        return self._lm_head(self.head.norm(x))


def verify_layer_loop_parity(model: Any, tol: float = 1e-3) -> float:
    """Startup self-check: the manual layer loop + final norm + lm_head must
    reproduce ``model()`` logits exactly (max|diff| was 0.0 on the real 35B).
    A future mlx-lm that returns pre-norm hidden from the text model (PR
    #990 merged) would double-apply the norm here — caught at load, fail
    fast rather than emit garbage drafts."""
    ops = QwenMTPOps(model, head=None)
    c_ref = make_prompt_cache(model)
    c_man = make_prompt_cache(model)
    tok = mx.array([[1]], mx.uint32)
    ref = model(tok, cache=c_ref)
    out = ops.target_logits(ops.target_hidden(tok, c_man))
    diff = float(mx.abs(ref - out).max().item())
    if not (diff <= tol):
        raise RuntimeError(
            f"qwen_mtp: layer-loop parity check FAILED (max|diff|={diff:g} > "
            f"{tol:g}) — manual pre-norm hidden loop no longer matches "
            f"model(); mlx-lm internals changed. Refusing MTP."
        )
    return diff


# ---------------------------------------------------------------------------
# Cache helpers (snapshot / restore / layout validation)
# ---------------------------------------------------------------------------


def make_head_cache(n: int) -> List[KVCache]:
    """Fresh KV cache entries for the head's full-attention layer(s)."""
    return [KVCache() for _ in range(n)]


def classify_cache(cache: Sequence[Any]) -> Tuple[List[int], List[int], List[int]]:
    """Split cache layer indices into (arrays_like, kv_like, unsupported).

    Duck-typed (mock-friendly, robust to mlx-lm renames):
      * arrays-like  — has a ``.cache`` list (ArraysCache/MambaCache style;
        recurrent state, NOT trimmable; snapshot/restore by reference).
      * kv-like      — has ``.trim`` and ``.offset`` (KVCache style).
      * unsupported  — anything else (QuantizedKVCache included via kv-like
        only if it exposes trim/offset; rotating caches would classify
        kv-like but this runner is only wired for qwen3_5 hybrids).
    """
    arrays_idx: List[int] = []
    kv_idx: List[int] = []
    other_idx: List[int] = []
    for i, c in enumerate(cache):
        if isinstance(getattr(c, "cache", None), list):
            arrays_idx.append(i)
        elif hasattr(c, "trim") and getattr(c, "offset", None) is not None:
            kv_idx.append(i)
        else:
            other_idx.append(i)
    return arrays_idx, kv_idx, other_idx


def validate_mtp_cache_reuse(
    prompt_cache: Sequence[Any],
    stored_ids: Sequence[int],
    prompt_token_ids: Sequence[int],
    n_target_layers: int,
    n_head_layers: int,
    resume_hidden: Optional[Any],
) -> Tuple[bool, str]:
    """Decide whether a REUSED session cache is safe for the MTP runner.

    Finalized-cache contract (see the module docstring + the finalize block
    in ``qwen_mtp_generate_step``): stored_ids include EVERY emitted token
    — the natural stop token too (finalize forwards it through the target,
    mirroring plain mlx-lm's lookahead ``generate_step``); the head sits at
    ``target_offset - 1`` with slots ``0..N-2`` committed as pairs
    ``(h_i, stored_ids[i+1])``; the LAST slot's pair token is next turn's
    first suffix token, committed lazily at resume from ``resume_hidden``
    (= h_{N-1}, surfaced by ``on_finalize``).

    Fail-closed requirements (any miss -> caller cold-fills):
      * layout: exactly n_target + n_head entries,
      * every offset-bearing target layer at ONE offset == len(stored_ids),
      * every head entry at offset == target_offset - 1 (finalize ran; the
        lazy last slot is still open),
      * pure append: the entire stored history is a strict prefix of the new
        prompt (the target's ArraysCache layers cannot trim, so divergence
        can never be rewound) with >= 1 new suffix token,
      * ``resume_hidden`` present (a mid-stream abort or a stale stash
        leaves it None — the lazy pair cannot be committed). Disk reloads
        never reach this check in the head-carrying layout: the session
        loader strips the trailing head entries on load (the stash is never
        persisted), so a reloaded session arrives head-less and takes the
        plain-fallback plan instead.
    """
    if len(prompt_cache) != n_target_layers + n_head_layers:
        return False, (
            f"layout {len(prompt_cache)} != {n_target_layers}+{n_head_layers}"
        )
    target = prompt_cache[:n_target_layers]
    head = prompt_cache[n_target_layers:]
    t_offs = {
        int(c.offset)
        for c in target
        if getattr(c, "offset", None) is not None
    }
    if len(t_offs) != 1:
        return False, f"target offsets diverged: {sorted(t_offs)[:4]}"
    t_off = t_offs.pop()
    h_offs = {int(getattr(c, "offset", -1)) for c in head}
    if t_off == 0 and not stored_ids:
        # fresh-shaped 41-entry cache (head trivially in lockstep at 0)
        if h_offs != {0}:
            return False, f"fresh cache head offset {sorted(h_offs)} != 0"
        return True, ""
    if h_offs != {t_off - 1}:
        return False, (
            f"head offset {sorted(h_offs)} != target {t_off} - 1 "
            f"(finalized head trails by the lazy last slot)"
        )
    if t_off != len(stored_ids):
        return False, f"offset {t_off} != stored ids {len(stored_ids)}"
    prefix_len = 0
    for a, b in zip(stored_ids, prompt_token_ids):
        if a != b:
            break
        prefix_len += 1
    if prefix_len != len(stored_ids):
        return False, f"divergence at {prefix_len} (ArraysCache untrimmable)"
    if prefix_len >= len(prompt_token_ids):
        return False, "no new suffix tokens"
    if resume_hidden is None:
        return False, (
            "resume hidden missing (mid-stream abort, disk reload, or stale "
            "stash) — the head's lazy last-slot pair cannot be committed"
        )
    return True, ""


# Reuse-plan actions (consumed by the engine's MTP gate). The fail-closed
# default is COLD_FILL; the two FALLBACK actions keep the cache but run the
# request on the PLAIN decode path. Falling back is ONE-WAY for the session:
# the head entries are only ever rebuilt by a cold-fill (which the engine
# never forces), so every later append-only turn re-plans FALLBACK_PLAIN —
# the right trade on the target (Qwen3.6 A3B MoE: measured MTP block1
# speedup 1.019x ≈ a wash vs plain decode, so reuse always beats
# re-speculation).
REUSE_MTP = "reuse_mtp"
REUSE_FALLBACK_PLAIN = "fallback_plain"
REUSE_FALLBACK_STRIP_HEAD = "fallback_strip_head"
REUSE_COLD_FILL = "cold_fill"


def _plain_append_consistent(
    target_entries: Sequence[Any],
    stored_ids: Sequence[int],
    prompt_token_ids: Sequence[int],
) -> bool:
    """Target-side eligibility for the plain-decode fallback: something IS
    cached, every offset-bearing entry sits at ONE offset == len(stored_ids),
    and the stored history is a strict prefix of the new prompt with >= 1
    suffix token. Mirrors ``validate_mtp_cache_reuse``'s target/append
    checks minus the head/hidden requirements — the plain path needs only
    these (pure-append reuse never trims, so the untrimmable ArraysCache
    layers are safe)."""
    if not stored_ids:
        # Nothing cached to reuse — a cold-fill costs nothing extra and
        # keeps MTP on (a fallback here would silently disable MTP on
        # every fresh-but-seen cache shape).
        return False
    offs = {
        int(c.offset)
        for c in target_entries
        if getattr(c, "offset", None) is not None
    }
    if offs != {len(stored_ids)}:
        return False
    if len(prompt_token_ids) <= len(stored_ids):
        return False
    for a, b in zip(stored_ids, prompt_token_ids):
        if a != b:
            return False
    return True


def plan_mtp_cache_reuse(
    prompt_cache: Sequence[Any],
    stored_ids: Sequence[int],
    prompt_token_ids: Sequence[int],
    n_target_layers: int,
    n_head_layers: int,
    resume_hidden: Optional[Any],
) -> Tuple[str, str]:
    """Decision matrix for a reused session cache under the MTP runner.

    Wraps ``validate_mtp_cache_reuse`` (unchanged contract) and, on failure,
    decides whether the cache is still PLAIN-reusable instead of always
    cold-filling — a cold-fill of a long history sacrifices cache reuse to
    keep MTP; reuse wins when the target side is provably consistent.

    Returns ``(action, why)``:
      * REUSE_MTP — validate passed; run the MTP runner on the cache.
      * REUSE_FALLBACK_PLAIN — head-less ``n_target`` layout (e.g. a plain
        base cache, a disk-reloaded session — the loader strips the head on
        load — or a session written by a non-MTP server) whose target side
        is pure-append-consistent: keep the cache, decode plain.
      * REUSE_FALLBACK_STRIP_HEAD — full ``n_target + n_head`` layout whose
        ONLY failure is head-side (stale head offset, or a missing/stale
        resume hidden) while the target side is pure-append-consistent:
        drop the head entries, keep the target, decode plain. The session
        then continues as a plain ``n_target``-entry session.
      * REUSE_COLD_FILL — anything else (divergence, desynced target
        offsets, no suffix, nothing cached): fail closed exactly as before
        — the untrimmable ArraysCache layers can never rewind divergence.
    """
    ok, why = validate_mtp_cache_reuse(
        prompt_cache,
        stored_ids,
        prompt_token_ids,
        n_target_layers,
        n_head_layers,
        resume_hidden,
    )
    if ok:
        return REUSE_MTP, ""
    n = len(prompt_cache)
    if n == n_target_layers and _plain_append_consistent(
        prompt_cache, stored_ids, prompt_token_ids
    ):
        return REUSE_FALLBACK_PLAIN, why
    if n == n_target_layers + n_head_layers and _plain_append_consistent(
        prompt_cache[:n_target_layers], stored_ids, prompt_token_ids
    ):
        # Layout + target offsets + pure-append all hold, so by elimination
        # the validate failure is head-side (head offset / resume hidden).
        return REUSE_FALLBACK_STRIP_HEAD, why
    return REUSE_COLD_FILL, why


def _eval_cache_states(cache: Sequence[Any]) -> None:
    """Best-effort materialization of lazy cache tensors (guards empty
    KVCache whose ``.state`` property raises when keys is None)."""
    arrays: List[mx.array] = []
    for c in cache:
        ks = getattr(c, "keys", None)
        if isinstance(ks, mx.array):
            arrays.append(ks)
        vs = getattr(c, "values", None)
        if isinstance(vs, mx.array):
            arrays.append(vs)
        cl = getattr(c, "cache", None)
        if isinstance(cl, list):
            arrays.extend(a for a in cl if isinstance(a, mx.array))
    if arrays:
        mx.eval(*arrays)


def mtp_prefill_base(
    tokens: Sequence[int],
    *,
    model: Any,
    head: Any,
    prompt_cache: List[Any],
    n_target_layers: int,
    prefill_step_size: int = 2048,
    ops: Optional[Any] = None,
) -> mx.array:
    """Prefill a FRESH ``n_target + n_head``-entry cache over ``tokens``
    into the FINALIZED shape and return the boundary hidden.

    Replicates the generate-step prompt prefill exactly: the target consumes
    ``tokens[:-1]`` in chunks while the head consumes the SAME hiddens paired
    with ``tokens[1:]`` (slot i = (h_i, tokens[i+1])); the held-out LAST
    token is then forwarded through the TARGET ONLY — landing precisely on
    the finalized contract ``validate_mtp_cache_reuse`` expects:
    ``target_offset == len(tokens)``, ``head_offset == len(tokens) - 1``
    (lazy last slot open), plus the returned ``h_{N-1}`` (1, 1, H) pre-final-
    norm hidden that pairs with the NEXT request's first suffix token.

    Used by the engine to build MTP-reusable BASE caches (shared system
    prefixes) outside generation. Per-chunk eval + clear_cache bound memory
    on long prefixes; everything runs on ``generation_stream``.
    """
    if prompt_cache is None or n_target_layers <= 0:
        raise ValueError("mtp_prefill_base requires an explicit prompt_cache")
    target_cache = prompt_cache[:n_target_layers]
    mtp_cache = prompt_cache[n_target_layers:]
    if not mtp_cache:
        raise ValueError("mtp_prefill_base: head cache entries missing")
    if not tokens:
        raise ValueError("mtp_prefill_base: empty token list")
    if ops is None:
        ops = QwenMTPOps(model, head)
    _, kv_idx, _ = classify_cache(target_cache)
    if not kv_idx:
        raise ValueError("mtp_prefill_base: no offset-bearing target layer")

    rem = mx.array(list(tokens), mx.uint32)
    with mx.stream(generation_stream):
        while int(rem.size) > 1:
            n = min(prefill_step_size, int(rem.size) - 1)
            chunk = rem[:n]
            hid_chunk = ops.target_hidden(chunk[None], target_cache)
            ops.head_hidden(hid_chunk, rem[1 : n + 1][None], mtp_cache)
            _eval_cache_states(prompt_cache)
            rem = rem[n:]
            mx.clear_cache()
        hid_last = ops.target_hidden(rem[None], target_cache)
    hid = hid_last[:, -1:, :]
    _eval_cache_states(prompt_cache)
    mx.eval(hid)
    # Post-condition (fail-closed): the finalized shape, verified per-layer.
    n_tok = len(tokens)
    bad_t = [
        (i, int(target_cache[i].offset))
        for i in kv_idx
        if int(target_cache[i].offset) != n_tok
    ]
    bad_h = [
        (i, int(getattr(c, "offset", -1)))
        for i, c in enumerate(mtp_cache)
        if int(getattr(c, "offset", -1)) != n_tok - 1
    ]
    if bad_t or bad_h:
        raise MTPCacheCorruption(
            f"mtp_prefill_base: cache did not land on the finalized shape "
            f"(want target {n_tok} / head {n_tok - 1}; bad target layers "
            f"{bad_t[:8]}, bad head layers {bad_h[:8]})"
        )
    return hid


# ---------------------------------------------------------------------------
# The MTP generate step
# ---------------------------------------------------------------------------


def qwen_mtp_generate_step(
    prompt: mx.array,
    model: Any,
    head: Any,
    *,
    block_size: int = 3,
    max_tokens: int = 256,
    sampler: Optional[Callable[[mx.array], mx.array]] = None,
    logits_processors: Optional[List[Callable[[mx.array, mx.array], mx.array]]] = None,
    prompt_cache: List[Any],
    n_target_layers: int,
    prefill_step_size: int = 2048,
    on_cache_corruption: Optional[Callable[[], None]] = None,
    on_finalize: Optional[Callable[[Optional[int], Optional[mx.array]], None]] = None,
    resume_hidden: Optional[mx.array] = None,
    ops: Optional[Any] = None,
) -> Generator[Tuple[int, Any, bool], None, None]:
    """MTP speculative decoding round loop. Yields ``(token, logprobs,
    from_draft)`` triples (same shape as ``pld_generate_step`` — adapt with
    ``_pld_response_adapter`` for detok/EOS).

    ``prompt`` is the post-prefix-trim SUFFIX (>= 1 token). ``prompt_cache``
    must hold the target's ``n_target_layers`` entries followed by the head's
    KV entries, in one of two entry shapes (validated by
    ``validate_mtp_cache_reuse``):
      * fresh: head offset == target offset == 0;
      * resumed finalized cache: head offset == target offset - 1 with the
        head's LAST slot uncommitted — ``resume_hidden`` (the previous
        stream's finalize hidden, h at the last target position) is then
        REQUIRED, and the entry commits the lazy pair
        ``(resume_hidden, suffix[0])`` through the head before prefill.

    Round shape (probe-validated): draft ``block_size`` tokens recursively on
    the head -> ONE target verify forward over [pending, drafts] -> accept
    prefix by sampled-token equality (greedy == exact match; the gemma4 MTP
    clone's rule) -> yield -> settle (snapshot-restore + replay of consumed
    tokens when not fully accepted) -> head trim + real-hidden recommit.

    Logits processors follow the PLD FIX-4 contract: per-position processing
    with EARLY STOP at the first rejected draft, history == suffix + emitted
    (call:emit is exactly 1:1; stateful processors stay correct).

    Fail-closed: every restore/trim is verified per-layer; any mismatch
    raises internally -> the stream TERMINATES at the corruption point (U6:
    the target cache is unverifiable, so continuing — even with plain
    single-token decoding — would generate from possibly corrupted state) +
    ``on_cache_corruption`` (caller invalidates the session cache; its
    engine terminal frame reports finish_reason="error" and the client's
    retry cold-fills). The
    ``finally`` settles the cache to exactly prompt + yielded - 1 tokens,
    then FINALIZES: commits the pending head pair AND forwards the pending
    (final/stop) token through the target — mirroring plain mlx-lm's
    lookahead ``generate_step`` — so the persisted cache contains every
    yielded token and lands at head_offset == target_offset - 1 (the lazy
    last slot). ``on_finalize(final_token, final_hidden)`` hands the caller
    the hidden needed for next turn's lazy commit (None/None on a broken or
    empty stream -> next turn cold-fills).
    """
    if prompt_cache is None or n_target_layers <= 0:
        raise ValueError("qwen_mtp_generate_step requires an explicit prompt_cache")
    target_cache = prompt_cache[:n_target_layers]
    mtp_cache = prompt_cache[n_target_layers:]
    if not mtp_cache:
        raise ValueError("qwen_mtp_generate_step: head cache entries missing")
    if ops is None:
        ops = QwenMTPOps(model, head)
    sampler = sampler or (lambda x: mx.argmax(x, axis=-1))
    block_size = max(1, int(block_size))

    arrays_idx, kv_idx, other_idx = classify_cache(target_cache)
    if other_idx:
        raise ValueError(
            f"qwen_mtp_generate_step: unsupported cache type at target layers "
            f"{other_idx[:8]} (need ArraysCache/KVCache only — no kv_bits)"
        )
    if not kv_idx:
        raise ValueError("qwen_mtp_generate_step: no offset-bearing target layer")
    _h_arrays, _h_kv, _h_other = classify_cache(mtp_cache)
    if _h_arrays or _h_other:
        raise ValueError("qwen_mtp_generate_step: head cache must be KVCache-like")

    def _t_off() -> int:
        return int(target_cache[kv_idx[0]].offset)

    def _h_off() -> int:
        return int(mtp_cache[0].offset)

    def _fire_corruption():
        if on_cache_corruption is not None:
            try:
                on_cache_corruption()
            except Exception:  # noqa: BLE001 — never break the stream
                logger.exception("[QwenMTP] on_cache_corruption callback failed")

    y = prompt.astype(mx.uint32)
    if int(y.size) < 1:
        raise ValueError("qwen_mtp_generate_step: empty prompt suffix")
    T0 = _t_off()
    # Entry invariant (callers validate first — defensive). Two legal shapes:
    #   * fresh cache: head == target == 0;
    #   * resumed finalized cache: head == target - 1 — the head's LAST slot
    #     was left uncommitted at finalize (its pair token is THIS suffix's
    #     first token); commit the lazy pair (resume_hidden, suffix[0]) now.
    #
    # U6/F6: this initialization/resume validation runs BEFORE the main
    # try below, so a raise here would ESCAPE the generator instead of
    # taking the corruption-terminal path (the engine would see an
    # exception, not the "error" terminal frame, and invalidation could
    # double-fire). Extend the corruption-handling boundary over it: any
    # MTPCacheCorruption here TERMINATES the (empty) stream with exactly
    # one on_cache_corruption invalidation — same broken/terminate
    # semantics as the mid-stream handlers.
    try:
        if T0 > 0 and _h_off() == T0 - 1:
            if resume_hidden is None:
                # No mutation yet — a caller-contract violation, but the
                # session cache still cannot be trusted for MTP resume:
                # terminate + invalidate (cold-fill) rather than raise.
                raise MTPCacheCorruption(
                    f"resumed cache (head {T0 - 1} == target {T0} - 1) without "
                    f"resume_hidden — the lazy last-slot pair cannot be committed"
                )
            with mx.stream(generation_stream):
                ops.head_hidden(resume_hidden, y[:1][None], mtp_cache)
            if _h_off() != T0:
                # Head mutated but did not land — nothing downstream can
                # trust it.
                raise MTPCacheCorruption(
                    f"lazy last-slot commit shortfall: head {_h_off()} != {T0}"
                )
        elif _h_off() != T0 or T0 != 0:
            raise MTPCacheCorruption(
                f"entry invariant violated: head {_h_off()} / target {T0} "
                f"(want fresh 0/0, or finalized head == target - 1)"
            )
    except MTPCacheCorruption as e:
        logger.error(
            "[QwenMTP] FAIL-CLOSED (resume/entry validation): %s — "
            "TERMINATING stream, invalidating session cache",
            e,
        )
        _fire_corruption()
        # No token was yielded and the main try/finally was never entered:
        # returning here ends the stream with ZERO frames; the engine's
        # corruption flag (set by the callback) drives its "error" terminal
        # and the next turn cold-fills. on_finalize is intentionally not
        # called — the corruption callback already cleared the resume stash.
        return

    hist: Optional[mx.array] = y if logits_processors else None

    def _process_and_sample(toks, logits):
        if logits_processors:
            for p in logits_processors:
                logits = p(toks, logits)
        lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        return sampler(lp), lp

    def _snapshot():
        # ArraysCache entries are replaced functionally by GatedDeltaNet
        # (never mutated in place) -> reference copies are exact + ~free.
        return (
            [(i, list(target_cache[i].cache)) for i in arrays_idx],
            [(i, int(target_cache[i].offset)) for i in kv_idx],
        )

    def _check_kv_offsets(expected: int, where: str):
        bad = [
            (i, int(target_cache[i].offset))
            for i in kv_idx
            if int(target_cache[i].offset) != expected
        ]
        if bad:
            raise MTPCacheCorruption(
                f"{where}: target KV layers not at offset {expected}: {bad[:8]}"
            )

    def _restore(snap):
        arrs, offs = snap
        for i, st in arrs:
            target_cache[i].cache = list(st)
        bad = []
        for i, off in offs:
            c = target_cache[i]
            delta = int(c.offset) - off
            if delta > 0:
                c.trim(delta)
            if int(c.offset) != off:
                bad.append((i, off, int(c.offset)))
        if bad:
            raise MTPCacheCorruption(f"restore shortfall (per-layer): {bad[:8]}")

    def _head_trim_to(expected: int):
        spec = _h_off() - expected
        if spec < 0:
            raise MTPCacheCorruption(
                f"head cache behind round base: {_h_off()} < {expected}"
            )
        if spec > 0:
            for c in mtp_cache:
                c.trim(spec)
        if _h_off() != expected:
            raise MTPCacheCorruption(
                f"head trim shortfall: at {_h_off()}, expected {expected}"
            )

    def _settle_round(ctx) -> Tuple[mx.array, int]:
        """Reconcile the caches with the CONSUMED token count of a round.
        Returns (hidden_after_last_forwarded_token, pending_token)."""
        consumed = ctx["consumed"]
        snap, drafts, emit = ctx["snap"], ctx["drafts"], ctx["emit"]
        a, k, T, b = ctx["a"], ctx["k"], ctx["T"], ctx["b"]
        if consumed <= 0:
            # Round computed but nothing delivered: rewind everything.
            _restore(snap)
            _head_trim_to(T)
            return ctx["prev_hid"], b
        if a == k and consumed == k + 1:
            real_h = ctx["vhid"]  # verify left target exactly at T + consumed
        else:
            _restore(snap)
            replay = [b] + drafts[: consumed - 1]
            with mx.stream(generation_stream):
                real_h = ops.target_hidden(
                    mx.array([replay], mx.uint32), target_cache
                )
            _check_kv_offsets(T + consumed, "replay")
        _head_trim_to(T)
        n_commit = consumed - 1
        if n_commit > 0:
            with mx.stream(generation_stream):
                ops.head_hidden(
                    real_h[:, :n_commit, :],
                    mx.array([drafts[:n_commit]], mx.uint32),
                    mtp_cache,
                )
        if _h_off() != T + n_commit:
            raise MTPCacheCorruption(
                f"head recommit shortfall: at {_h_off()}, expected {T + n_commit}"
            )
        # Round invariant: head trails target by exactly the pending token.
        if _h_off() != _t_off() - 1:
            raise MTPCacheCorruption(
                f"round invariant violated: head {_h_off()} != target {_t_off()} - 1"
            )
        return real_h[:, consumed - 1 : consumed, :], int(emit[consumed - 1])

    stats = {"rounds": 0, "drafted": 0, "accepted": 0, "emitted": 0}
    broken = False  # fail-closed: plain decode for the rest of the stream
    fatal = False  # unexpected exception: skip finally settle/finalize
    round_ctx: Optional[dict] = None
    cur_hid: Optional[mx.array] = None  # hidden after the last FORWARDED token
    cur_b: Optional[int] = None  # pending token (yielded, not yet forwarded)

    try:
        # ---- prefill: target consumes suffix[:-1]; head consumes the SAME
        # hiddens paired with suffix[1:] (slot i = (h_i, y_{i+1})) ----
        rem = y
        with mx.stream(generation_stream):
            while int(rem.size) > 1:
                n = min(prefill_step_size, int(rem.size) - 1)
                chunk = rem[:n]
                hid_chunk = ops.target_hidden(chunk[None], target_cache)
                ops.head_hidden(hid_chunk, rem[1 : n + 1][None], mtp_cache)
                _eval_cache_states(prompt_cache)
                rem = rem[n:]
                mx.clear_cache()
            # ---- bootstrap: forward the held-out last suffix token ----
            hid_last = ops.target_hidden(rem[None], target_cache)
            logits0 = ops.target_logits(hid_last[:, -1:, :])
        tok0, lp0 = _process_and_sample(hist, logits0[:, 0, :])
        mx.eval(tok0)
        b = int(tok0.item())
        cur_hid, cur_b = hid_last[:, -1:, :], b
        if hist is not None:
            hist = mx.concatenate([hist, mx.array([b], mx.uint32)])
        ntoks = 1
        stats["emitted"] += 1
        yield b, lp0, False

        while max_tokens < 0 or ntoks < max_tokens:
            remaining = (max_tokens - ntoks) if max_tokens >= 0 else (block_size + 1)
            k = 0 if broken else min(block_size, max(0, remaining - 1))

            if k <= 0:
                # Plain single-token step (budget tail, or fail-closed mode).
                with mx.stream(generation_stream):
                    if not broken:
                        # Keep the head in lockstep: commit the pending pair.
                        ops.head_hidden(
                            cur_hid, mx.array([[cur_b]], mx.uint32), mtp_cache
                        )
                    hid_last = ops.target_hidden(
                        mx.array([[cur_b]], mx.uint32), target_cache
                    )
                    logits1 = ops.target_logits(hid_last[:, -1:, :])
                tok, lp = _process_and_sample(hist, logits1[:, 0, :])
                mx.eval(tok)
                t = int(tok.item())
                cur_hid, cur_b = hid_last[:, -1:, :], t
                if hist is not None:
                    hist = mx.concatenate([hist, mx.array([t], mx.uint32)])
                ntoks += 1
                stats["emitted"] += 1
                yield t, lp, False
                continue

            try:
                # ---- draft k tokens recursively (greedy head logits;
                # step 1 commits the pending REAL pair, steps 2..k write
                # surrogate-hidden slots that are trimmed at settle) ----
                T = _t_off()
                if _h_off() != T - 1:
                    raise MTPCacheCorruption(
                        f"pre-draft invariant: head {_h_off()} != target {T} - 1"
                    )
                drafts: List[int] = []
                with mx.stream(generation_stream):
                    h_cur, nxt = cur_hid, cur_b
                    for _ in range(k):
                        x = ops.head_hidden(
                            h_cur, mx.array([[nxt]], mx.uint32), mtp_cache
                        )
                        dlog = ops.head_logits(x[:, -1:, :])
                        d = int(mx.argmax(dlog[:, -1, :], axis=-1).item())
                        drafts.append(d)
                        h_cur, nxt = x[:, -1:, :], d
                    # ---- snapshot + single verify forward ----
                    snap = _snapshot()
                    vhid = ops.target_hidden(
                        mx.array([[cur_b] + drafts], mx.uint32), target_cache
                    )
                    vlogits = ops.target_logits(vhid)  # (1, k+1, V)
                _check_kv_offsets(T + 1 + k, "verify")
            except MTPCacheCorruption as e:
                # U6: TERMINATE the stream — do NOT fall back to plain
                # decoding. The target cache is in an unverifiable state
                # (possibly advanced/rolled-back mid-round), so every token
                # generated from it after this point would be untrustworthy.
                # ``broken`` makes the ``finally`` skip settle/finalize and
                # report on_finalize(None, None); the corruption callback has
                # already invalidated the session cache, so the engine ends
                # the stream with an error terminal and the next turn takes
                # an honest MISS (client retries).
                logger.error(
                    "[QwenMTP] FAIL-CLOSED (draft/verify): %s — TERMINATING "
                    "stream, invalidating session cache",
                    e,
                )
                broken = True
                _fire_corruption()
                return

            # ---- acceptance walk (PLD FIX-4: per-position processors with
            # early stop; accept rule = sampled-token equality) ----
            if logits_processors:
                tokens_l: List[int] = []
                lp_rows: list = []
                for i in range(k + 1):
                    yi, lpi = _process_and_sample(hist, vlogits[:, i, :])
                    mx.eval(yi)
                    ti = int(yi.item())
                    tokens_l.append(ti)
                    lp_rows.append(lpi[0])
                    if i >= k or ti != drafts[i]:
                        break  # bonus position, or first rejected draft
                    hist = mx.concatenate([hist, mx.array([ti], mx.uint32)])
            else:
                toks, lp_all = _process_and_sample(None, vlogits[0])
                mx.eval(toks)
                tokens_l = [int(t) for t in toks.tolist()]
                lp_rows = list(lp_all)

            a = 0
            while a < k and a < len(tokens_l) and tokens_l[a] == drafts[a]:
                a += 1
            emit = drafts[:a] + [tokens_l[a]]
            stats["rounds"] += 1
            stats["drafted"] += k
            stats["accepted"] += a

            round_ctx = {
                "snap": snap,
                "prev_hid": cur_hid,
                "b": cur_b,
                "drafts": drafts,
                "emit": emit,
                "vhid": vhid,
                "a": a,
                "k": k,
                "T": T,
                "consumed": 0,
            }
            for j, t in enumerate(emit):
                if max_tokens >= 0 and ntoks >= max_tokens:
                    break
                if j == a and hist is not None:
                    hist = mx.concatenate([hist, mx.array([t], mx.uint32)])
                ntoks += 1
                stats["emitted"] += 1
                # Count-before-yield (PLD FIX-5): a GeneratorExit at this
                # yield must settle WITH this already-delivered token.
                round_ctx["consumed"] += 1
                yield int(t), lp_rows[j], (j < a)

            try:
                cur_hid, cur_b = _settle_round(round_ctx)
                round_ctx = None
            except MTPCacheCorruption as e:
                # U6: TERMINATE (see the draft/verify handler). The tokens
                # already yielded this round were computed by a verified
                # target forward from the pre-round state and remain
                # trustworthy; it is the CACHE that failed to settle, so
                # continuing to generate from it — even plain — would emit
                # tokens conditioned on corrupted state.
                logger.error(
                    "[QwenMTP] FAIL-CLOSED (settle): %s — TERMINATING "
                    "stream, invalidating session cache",
                    e,
                )
                round_ctx = None
                broken = True
                _fire_corruption()
                return
            if stats["rounds"] % 64 == 0:
                mx.clear_cache()
    except GeneratorExit:
        raise  # normal early close (EOS/cancel) — finally settles exactly
    except BaseException:
        # Unexpected failure mid-mutation: the cache cannot be trusted.
        fatal = True
        _fire_corruption()
        raise
    finally:
        if not fatal:
            try:
                if round_ctx is not None and not broken:
                    cur_hid, cur_b = _settle_round(round_ctx)
                    round_ctx = None
            except Exception:  # noqa: BLE001 — includes MTPCacheCorruption
                logger.exception("[QwenMTP] finally-settle failed")
                broken = True
                _fire_corruption()
            try:
                if not broken and cur_hid is not None and cur_b is not None:
                    # Finalize: (1) commit the pending pair into the head,
                    # (2) forward the pending (final/stop) token through the
                    # TARGET — mirroring plain mlx-lm's lookahead
                    # generate_step — so the persisted cache + the engine's
                    # stored token_ids INCLUDE every yielded token and
                    # target_offset == len(stored_ids) means the same thing
                    # on the MTP and non-MTP paths (the chat-template suffix
                    # builder relies on stored ids ending WITH the stop
                    # token; next turn's suffix starts with "\n"-style glue,
                    # never the stop). The head lands one behind: its last
                    # slot pairs with NEXT turn's first token and is
                    # committed lazily at resume from the hidden surfaced
                    # via ``on_finalize(final_token, final_hidden)``.
                    with mx.stream(generation_stream):
                        ops.head_hidden(
                            cur_hid, mx.array([[cur_b]], mx.uint32), mtp_cache
                        )
                        hid_final = ops.target_hidden(
                            mx.array([[cur_b]], mx.uint32), target_cache
                        )
                    if _h_off() != _t_off() - 1:
                        raise MTPCacheCorruption(
                            f"finalize: head {_h_off()} != target {_t_off()} - 1"
                        )
                    hid_final = hid_final[:, -1:, :]
                    _eval_cache_states(prompt_cache)
                    mx.eval(hid_final)
                    if on_finalize is not None:
                        on_finalize(int(cur_b), hid_final)
                elif on_finalize is not None:
                    on_finalize(None, None)
            except Exception:  # noqa: BLE001
                logger.exception("[QwenMTP] finalize failed")
                _fire_corruption()
                if on_finalize is not None:
                    try:
                        on_finalize(None, None)
                    except Exception:  # noqa: BLE001
                        pass
        if stats["rounds"] > 0:
            logger.info(
                "[QwenMTP] rounds=%d drafted=%d accepted=%d "
                "mean_accept=%.2f/%d emitted=%d%s",
                stats["rounds"],
                stats["drafted"],
                stats["accepted"],
                stats["accepted"] / stats["rounds"],
                block_size,
                stats["emitted"],
                " (FAIL-CLOSED: speculation was disabled mid-stream)"
                if broken
                else "",
            )
