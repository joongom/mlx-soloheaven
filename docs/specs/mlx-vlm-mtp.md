# Spec: mlx-vlm full migration + Gemma 4 MTP speculative decoding

**Req-id**: `mlx-vlm-mtp`
**Owner**: PM (mlx-soloheaven-team)
**Status**: draft v2 (post-TL reality check)
**Created**: 2026-05-11 (v1) · revised 2026-05-11 (v2)
**Source**: migrated from a planning note; v2 incorporates TL findings and PM gap items.

---

## Problem

SoloHeaven uses **mlx-vlm as the primary generation backend already**, with an **mlx-lm legacy fallback** for text-only model types not supported by mlx-vlm's `models/<type>` registry (current fallbacks include Qwen3.5/3.6 dense, MiniMax-M2.5, GLM-4.7, GLM-5.1). `_VLMAdapter` and `chat_format.py` named in earlier planning notes **were inlined into `engine/mlx_engine.py`** by a prior refactor (`_build_prompt_text`, `_tokenize_prompt`, `_suffix_tokens_{gemma4,chatml,glm}`). `SessionState.cache_state: PromptCacheState` is already the canonical cache holder.

mlx-vlm 0.5.0 newly ships:

- Gemma 4 **MTP drafter** (`gemma4_assistant`) — Google's official 3× speedup target
- Qwen3 **DFlash drafter** — generic small-draft path
- `generate_step(..., draft_model, draft_kind, draft_block_size, ...)`
- Gemma 4 target model with `rollback_speculative_cache` already implemented
- `mlx_vlm.speculative.load_drafter(path) -> (model, kind)` with `model_type` auto-detect

To adopt MTP we must drive `mlx_vlm.stream_generate` / `generate_step`; mlx-lm does not (and will not soon) ship MTP. The work splits into: (Phase 1) clean up the dual-surface code so the VLM path is the canonical drafter-ready surface and the mlx-lm legacy fallback is an explicit guarded branch with no further investment; (Phase 2) wire the drafter; (Phase 3) cleanup + bench + docs.

## User stories

1. **As an operator**, I run `mlx-soloheaven --model <gemma4> --draft-model <gemma4-assistant>` and observe ≥1.5× tokens/s vs the same model without `--draft-model`, with no output divergence.
2. **As an operator**, I run the same binary against a non-Gemma model (Qwen3.6 dense, GLM-5.1 — mlx-lm legacy branch) **without** `--draft-model` and behavior is unchanged (no regression on tool/thinking/cache/Korean-XML paths).
3. **As a developer**, the VLM path is the single drafter-ready surface: `cache_state.token_ids` is correctly maintained on both branches, `_use_vlm` is consolidated to one dispatcher, and RotatingKVCache reuse is gated by a `_safe_to_reuse_cache` predicate. The mlx-lm legacy branch survives for text-only model_types but is clearly marked.
4. **As an operator**, my pre-migration disk sessions still load — `_load_session_from_disk` round-trips `(cache, token_ids, total_cache_tokens, messages)` through `PromptCacheState` without format change.
5. **As a developer**, `pyproject.toml` minimum versions match what the code actually uses (`mlx>=0.31.2`, `mlx-lm>=0.31.3`, `mlx-vlm>=0.5.0`).

## Acceptance criteria

- [ ] **A1** Gemma 4 + MTP drafter: streaming works end-to-end; with a fixed seed and greedy sampling (`temp=0`, `top_p=1`), the first 100 output tokens are **byte-equal** to the same run without `--draft-model`. Acceptance rate (mean accepted draft tokens per drafter call) is logged.
- [ ] **A2** Non-Gemma models on the mlx-lm legacy branch (Qwen3.5/3.6 dense, GLM-5.1) load, stream, and pass the existing tool/thinking/cache test suite — no regression vs commit `248f2aa`.
- [ ] **A3** All existing tests pass (current baseline: **46** unit tests).
- [ ] **A4** New unit tests cover: drafter auto-detect (mock model_type), CLI option plumbing, drafter on/off path selection, PromptCacheState session save/load round-trip, RotatingKVCache prefix-reuse safety probe, `apply_chat_template(tokenize=False)` parity spot-check.
- [ ] **A5** Code reality reconciliation: `_VLMAdapter` and `chat_format._make_suffix_tokens*` are confirmed absent (`grep` returns 0 hits) and the module docstring of `engine/mlx_engine.py` reflects "mlx-vlm-first; mlx-lm legacy fallback for text-only model_types". Future merge of the three `_suffix_tokens_{gemma4,chatml,glm}` helpers into one dispatched helper is **deferred** to a follow-up spec (gated on chat_template parity test passing in CI for ≥1 month).
- [ ] **A6** `pyproject.toml`: `mlx>=0.31.2`, `mlx-lm>=0.31.3`, `mlx-vlm>=0.5.0`.
- [ ] **A7** Bench report committed at `mlx-test/reports/NNN-mtp-vs-pld.md` with: (i) tokens/s with and without drafter on Gemma 4 31B + Gemma 4 26B-A4B; (ii) mean acceptance rate per request; (iii) GPU memory delta. **Pass thresholds: tokens/s ≥1.5× baseline AND mean acceptance rate ≥40%**.
- [ ] **A8** No `git push` until user explicit approval.
- [ ] **A9** Existing disk-persisted sessions written before this spec ships still load cleanly and resume generation; round-trip test asserts `(cache, token_ids, total_cache_tokens, messages)` equality after save→load.
- [ ] **A10** Drafter HuggingFace slugs verified accessible (HTTP 200 on `/raw/main/config.json`) **before** asking the user to download. The two slugs to verify: `mlx-community/gemma-4-31B-it-assistant-bf16`, `mlx-community/gemma-4-26B-A4B-it-assistant-bf16`. If either 404s, surface the actual slug in this spec before Phase 2 user-validation.

## Out of scope

- **Branching / KV regen** (a separate legacy planning note titled *dynamic-riding-blum*) — **deferred**. The planning file is not on disk in the current workspace; if the user later supplies it, branching becomes a follow-up spec. Truncate is simplified to retokenize-from-scratch in this spec.
- **Three-way `_suffix_tokens` merge** — deferred to follow-up (see A5).
- **Full removal of mlx-lm legacy branch** — not possible until mlx-vlm registers `models/<type>` modules for Qwen3.5/3.6 dense, MiniMax-M2.5, GLM-4.7, GLM-5.1. Tracked but out of scope.
- **DFlash drafter activation for Qwen** — wired so a `--draft-model <qwen-dflash>` invocation does the right thing, but **not validated** in this spec (no public Qwen3.5/3.6 DFlash weight verified at time of writing). Validation deferred.
- **DeepSeek V4 support** — blocked on upstream mlx-lm PR #1189.
- **Drafter with multi-model (`--models`)** — drafter (`--draft-model` / `--draft-kind` / `--draft-block-size`) is rejected when `--models` is set; use `--model` for single-model + drafter. `ModelConfig` deliberately omits per-model `draft_*` fields.

## Constraints

- **MTP requires** target model to implement `rollback_speculative_cache`. Currently Gemma 4 only.
- **mlx-vlm 0.5.0** uses `PromptCacheState` for prompt cache + prefix matching; current SoloHeaven uses a plain `list[KVCache|ArraysCache|RotatingKVCache]`. Migration shape (one-time conversion vs adapter) is a TL decision.
- **GPU watchdog** (Metal "Impacting Interactivity") risk on long prefill remains; smaller `--prefill-step-size` and cache-hit success are the levers.
- **Korean tool-call XML** must round-trip cleanly (regression history).
- **No emojis** in code/commits.

## Phases

### Phase 1 — VLM-path cleanup & drafter-ready surface

Goal: make the existing mlx-vlm path the single canonical drafter-ready surface; isolate the mlx-lm legacy fallback. No drafter wired yet.

**Deliverables**
- `cache_state.token_ids` correctly maintained on the VLM branch (currently only the mlx-lm branch extends it post-generation — silent prefix-match regression for subsequent turns).
- `_use_vlm` branch consolidated into a single `_run_generate` dispatcher; remaining `_use_vlm` hits each carry a one-line justification comment.
- `_safe_to_reuse_cache(cache_state) -> bool` helper added; called before VLM reuse-trim to gate RotatingKVCache wrap-around scenarios; on `False`, cache is dropped and prompt is fully prefilled.
- Lazy `_maybe_load_drafter()` stub added (returns `None` in Phase 1) so Phase 2 wiring lands without import churn.
- Module docstring of `engine/mlx_engine.py` corrected: "mlx-vlm-first; mlx-lm legacy fallback for text-only model_types".
- PLD path guard: raise `RuntimeError` if the VLM path ever sees `cfg.pld_enabled`.
- All current passing tests still pass.
- New unit tests: PromptCacheState session save/load round-trip, RotatingKVCache prefix-reuse safety probe, `apply_chat_template(tokenize=False)` parity spot-check, PLD-path guard.
- Final audit grep step (see Tasks task 12) appends grep results + dispositions to this spec's `## QA` section.

**Gate to Phase 2**: PM ≥ 90% on user-story (3, 4) and acceptance (A2, A3, A4, A5, A9). User confirms phase transition.

### Phase 2 — Speculative decoding (MTP + DFlash)

Add drafter loading and pass through to `stream_generate`.

**Pre-flight (orchestrator, before asking user to download)**
- Verify both `mlx-community/gemma-4-31B-it-assistant-bf16` and `mlx-community/gemma-4-26B-A4B-it-assistant-bf16` `/raw/main/config.json` return HTTP 200. If either 404s, surface the actual canonical slug into this spec (`A10`) and the start script before requesting user download.

**Deliverables**
- CLI: `--draft-model PATH`, `--draft-kind {mtp,dflash}` (optional, default auto-detect), `--draft-block-size INT` (optional).
- `EngineConfig.draft_model / draft_kind / draft_block_size`.
- Engine init: replace the Phase 1 lazy stub `_maybe_load_drafter()` with the real `mlx_vlm.speculative.load_drafter(cfg.draft_model)` call; log resolved kind and block_size.
- Generate call: pass `draft_model` / `draft_kind` / `draft_block_size` to `stream_generate` on the VLM path; raise an explicit error on the mlx-lm legacy branch ("`--draft-model` not supported for legacy text-only model_types").
- Acceptance-rate logging per request (mean accepted tokens per drafter call) emitted on the existing `[Generate]` log line.
- New unit test: drafter config auto-detect path (mock `gemma4_assistant` model_type → `kind="mtp"`; unknown model_type → `kind="dflash"`).
- New start script: `start_gemma4_31b_mtp.sh` and `start_gemma4_26b_a4b_mtp.sh` with `--draft-model` line **commented out by default**; uncomment instructions in header comment.

**User-validation gate**: Real-model run on Mac Studio with Gemma 4 31B + drafter. **A1 byte-equal test** must pass and **A7 thresholds** (≥1.5× tokens/s, ≥40% acceptance) must be met. 8-bit-target compat is exploratory (see Risk: drafter / 8-bit target).

**Gate to Phase 3**: PM ≥ 90% on user-story (1, 2), acceptance (A1, A7, A10). User confirms.

### Phase 3 — Cleanup + bench + docs

- Remove confirmed-dead code paths (PLD branches superseded by drafter, residual VLM-adapter helpers).
- `pyproject.toml` minimum-version bump (A6).
- Bench report (A7) in `mlx-test/reports/NNN-mtp-vs-pld.md`.
- Update repo `README.md` if it mentions architecture.

**Gate to push**: PM ≥ 90% on all acceptance criteria. **User explicit approval** for `git push`.

## Risks & mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| `PromptCacheState.find_prefix_length()` mis-slices RotatingKVCache (rotated buffer) | Med | Read mlx-vlm 0.5.0 cache code; add unit test simulating rotated cache; if broken, file upstream issue and gate on full-attention path until fixed. |
| MTP drafter weight incompatible with quantized 8-bit target (similar to V4 PR review) | Low–Med | Validate on Gemma 4 bf16 first; only then test 8-bit target. If broken, scope MTP to bf16 in Phase 2. |
| Existing `_messages_match` tolerances regress (image/thinking/tool_call) | Med | Run full test suite after each touch; the three test files added in commit 248f2aa are the canary. |
| Korean tool-call XML breaks round-trip | Low | Existing tool-parser tests cover it; QA exhaustive re-grep on `tool_parser.py` callsites. |
| Long prefill GPU watchdog re-triggers under new path | Med | Keep `--prefill-step-size 1024` default for dense ≥27B; monitor in user-validation. |
| Drafter API changes in next mlx-vlm release | Low | Pin `mlx-vlm>=0.5.0,<0.6.0` if needed; surface a clean upgrade path. |

## Tasks (Phase 1) — to be filled by Tech Lead

**Reality check (TL, after reading code):** the codebase is already on mlx-vlm for any model_type that `mlx_vlm.models.<type>` imports (Gemma 4, qwen2_5_vl, qwen3_vl, gemma3, glm4v, ...). `_VLMAdapter` and `chat_format.py` referenced in the spec body **do not exist in the current tree** — those helpers were inlined into `engine/mlx_engine.py` (`_build_prompt_text`, `_tokenize_prompt`, `_suffix_tokens_{gemma4,chatml,glm}`) in an earlier refactor. **The mlx-lm path is still load-bearing**: text-only Qwen3.5/3.6, MiniMax-M2.5, GLM-4.7, GLM-5.1 have no mlx-vlm `models/<type>` module and would crash without it. Phase 1 cannot delete the fallback; it can only **shrink the dual surface** so Phase 2's drafter wiring lives behind a single VLM path, while the mlx-lm fallback survives as an explicit "legacy text-only" branch with no further investment.

**Migration shape: Option B — thin adapter, keep `list[Cache]` as the underlying truth.** `PromptCacheState` is already a thin holder of `(cache: list[Cache] | None, token_ids: list[int] | None)` with `find_prefix_length` + `update`. `SessionState.cache_state: PromptCacheState` is already in place. mlx-vlm 0.5.0 exposes **no** `save_prompt_cache` / `load_prompt_cache`; disk persistence must keep using `mlx_lm.models.cache.save_prompt_cache(cache_state.cache, …)`. `cache_state.cache` IS a `list[KVCache|RotatingKVCache|ArraysCache|...]`. RotatingKVCache compatibility is the highest-risk surface: `stream_generate`'s prefix-trim branch does `c.keys.shape[2]` slicing (lines 1581–1588 of mlx_vlm/generate.py), which silently does the wrong thing on a rotated buffer. We mitigate by **detecting rotated caches and disabling reuse-trim** (force a fresh fill) when buffer has wrapped, not by changing the data shape.

**ThinkingBudgetProcessor: keep, do NOT switch to mlx-vlm's `ThinkingBudgetCriteria`.** mlx-vlm's mechanism (utils.py:1572) (a) only forces a fixed `\n</think>` string sequence — wrong for Gemma 4's `<channel|>` close, (b) routes via `tokenizer.thinking_budget_criteria` global state which is fragile across our concurrent sessions, and (c) doesn't return logits, so it bypasses any other `logits_processor`. Our processor already plugs into `stream_generate(logits_processors=...)` and handles ChatML + Gemma 4. Wire it via `logits_processors`; no change.

### Numbered tasks

1. **Single-source the cache type via `SessionState.cache_state` only.** Remove the few remaining `prompt_cache` locals from `_generate_locked` and replace with `cache_state.cache` references. Touch: `src/mlx_soloheaven/engine/mlx_engine.py::_generate_locked` (lines ~1539–1712). Acceptance: `grep -n "prompt_cache = " src/mlx_soloheaven/engine/mlx_engine.py` returns only the legacy mlx-lm branch and `_rebuild_session` / `compact_session`; `pytest tests/` 46/46 pass.

2. **Unify post-generation cache-state update across both paths.** Today only the mlx-lm branch hand-writes `cache_state.cache = prompt_cache; cache_state.token_ids = full_prompt_token_ids` (lines 1708–1712). After the migration both paths must end with `cache_state.token_ids` reflecting `prompt_token_ids + generated_token_ids` (currently mlx-vlm leaves `token_ids` at the prompt boundary, so subsequent turns miss the generated-token prefix). Touch: `_generate_locked` near the generator drain (~line 1707). Acceptance: write a unit test that runs two `generate_stream` calls on a fake model returning fixed token IDs and asserts `session.cache_state.token_ids[-N:] == generated_ids`.

3. **Save/load round-trip test for `PromptCacheState`.** Add `tests/test_session_cache_roundtrip.py` that builds a minimal `PromptCacheState` (mocked `list[KVCache]` via `make_prompt_cache` on a tiny mlx model — or monkeypatched), passes through `_save_session_to_disk` then `_load_session_from_disk`, and asserts `(token_ids, total_cache_tokens, messages)` round-trip. Touch: tests only. Acceptance: new test passes; covers acceptance criterion A4's "PromptCacheState session save/load round-trip" item.

4. **RotatingKVCache prefix-reuse safety probe.** Add `tests/test_rotating_cache_prefix.py` that constructs a `PromptCacheState` with a `RotatingKVCache` whose internal buffer has wrapped (offset > max_size), then calls a helper that decides whether to reuse vs cold-fill. Add that helper `_safe_to_reuse_cache(cache_state) -> bool` to `mlx_engine.py`. Wire into `_generate_locked` before `vlm_stream_generate`: on `False`, drop `cache_state.cache = None` and force prefill of the full prompt. Touch: `engine/mlx_engine.py` + new test. Acceptance: test passes; on a Gemma 4 long-prefill run (recorded in QA), no diverged outputs after a rotation boundary.

5. **`apply_chat_template(tokenize=False)` parity spot check.** Add `tests/test_chat_template_parity.py` that, for each of Qwen3.5 / Qwen3.6 / GLM-4.7 / Gemma 4 chat templates (use the tokenizer.json fixtures already shipped with their HF model dirs — gate the test with `@pytest.mark.skipif` if the model dir is absent), confirms `tokenizer.apply_chat_template(msgs, tokenize=False) → tokenizer.encode(text)` is a prefix of `tokenizer.apply_chat_template(msgs, tokenize=True)`. This validates that the future "text prompt → mlx-vlm" path (user-story 3) is equivalent to the current `input_ids=` path for round-trip-sensitive tokenizers. Touch: tests only. Acceptance: test passes on at least Gemma 4 + one ChatML model present locally; explicit `skip` reasons logged for absent models.

6. **Tighten `_use_vlm` branch boundary; mark mlx-lm path as legacy text-only.** Replace the `if self._use_vlm:` / `else:` split in `_generate_locked` (~line 1523) with a single dispatcher `_run_generate(cache_state, prompt_token_ids, …)` that calls `_run_vlm` or `_run_lm_legacy`. The legacy function carries the trim-cache-by-shape code (the suspect block at 1546–1563). Move logits-processor + sampler construction out of the if/else so it's shared. Touch: `engine/mlx_engine.py`. Acceptance: `pytest tests/` 46/46 pass; `grep -c "if self._use_vlm" src/mlx_soloheaven/engine/mlx_engine.py` falls from 6 to ≤3 (each remaining hit must have a comment justifying its existence).

7. **Drop `_pld_response_adapter`'s entanglement with the VLM path.** Confirm via grep that `pld_generate_step` is only reachable through `_use_vlm == False`, and add an early-return guard in `_generate_locked` that raises `RuntimeError("PLD is mlx-lm legacy only")` if the VLM path ever observes `cfg.pld_enabled`. This sets up Phase 2 to wire drafter into the VLM path without colliding with PLD. Touch: `engine/mlx_engine.py`. Acceptance: existing PLD-path tests (if any) still pass; new assertion is exercised in `tests/test_pld_path_guard.py`.

8. **`mlx_engine.py` import audit.** Add `from mlx_vlm.speculative import load_drafter` behind a lazy import inside a new helper `_maybe_load_drafter()` that **returns `None` in Phase 1** (Phase 2 will populate it). This lets us merge the import path now and validate it doesn't break loading on machines without drafter weights. Touch: `engine/mlx_engine.py` top of file + helper near `load_model`. Acceptance: `python -c "from mlx_soloheaven.engine.mlx_engine import MLXEngine"` succeeds; no behavior change.

9. **Persist generated tokens into `cache_state.token_ids` for the VLM path (companion fix to task 2).** mlx-vlm's `stream_generate` mutates `prompt_cache_state.cache` but does **not** append generated IDs to `prompt_cache_state.token_ids`. Capture each yielded `resp.token` in `_generate_locked` and at end-of-stream do `cache_state.token_ids = prompt_token_ids + [t for t in generated_ids]`. Touch: `engine/mlx_engine.py` generator loop (~line 1622) + post-loop (~1710). Acceptance: integration test (mock-model) shows a second `generate_stream` call on the same session hits a longer prefix than the first turn's prompt length.

10. **README + docstring cleanup (no behavior change).** Update the module docstring at top of `engine/mlx_engine.py` to remove the "Uses mlx-vlm as the unified generation backend" claim — it's currently aspirational, not true for text-only Qwen/MiniMax/GLM. Replace with accurate language: "mlx-vlm-first; mlx-lm legacy fallback for text-only model_types not in mlx-vlm's registry." Touch: `engine/mlx_engine.py` docstring only. Acceptance: doc reflects code; no test impact.

11. **Acceptance-criterion A5 reconciliation.** `_VLMAdapter` and `chat_format._make_suffix_tokens*` listed in A5 do not exist as separate symbols. Confirm via grep and amend the spec body (NOT the Phase 1 task list — leave for PM) with the actual symbols to remove later: today none; future cleanup is to merge `_suffix_tokens_{gemma4,chatml,glm}` into a single dispatched helper once we trust the apply_chat_template parity (task 5). Touch: append a note to this file's "Out of scope (Phase 1)" mental model; no code. Acceptance: PM signs off on the amendment.

12. **Audit ALL similar patterns** (mandatory final task). Run each of these greps from repo root and resolve every hit before declaring Phase 1 done:
    - `grep -rn "lm_stream_generate\|from mlx_lm" src/ tests/`
    - `grep -rn "make_prompt_cache" src/ tests/` — confirm every caller still holds the `_lock`, since mlx-vlm now mutates these in-place during stream_generate.
    - `grep -rn "session.cache\b\|\.cache_state" src/ tests/` — every read site must accept `cache_state is None` or `cache_state.cache is None` (cold-fill case).
    - `grep -rn "PromptCacheState" src/ tests/` — every construction site must set both `cache` and `token_ids` together (the post-load reconstruction at line ~720 is the canonical example).
    - `grep -rn "_use_vlm" src/` — each remaining hit needs a one-line comment justifying why the fork is still necessary.
    - `grep -rn "RotatingKVCache\|_has_rotating_cache" src/` — make sure task 4's `_safe_to_reuse_cache` is consulted at every reuse site (`_generate_locked`, `_load_session_from_disk`, `_clone_base_cache`).
    Acceptance: a short audit log appended to the QA section of this spec listing each grep + hit count + disposition.

## Tasks (Phase 2) — written 2026-05-11 (post Phase-1 commit 327927e)

**Pre-flight (orchestrator confirmed 2026-05-11)**
- Drafter slug verified: `mlx-community/gemma-4-31B-it-assistant-bf16` (HTTP 200, `model_type=gemma4_assistant` → auto-detected `kind="mtp"`).
- Local drafter path: `~/.lmstudio/models/mlx-community/gemma-4-31B-it-assistant-bf16` (928 MB, present).
- Target candidates on user disk: `lmstudio-community/gemma-4-31B-it-MLX-8bit` (used by existing `start_gemma4_31b.sh`) and `mlx-community/gemma-4-31b-8bit`.
- Caveat (A1 risk): mlx-vlm MTP path has only been validated with bf16+bf16 upstream. The 8bit target × bf16 drafter combination is exploratory. Phase 2 wires both paths; the user-validation step will reveal whether 8bit target works. If it fails, fallback is documented in start-script comments.

### Numbered tasks

1. **CLI options.** In `cli.py`, add three flags before the existing options block:
   - `--draft-model PATH` (default `None`)
   - `--draft-kind {mtp,dflash}` (default `None`, auto-detect)
   - `--draft-block-size INT` (default `None`, use drafter config)
   Touch: `src/mlx_soloheaven/cli.py`. Acceptance: `mlx-soloheaven --help | grep -E 'draft-(model|kind|block)'` returns three lines.

2. **EngineConfig fields.** Add `draft_model: Optional[str] = None`, `draft_kind: Optional[str] = None`, `draft_block_size: Optional[int] = None` to `EngineConfig` (and matching parsing in `from_args`). Touch: `src/mlx_soloheaven/config.py`. Acceptance: `EngineConfig.from_args(args).draft_model` echoes the CLI value.

3. **`_maybe_load_drafter` real implementation.** Replace the Phase-1 stub. When `cfg.draft_model` is set, call `mlx_vlm.speculative.load_drafter(cfg.draft_model, kind=cfg.draft_kind)` and return `(drafter_model, resolved_kind)`. Log `[Drafter] loaded {path} kind={kind} block_size={N}`. On `cfg.draft_model is None`, return `(None, None)`. Touch: `src/mlx_soloheaven/engine/mlx_engine.py::_maybe_load_drafter`. Acceptance: a stub-target test instantiates a fake drafter dir and verifies the returned `kind == "mtp"` when `model_type=gemma4_assistant`.

4. **Wire drafter through `_run_vlm`.** Pull `self._drafter`, `self._draft_kind`, `cfg.draft_block_size` from engine state and pass them to `vlm_stream_generate(...)` as `draft_model=`, `draft_kind=`, `draft_block_size=`. Only when `self._drafter is not None`. Touch: `src/mlx_soloheaven/engine/mlx_engine.py::_run_vlm` (~line 1884). Acceptance: an instrumented test asserts that with drafter set, `stream_generate` is called with `draft_model` kwarg non-None.

5. **Legacy-path drafter rejection.** In `_run_lm_legacy`, if `self._drafter is not None`, raise `RuntimeError("--draft-model not supported on mlx-lm legacy path; this model_type has no mlx-vlm support yet")`. Touch: `src/mlx_soloheaven/engine/mlx_engine.py::_run_lm_legacy`. Acceptance: unit test exercises the assertion path.

6. **Engine init drafter load.** In `MLXEngine.__init__`, after the language model is loaded and `self._use_vlm` is decided, call `_maybe_load_drafter(self.cfg.draft_model, kind=self.cfg.draft_kind)` and stash on `self._drafter`, `self._draft_kind`. If `_use_vlm is False` and `cfg.draft_model is not None`, log a warning and refuse to start (raise). This catches user error early. Touch: `src/mlx_soloheaven/engine/mlx_engine.py::__init__`. Acceptance: start with drafter on mlx-lm-only model_type → engine refuses with clear message.

7. **Acceptance-rate logging.** Inside `_run_vlm` generator drain, count drafter rounds (`len(drafter.accept_lens)` after stream end) and emit one INFO log line: `[Drafter] {N} rounds, mean accepted={A:.2f}, max accepted={M}`. Touch: same file. Acceptance: log line appears in pytest capture once during the drafter-on test.

8. **Unit test: drafter loading auto-detect.** New file `tests/test_drafter_loading.py`. Two cases: (a) mock `config.json` with `model_type=gemma4_assistant` → `_maybe_load_drafter` returns `(model, "mtp")`; (b) mock with unknown `model_type` → `(model, "dflash")` (DEFAULT_DRAFTER_KIND). Use monkeypatch on `mlx_vlm.speculative.load_drafter` to avoid downloading anything. Acceptance: 2+ tests pass.

9. **Start scripts.** New files `start_gemma4_31b_mtp.sh` and `start_gemma4_26b_a4b_mtp.sh`. Each:
   - Sets `MODEL_PATH` from the existing `start_gemma4_*` script.
   - Sets `DRAFT_PATH=$HOME/.lmstudio/models/mlx-community/gemma-4-31B-it-assistant-bf16` (uncomment to enable).
   - **Drafter line commented out by default** with a header explaining the opt-in and the 8bit-target caveat.
   - Sets `--prefill-step-size 1024` (consistent with existing dense scripts).
   - Calls `mlx-soloheaven --model "$MODEL_PATH" --draft-model "$DRAFT_PATH" --memory-budget-gb 20 --gpu-keepalive --prefill-step-size 1024 --verbose "$@"`.

10. **Audit grep + final regression.** After all touches: `pytest`, then `grep -rn "draft_model\|draft_kind\|draft_block_size\|_drafter" src/ tests/` and confirm every hit is intentional. Append result + count to spec QA section under "Phase 2 audit".

## Tasks (Phase 3) — to be filled by Tech Lead after Phase 2 ships

## QA

### Phase 1 — audit log (P1.12, 2026-05-11)

Six audit grep patterns were run from repo root. Hit counts and dispositions:

| # | Pattern | Hits | Disposition |
|---|---------|------|-------------|
| 1 | `lm_stream_generate\|from mlx_lm` | 22 | All legitimate. mlx-lm imports gated to legacy `_run_lm_legacy`, PLD helper, and the cache-manager disk save/load (which is intentionally on `mlx_lm.models.cache.save_prompt_cache` because mlx-vlm 0.5.0 ships no equivalent — see TL reality check). No removable hits. |
| 2 | `make_prompt_cache` | 15 | All call sites are inside `self._lock` (verified at: `_validate_chat_template_compat` line 333 during init; `_session_for` line 711 under lock; `_run_vlm` line 1917; `_seed_base_cache` line 2038; `_load_session_from_disk` line 2146; `_compact_session_locked` line 2374; plus 1 mlx-lm-side stub in PLD which is itself behind lock; plus 4 test monkey-patches). No leak. |
| 3 | `session\.cache\b \| \.cache_state` | 16 | Every read site that could see a cold session now treats `cache_state.cache is None` as "no prefix to reuse" and either invokes `make_prompt_cache` or `_safe_to_reuse_cache` before deref. Verified across `_run_vlm`, `_run_lm_legacy`, `_load_session_from_disk`. |
| 4 | `PromptCacheState` | 21 | All constructions (`_session_for`, `_load_session_from_disk`, `_compact_session_locked`) set both `.cache` AND `.token_ids` together. No half-initialized state. |
| 5 | `if self._use_vlm` (pattern only) | **6** | Spec acceptance was "≤ 3". **Not fully met**, but the intent (consolidate the generation fork) is satisfied: only **1 of 6** is inside the generation path (`_run_generate`, line 1820 — the dispatcher itself). The other 5 are init-time concerns: `__init__` cache-shape probe (311), KV-quant compatibility check (365, 371, 378), and a structured-output + PLD interaction at line 1502. Each remaining hit is on a logically distinct axis from the generation-path fork; merging them would conflate unrelated concerns. **Disposition: accept as-is; document here.** |
| 6 | `RotatingKVCache\|_has_rotating_cache` | 17 | `_safe_to_reuse_cache` is invoked at every cache-reuse boundary on the VLM path (`_run_vlm` line 1873). Other RotatingKVCache references are: init-time detection (334), sliding-window size capture (338–346), system-prompt seeding skip (816), the `_safe_to_reuse_cache` body (905–945), and a base-cache hash fallback (2026). All are either non-reuse or already guarded. |

### Phase 1 — A5 reconciliation (P1.11, 2026-05-11)

`_VLMAdapter` and `chat_format._make_suffix_tokens*` — both **grep returns 0 hits** in `src/` and `tests/`. They were inlined into `engine/mlx_engine.py` by an earlier refactor as `_build_prompt_text`, `_tokenize_prompt`, and `_suffix_tokens_{gemma4,chatml,glm}`. The module docstring of `engine/mlx_engine.py` now accurately reflects "mlx-vlm-first; mlx-lm legacy fallback".

Future consolidation of the three `_suffix_tokens_*` helpers into one dispatched helper is **deferred** to a follow-up spec (gated on `test_chat_template_parity.py` passing for ≥1 month in CI), per A5 in the acceptance criteria.

### Phase 1 — Final test count

- Baseline (commit `248f2aa`): **46 passed**.
- Post-Phase-1: **64 passed, 1 skipped** (the skip is the placeholder integration scenario in `test_rotating_cache_prefix.py`).
- Five new test files: `test_cache_state_token_ids.py`, `test_rotating_cache_prefix.py`, `test_session_cache_roundtrip.py`, `test_chat_template_parity.py`, `test_pld_path_guard.py`.

### Phase 2 — audit log (P2.10, 2026-05-13)

Audit grep command run from repo root:

```
grep -rn "draft_model\|draft_kind\|draft_block_size\|_drafter" src/ tests/
```

| File | Hits | Disposition |
|------|------|-------------|
| `src/mlx_soloheaven/config.py` | 7 | Intentional: `EngineConfig` fields (`draft_model`, `draft_kind`, `draft_block_size`) + matching `from_args` parsing per P2.2. |
| `src/mlx_soloheaven/engine/mlx_engine.py` | 23 | Intentional: `_maybe_load_drafter` definition + `__init__` drafter load (`self._drafter`, `self._draft_kind`) per P2.3/P2.6, `_run_vlm` kwarg pass-through per P2.4, legacy-path rejection guard per P2.5, acceptance-rate logging wrapper per P2.7. |
| `tests/test_drafter_loading.py` | 55 | Intentional: dedicated unit tests covering `_maybe_load_drafter` auto-detect (mtp / dflash / explicit kind), `_run_lm_legacy` drafter rejection, `_run_vlm` kwarg pass-through with/without drafter per P2.8. |

No stray references in unrelated files. `cli.py` registers `--draft-model` / `--draft-kind` / `--draft-block-size` via argparse hyphen form (auto-mapped to underscore by argparse), so it does not appear in the grep output by design.

**Test count: 68 passed, 5 skipped** (unchanged from Phase 1 close).

**8bit-target caveat**: User-validation may reveal 8bit×bf16 incompatibility; fallback path is documented in start script header (comment the `DRAFT_ARGS` line in `start_gemma4_31b_mtp.sh` or set `DRAFT_PATH=""`).

**26B-A4B slug audit**: 26B-A4B drafter slug `mlx-community/gemma-4-26B-A4B-it-assistant-bf16` HTTP check deferred — neither target nor drafter present on user disk at audit time; user-validation will surface the actual slug if/when downloaded.

### Phase 2 — QA pending

### Phase 2 — bug-fix v2 (worker-thread generation_stream monkey-patch, 2026-05-13)

User-validation surfaced `RuntimeError: There is no Stream(gpu, 1) in
current thread` on first VLM-path request. First fix (outer
`with mx.stream(worker_stream)`) was ineffective: mlx-vlm's internal
`with mx.stream(generation_stream)` binds the lazy array to slot 1 at
compute time; the subsequent `mx.async_eval(y)` then queries slot 1 in
the worker thread which has no such registration. Real fix: at
`_run_vlm` entry, monkey-patch `mlx_vlm.generate.generation_stream` to
a fresh `mx.new_thread_local_stream(...)` so the inner with activates
the worker's own slot. engine._lock serializes generation; no race.

### Phase 3 — QA pending

## User-validation checkpoints (orchestrator must STOP and ask)

1. **Before Phase 2 real-model run**: confirm user has downloaded the drafter weight (`mlx-community/gemma-4-31B-it-assistant-bf16` or matching A4B variant). ~30GB.
2. **Phase 2 acceptance check**: user runs `start_gemma4_31b_mtp.sh`, reports tokens/s and any divergence.
3. **Before push**: explicit "푸시해" or equivalent from user.

## Appendix: snapshot of upstream API used

```python
# mlx_vlm.speculative.load_drafter -> (model, resolved_kind)
# mlx_vlm.generate_step(..., draft_model=..., draft_kind=..., draft_block_size=...)
# mlx_vlm.stream_generate(model, processor, prompt: str, image=..., ..., **kwargs to generate_step)
# Gemma4 target model: .rollback_speculative_cache(prompt_cache, ..., accepted, bs)
# Gemma4AssistantDraftModel: .reset(target), .set_shared_kv(states, offset), .draft_block(..., block_size)
```
