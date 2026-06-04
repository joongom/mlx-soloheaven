# Gemma 4 MTP drafter accepts ~0% after RotatingKVCache wrap

**Affected**: `mlx-vlm == 0.5.0`, Gemma 4 family with sliding-window attention
(model_type `gemma4`/`gemma4_text`), MTP drafter (`gemma4_assistant`).

**Symptom**: Drafter mean acceptance collapses once any `RotatingKVCache`
layer wraps (`offset > max_size`). Observed `mean_accepted = 0.26` over
958 rounds on Gemma 4 31B 8bit + bf16 drafter + `sliding_window = 1024`
after a COLD-FILL refill restarted the cache. Pre-wrap acceptance on the
same workload sits at `~1.17`. Net effect: drafter becomes a throughput
liability post-wrap and stays that way for the rest of the conversation.

---

## Reproduction (qualitative)

Setup:

- Model: Gemma 4 31B 8-bit, sliding window = 1024.
- Drafter: bf16 Gemma 4 assistant, `block_size = 3`.
- Long multi-turn conversation that exceeds `1024` total tokens.
- Watch `accept_lens` per request: mean drops from ~1.17 → ~0.26 the
  first request after wrap, and stays there.

Disabling MTP eliminates the regression; the drafter weights are
healthy. The regression is in the wrap path, not the drafter.

---

## Hypothesis — three coordinated bugs

### B1. `shared_kv_states` is in ring-buffer order, not temporal order

`mlx_vlm/models/gemma4/language.py::Gemma4TextModel.__call__` (lines
540–544) populates `shared_kv_sink[layer.layer_type]` with `(K, V)`
returned from each layer's attention. For `sliding_attention` layers
those tensors come from `RotatingKVCache.update_and_fetch`, which after
wrap returns the ring buffer in **physical-slot order** — `slot[i]` is
**not** `token-at-temporal-index-i`.

The drafter then consumes this as `shared_kv` and runs
`mlx_vlm/speculative/drafters/gemma4_assistant/masks.py::build_swa_mask`,
which assumes:

```python
q_idx = mx.arange(query_offset, query_offset + query_len)[:, None]
k_idx = mx.arange(kv_len)[None, :]
dist  = q_idx - k_idx
inside = (dist > -window) & (dist < window)
```

i.e. `k_idx` is temporal. Once the ring has wrapped, slot-`i` ≠ time-`i`;
the SWA mask misidentifies which keys are inside the window and the
drafter attends to the wrong context. Logits become noisy enough that
the verifier rejects the great majority of draft tokens.

### B2. `kv_offset` is logical-cumulative, not bounded by `max_size`

`mlx_vlm/generate.py::_mtp_rounds` (line 553, again at 614):

```python
kv_offset = int(prompt_cache[0].offset)
draft_model.set_shared_kv(shared_kv_states, kv_offset)
```

`RotatingKVCache.offset` is monotonically incremented (never reset);
after wrap it exceeds `max_size`. The drafter uses this as
`query_offset` in `build_swa_mask`, which compounds B1 — the mask now
expects the query token to live "outside" the ring, so virtually all
distances are out of window.

### B3. `rollback_speculative_cache` calls `trim` without `is_trimmable` check

`mlx_vlm/models/gemma4/language.py::LanguageModel.rollback_speculative_cache`
(lines 635–636):

```python
if trim > 0 and hasattr(c, "trim"):
    c.trim(trim)
```

For `RotatingKVCache`, `is_trimmable()` returns `False` once
`offset >= max_size`, but `trim(n)` still mutates:

```python
self.offset -= n
self._idx   -= n
```

so the ring's bookkeeping drifts each MTP round (cumulative
under-counting of `offset` and `_idx`). Successive rounds then write
into wrong slots and B1 worsens.

---

## Suggested fix

### B1 — temporal-order the sink for wrapped RotatingKVCache

In `Gemma4TextModel.__call__`, when writing the sink:

```python
if shared_kv_sink is not None:
    for idx, layer in enumerate(self.layers):
        kvs, _ = intermediates[idx]
        if kvs is None:
            continue
        c = cache[idx]
        if (
            isinstance(c, RotatingKVCache)
            and c.offset > c.max_size
        ):
            K, V = kvs
            shared_kv_sink[layer.layer_type] = (
                c._temporal_order(K),
                c._temporal_order(V),
            )
        else:
            shared_kv_sink[layer.layer_type] = kvs
```

### B2 — clamp `kv_offset` to `max_size` on wrapped caches

In `_mtp_rounds` (and `_mtp_rounds_batch`):

```python
c = prompt_cache[0]
if isinstance(c, RotatingKVCache) and c.offset > c.max_size:
    kv_offset = c.max_size
else:
    kv_offset = int(c.offset)
```

### B3 — guard `c.trim` behind `is_trimmable()`

In `rollback_speculative_cache`:

```python
if trim > 0 and hasattr(c, "trim"):
    if hasattr(c, "is_trimmable") and not c.is_trimmable():
        # RotatingKVCache after wrap: trim mutates _idx/offset
        # without rolling the ring, leaving bookkeeping skewed.
        continue
    c.trim(trim)
```

---

## Workaround in our stack

`mlx_soloheaven/engine/mlx_engine.py::_install_mtp_wrap_patches` applies
B1+B2+B3 as runtime monkey-patches on the dedicated VLM worker thread
during engine init. We also keep a `_will_wrap_during_generate` safety
gate in `_run_vlm` that drops the drafter for the single request that
would cross the boundary, so even if the monkey-patch ever fails to
apply we never re-encounter the 0% acceptance plateau.

Both layers are reverted automatically once mlx-vlm ships a fix
upstream — the monkey-patch is idempotent and gated on
`type(c).__name__ == "RotatingKVCache"`, so a future redesign of the
cache layout will simply make the patch a no-op.
