# DeepSeek-V4 네이티브 디코드 런타임 — 디버깅 로그

이 문서는 native Metal 재생 런타임(`src/mlx_soloheaven/native/`)을 개발하며
겪은 **간헐적(intermittent) 커널 버그** 한 건을 추적·해결한 과정을 남긴다.
관련 문서: [`../../src/mlx_soloheaven/native/README.md`](../../src/mlx_soloheaven/native/README.md)
(사다리/설계), [`deepseek-v4.md`](deepseek-v4.md)(측정 대장).

핵심 요약(바쁜 사람용):

* **진짜 버그**: `dsv4_comp_step`가 읽는 compressor 상태 버퍼를 `ratio*cd`로
  할당했는데 커널은 `rows*cd = coff*ratio*cd`를 인덱싱한다. `coff==2`인
  ratio-4 레이어에서 상태가 **절반 크기**라 매 토큰 한 그룹씩 OOB로 인접
  버퍼를 읽고/썼다. 실모델 43개 레이어 중 **21개(ratio-4)**가 영향 —
  테스트 아티팩트가 아니라 **프로덕션 정확성 버그**였다.
* **결정적 도구**: `MTL_SHADER_VALIDATION=1`. 추측 대신 정확한 OOB 접근
  (커널명·오프셋·버퍼 길이·소스 라인)을 찍어줬다.
* **남은 플레이크**의 정체: native 디코더는 **완전 결정론적·정확**하고,
  full-model 통합 테스트가 비교하던 **`mx.compile` 참조 경로가 스위트 압력
  하에서 비결정적**이었다. → 테스트를 참조 자기일관성 가드로 견고화.

---

## 1. 배경 / 증상

두 테스트가 **스위트 전체 실행에서만(in-suite)** 간헐 실패했다. 단독 실행은
항상 통과:

* `test_native_ratio4_attention_plan_matches_reference` — ratio-4(indexer)
  어텐션 한 레이어 plan을 eager `Attention.decode_step_math`와 diff.
* `test_native_decoder_full_model_matches_reference` — 3-레이어(dense +
  ratio-128 + ratio-4) 전체 디코드를 `NativeDecoder`로 재생.

실패율은 in-suite 기준 **약 40%**. 손상은 **이중모드(bimodal)**였다 —
출력이 정확히 맞거나(diff `0.0`), 특정 **결정적 오답**이거나. attn_core 직접
출력 `acore`의 abs-sum을 찍어보면:

```
acoreABS=188.760   # 정답 (매번 동일)
acoreABS=149.749   # 오답
acoreABS=122.386   # 다른 오답
acoreABS=nan       # 가끔 NaN
```

512개 원소 중 ~280개가 틀리고, 값 자체는 재현될 때마다 동일(랜덤 쓰레기가
아님). **직전에 실행된 테스트가 남긴 GPU 상태**에 의존했다.

---

## 2. 가설과 반증 (막다른 길들)

반증 과정 자체가 자산이므로 각 가설과 **어떻게 배제했는지**를 남긴다.

### 2.1 배리어 / 실행 순서 — 배제

MLX 버퍼는 hazard-untracked라 dependent dispatch 사이에 명시적
`memoryBarrierWithScope:MTLBarrierScopeBuffers`가 필요하다. 배리어 누락을
의심했으나:

* dispatch **하나당 command buffer 하나 + `waitUntilCompleted`**로 완전
  직렬화해도 손상이 **그대로** 재현됐다 → 순서/배리어 문제가 아니다.
* 반대로 `computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent`로
  바꾸니 손상이 **더 심해졌다**(배리어는 다 있는데도) → 배리어가 hazard를
  못 막는 게 아니라 **hazard 자체가 다른 곳**에 있다는 신호.

### 2.2 indexer top-k `cidx` stale read — 배제

attn_core가 `cidx`(선택된 압축 그룹 인덱스)를 top-k 커널이 쓰기 전에 읽어
scratch 초기값(0)을 본다는 가설. scratch `cidx` 초기값을 `0` → `-1`로 바꿔도
손상 불변 → `cidx` 아님. (읽어보면 `cidx=[-1,-1,-1,-1]`로 항상 올바르게 아무
그룹도 선택 안 함.)

### 2.3 attn_core threadgroup 미초기화 — 배제

`qh/kvr/cs/sc/red` 등 threadgroup 배열을 커널 진입 시 전부 0으로 초기화해도
손상 불변 → attn_core 내부 threadgroup 미초기화 아님.

### 2.4 divergent SIMD reduction — 조사 후 배제 ("그럴듯하지만 틀린" 사례)

`for (int j = tid; j < K; j += TG) lm = max(lm, sc[j]); lm = simd_max(lm);`
에서 `K < TG`이면 스레드마다 iteration 수가 달라 simdgroup이 발산하고,
`simd_max`/`simd_sum`이 참여 안 한 레인의 정의되지 않은 값을 읽는다는 가설.
균일 trip-count(`KP = ceil(K/TG)*TG`)로 재작성해봤으나 출력이 **비트 동일**
(bit-identical)이었다. → 리덕션은 처음부터 정확했다. 이유: **누산기가 루프
전에 초기화**(`lm=-INFINITY`, `acc=0`)되므로 0-iteration 레인도 리덕션의
항등원을 들고 있다. Apple GPU는 post-dominator에서 재수렴하므로 `simd_*`는
정의된 값만 본다. 교훈: 발산 리덕션이 항상 버그는 아니다 — 누산기 초기화가
있으면 안전하다.

### 2.5 Heisenbug 서명 확보

중간 버퍼(`kvn`, `q_raw`)를 numpy로 **read-back 하자 버그가 사라졌다**. 이는
**메모리 해저드**(OOB/aliasing)의 전형적 서명 — read-back이 MLX 할당기
상태를 흔들어 "피해 버퍼"의 주소가 바뀌면서 증상이 가려진다. 즉 버그의 피해가
**버퍼 할당 순서에 의존**한다.

---

## 3. 결정적 도구: Metal Shader Validation

추측을 멈추고 Apple의 GPU 검증 계층을 켰다:

```
MTL_SHADER_VALIDATION=1 MTL_SHADER_VALIDATION_REPORT_TO_STDERR=1 python probe.py
```

정확한 OOB 접근을 찍어줬다:

```
Invalid device load at offset 4096, executing kernel function: "dsv4_comp_step"
buffer: <unnamed>, length:4096, resident:Read Write
  * frame #0: dsv4_comp_step() - /program_source:885:56

Invalid device load at offset 2048, executing kernel function:
  "affine_qmv_fast_bfloat16_t_gs_64_b_8_batch_0"
buffer: <unnamed>, length:1024   encoder: "0", dispatch: 7
```

* `offset 4096 == length 4096` → 상태 버퍼 **딱 한 그룹 초과** 읽기.
* qmv는 1024바이트 입력을 offset 2048(=2배)에서 읽기.

**주의(caveat)**: GPU validation을 켜면 MLX eager 참조 경로가 깨진다
(`RuntimeError: [Scatter::eval_gpu] Invalid number of threads`). 그래서 eager
참조 없이 **native plan만** validation 하에 돌리는 probe를 따로 만들어야 했다.

---

## 4. 근본 원인 1 — compressor 상태 under-allocation (진짜 버그)

`dsv4_comp_step`는 compressor 상태를 `[rows, cd]`로 인덱싱한다:

```
const int cd   = coff * d;        // 열: coff * head_dim
const int rows = coff * ratio;    // 행: coff * ratio
...
float kvv = ... : kv_st[src * cd + i];   // src ∈ [0, rows)
```

정답 크기는 참조 `CompressorState.reset`에 박혀 있다:

```python
shape = (batch, coff * ratio, coff * head_dim)   # = (1, rows, cd)
```

그런데 native 디코더와 테스트 scratch는 `ratio * cd`로 할당했다 — **행 수에서
`coff` 인자가 빠졌다**. ratio-128은 `coff==1`이라 우연히 맞았지만, ratio-4는
`coff==2`라 상태가 **절반 크기**. 커널이 매 토큰 `[ratio, 2*ratio)` 행을 읽어
인접 버퍼로 넘어갔고, 인접 내용은 할당 순서에 따라 달라져 in-suite에서만
쓰레기가 됐다.

실모델(`compress_ratios` 정규화 후 43 레이어)의 분포는 `{ratio-4: 21,
ratio-128: 20, dense: 2}` — **21개 레이어가 영향**받는 프로덕션 정확성 버그다.

수정: `coff * ratio * cd`로 할당.

* `src/mlx_soloheaven/native/decoder.py::_alloc_layer` (kv_a/sc_a/kv_b/sc_b,
  그리고 indexer의 ikv_*/isc_*)
* 테스트 scratch(`tests/test_dsv4_native.py`)도 동일하게.

이 수정 후 validation에서 `dsv4_comp_step` OOB가 사라졌고, in-suite 플레이크가
**~40% → ~10%**로 떨어졌다.

---

## 5. 근본 원인 2 — qmv `N % 8` (테스트 설정 아티팩트)

`affine_qmv_fast_..._batch_0`는 하드 제약이 있다: **`N % 8 == 0` 그리고
`K % 512 == 0`**. `weights_proj`는 `N = index_n_heads`인데 테스트가
`index_n_heads=2`(≠ 8의 배수)를 써서, qmv가 8행 단위로 처리하며 2행짜리
weight를 OOB로 읽었다.

**실모델은 `index_n_heads=64`(8의 배수)라 프로덕션엔 없는 문제** — 순수
테스트 설정 아티팩트. 수정: 압축-레이어 테스트들이 `index_n_heads=8` 사용.
이후 validation에서 qmv OOB도 사라졌다.

---

## 6. 남은 플레이크의 정체 — 컴파일 참조가 비결정적

OOB 2건을 고친 뒤에도 full-model 테스트가 ~10% 실패했다. **자기일관성
probe**로 범인을 특정했다 — 같은 가중치·입력으로 양쪽 경로를 두 번씩 계산:

```
nat_self_max = 0.00000   # 항상 → NativeDecoder는 완전 결정론적
ref_self_max = 1.99609   # 실패 런에서 → mx.compile 참조가 비결정적
```

즉 **native 디코더는 결정론적이고 정확**했다. `model()`의 `mx.compile`된
디코드 참조(`Block.compiled_step` → `mx.compile(self.decode_step_math)`)가
**스위트 누적 메모리 압력 하에서 같은 입력에 다른 로짓**(~2.0 차이)을 냈다.
테스트가 이 불안정한 참조와 tight하게 비교해서 깨진 것.

부수 확인: `embed.weight.sum()`이 런마다 불변 → **가중치는 동일**(RNG/seed
문제 아님). native 출력도 stable 참조와는 tiny median(`~1e-3`)으로 일치했고,
argmax도 참조 top-3 안에 들었다. 손상 런은 median이 `~0.55`로 튀었다.

### 견고화한 테스트

native의 정확성은 per-layer-type plan 테스트들(dense/ratio-128/ratio-4/
FFN/full-block, 모두 eager `decode_step_math` 참조와 tight diff)이 이미 tightly
증명한다. full-model 통합 테스트는:

1. native 결정론 단언(두 번 재생 → 비트 동일),
2. 참조를 두 번 계산해 **자기일관적일 때만** 교차검증,
3. bf16 near-tie로 top-k 순위가 흔들리므로 argmax **순위**가 아니라 **로짓
   값**으로 비교(`exp[native_argmax] >= exp.max() - 0.1`) + median 바운드.

이후 full 스위트 **1394 passed** 15회+ 연속, 플레이크 소멸.

### 열린 질문 (follow-up)

`mx.compile`된 디코드 참조의 in-suite 비결정성은 **실서버가 쓰는 경로**다
(third-path native 이전의 기본 디코드). tiny 모델 + 스위트 압력에서만
재현됐고 단독 프로세스에선 오염을 넣어도 재현 안 됐다. 실서버(안정적 할당,
warmup 후)에서 나타나는지는 **미확인** — 별도 조사 필요. 후보: `mx.compile`
캐시/트레이스 상태가 메모리 압력에서 흔들리는지.

---

## 7. 교훈 / 체크리스트 (다음 Metal 커널 디버깅용)

* **in-suite에서만 실패 + 단독은 통과 + read-back 하면 사라짐** ⇒ 메모리
  해저드(OOB/aliasing) 의심. **가장 먼저 `MTL_SHADER_VALIDATION`을 켜라.**
  (단, GPU validation은 MLX eager Scatter를 깨니 native-only probe로.)
* **완전 직렬화**(dispatch당 command buffer + wait) 테스트로 순서 버그와
  in-kernel/OOB 버그를 깔끔히 분리하라. 직렬화해도 재현되면 OOB/미초기화다.
* **이중모드 결정적 손상**(정답 아니면 특정 오답)은 랜덤 쓰레기가 아니라
  **특정 stale 버퍼**를 가리킨다.
* 커널의 인덱스 상한을 **참조 상태 shape**와 대조하라. 여기선
  `CompressorState.reset`의 `(coff*ratio, coff*head_dim)` 중 `coff` 행 인자를
  놓친 게 원인이었다.
* 라이브러리 qmv(`affine_qmv_fast_*`) 하드 제약: **`N % 8 == 0`,
  `K % 512 == 0`**. 작은 차원(`index_n_heads` 등) 테스트 설정 주의.
* 발산 SIMD 리덕션이 **항상** 버그는 아니다 — 누산기를 루프 전에 초기화하면
  0-iteration 레인이 항등원을 들어 `simd_*`가 안전하다.
* MLX 버퍼는 hazard-untracked. 그리고 `arr[:] = ...` in-place setitem은 **새
  버퍼를 할당**해 등록된 MTLBuffer로 전파되지 않는다(seed는 buffer_contents
  + memmove로).
