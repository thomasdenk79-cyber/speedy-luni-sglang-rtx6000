# Native NVFP4 on SM120 — Phase A/B/C plan

**Scope:** source-only preparation. No GPU used for this document; the
qualification screen on port 8001 was not interrupted.

## 1. Reclassification

P0.2 (`SGLANG_NVFP4_DENSE_PREFILL_FP8`) and P0.2b
(`SGLANG_NVFP4_DENSE_PREFILL_FP8_CACHE`) are **diagnostic experiments, not
production candidates**. P0.2b reached `qualified`
(`donor-nvfp4-dense-fp8cache-full-20261125T100830Z`, 39 artifacts) but is
Pareto-dominated by the original FP8 baseline: ~4 % lower cold prefill and
~12 % lower KV capacity (763392 vs 947392). A gain over a regressed
intermediate is not a success. Both switches stay off.

## 2. Target architecture (binding)

- NVFP4 is the canonical GPU-resident weight representation.
- No persistent expanded FP8/BF16 weight copies, no per-forward dequant
  materialisation in global memory, no NVFP4 -> FP32 -> FP8 in the hot path.
- Quantise the original BF16 weight to NVFP4 at load time; do layout
  transformation and scale swizzling at load time.
- Keep the qualified W4A16 decode GEMV; add native W4A4 block-scaled GEMM for
  medium/large M. Share the physical packed weight allocation; if layouts
  differ, adapt the GEMV rather than duplicating the weight.
- Preserve CUDA-graph compatibility, direct projection layout, split-output
  stores and stream overlap. Fuse activation quantisation where beneficial.
- Flags: `SGLANG_NVFP4_DENSE_PREFILL_NATIVE` (off by default) and
  `SGLANG_NVFP4_DENSE_PREFILL_NATIVE_BACKEND` (explicit, default `auto`).

## 3. Source findings (line-referenced, read-only)

- Packed format (`w4a16_nvfp4_gemv.py`): `wq[N,K/2]` uint8, low nibble = even k
  (`:494`, decode `:526-527`); `bs[N,K/16]` uint8 holding raw e4m3 bit patterns
  (`:487-488`); `gscale = amax(|W|)/(448*6)` (`:471`); the 2^14 bit trick is
  folded into the epilogue global scale only (`:60-63,195`).
- One `wq` **can** be shared: `wq.t()` is a zero-copy `[K/2,N]` B for `mm_fp4`
  (`nvfp4_dense.py:449`), exactly ModelOpt's `layer.weight.T`
  (`modelopt_quant.py:2167`). `bs` cannot: GEMV needs row-major e4m3 bits,
  GEMM needs a 128x4-swizzled copy from `block_scale_interleave`
  (`nvfp4_dense.py:607`, ModelOpt's manual permute `:2061-2068`).
- `alpha = amax_x/(448*6) * gscale`, computed on device (no host sync).
- GEMV max M is 16 default / 32 hard cap; its M<=128 loop is capture-safe.
  Direct projection layout and `front_gate_stream` are M<=16 decode-only
  (`qwen3_5.py:865-882`) and are untouched by the large-M native path.
- Native SM120 kernel exists **upstream**:
  `kernels/kda_kernels/qwen3x_nvfp4_gemm_sm120.py` (`MmaMXF4NVF4Op`, last
  commit `977e769b`, PR sglang#36865 / merge `c593527f33`). It only accepts
  M in {1,2,4,8} and a narrow shape whitelist that excludes (N=16384,K=2560);
  usable as a CUTLASS-DSL template, not directly reusable.

## 4. Toolchain verified (CPU-only)

| Component | Version / fact |
|---|---|
| torch | 2.13.0+cu130 |
| CUDA runtime | 13.0 |
| nvcc | 13.5.37 (`/opt/cuda/bin/nvcc`) |
| FlashInfer | 0.6.17 |
| CUTLASS (bundled) | 4.5.0 (`flashinfer-vt/flashinfer/data/cutlass`) |
| sglang fork / upstream mirror | `8edd1fa` / `108c00266f` |
| Profilers | `cuobjdump`/`nvdisasm` present; `ncu`/`nsys` absent |

On SM120 sglang's FP4 AUTO selects `flashinfer_cutlass`
(`is_sm100_supported()` is SM100-only); FlashInfer `mm_fp4(backend="auto")`
prefers `b12x` on CUDA 13. Phase A pins the backend explicitly.

## 4b. Upstream capability evidence (exact commits)

- FlashInfer 0.6.17 `mm_fp4`: `b12x` is **SM120-only** (`[120,121]`), CUDA >= 13,
  128x4 E4M3 scales, NVFP4 only, K % 32 == 0; on SM120 `auto` prefers b12x,
  then cutlass, then cudnn. `cutlass`/`cudnn` also support SM120; `trtllm`
  `[100,103,107]` and `cute-dsl` `[100,103]` do **not** (explicit rejection,
  not a silent fallback). The SM120 b12x kernel uses `MmaMXF4NVF4Op`
  m16n8k64 with Float4E2M1FN / Float8E4M3FN — native block-scaled FP4 MMA.
- CUTLASS example 79a (last touch `0d2b201e8c1c4a03efa6e9c468161916e2334725`)
  is explicitly **SM120/SM121**, `OpClassBlockScaledTensorOp`, cluster 1x1x1,
  CUDA >= 12.8; it is not SM100-only. The bundled CUTLASS 4.5.0 has no
  `examples/79_*`, so using it would require the upstream example.
- `Infatoshi/nvfp4-sm120` HEAD `12e9256073e8eccfaa46c1ae7a6e4773c373328e`
  (GEMM `92f9a987fd8fcb566d35f7a807de1007b482c101`): native FP4 via
  `torch._scaled_mm` + torchao `_addmm_nvfp4_dispatch`, ~1100 TFLOPS at
  16384^3. **Silent fallback:** `F.linear`/`torch.mm` on `NVFP4Tensor`
  dequantizes; must call the dispatch directly.
- Colfax tutorial (2026-06-20): SM12x uses warp-level `mma.sync`, **not**
  `tcgen05`/TMEM; SM10x GEMM kernels will not run on SM12x. Only block-scaled
  instruction: `mma.sync.aligned.kind::mxf4nvf4.block_scale...m16n8k64`.
- SGLang upstream `qwen3x_nvfp4_gemm_sm120.py` silently returns `None` for any
  shape outside its whitelist; `sm120_nvfp4_moe.py` likewise. Never rely on
  those without asserting the dispatched kernel.

## 5. Phase A — kernel microbenchmarks (before any further full-model run)

Prepared source: `/tmp/opencode/phase-a/bench_nvfp4_phase_a.py` and README.
Shape N=16384, K=2560; M = 1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192
plus observed chunk sizes. Candidates: BF16 reference, production FP8 baseline
(`sgl_kernel.fp8_scaled_mm`), qualified W4A16 GEMV, FlashInfer `mm_fp4` with
explicit `b12x`/`cutlass`/`cudnn`, sglang `fp4_gemm`, and `mm_bf16_fp4` as the
no-activation-quant fallback. Measure activation quant, GEMM, launches/syncs,
temporary/persistent VRAM, memory traffic, tensor-core utilisation, CUDA-graph
replay, and numerical error vs BF16 and FP8. Verify native block-scaled FP4
MMA with `cuobjdump -sass` (`OMMA`/`UTCMMA`); no silent fallback — errors are
recorded per candidate. Crossover M must come from these measurements.

## 6. Phase B / C

Phase B: implement the winner behind the default-off flag above, validate
nibble order, e4m3 bits, swizzle, global-scale direction, activation scale,
orientation, alignment/padding and alpha against ModelOpt. If FP4 activation
quantisation hurts accuracy, try calibrated/dynamic scaling, 4over6, selective
precision or fused W4A16 before accepting any expanded weight copy.

Phase C gates, exclusively vs the **original qualified baseline**: cold
prefill >= baseline within variance, KV >= 947392, no material accuracy loss,
no decode regression at C1-C10, preserved speculative acceptance, long-context
and prefix-cache behaviour, no fallback/OOM/graph/concurrency regressions.
Report TTFT, inter-token latency, decode throughput, aggregate throughput, KV
capacity and quality separately. Short isolated screens first.

## 7. Current status

P0.2c screen `donor-nvfp4-dense-native-fp4-20261125T112203Z` is a **diagnostic
screen**, not a qualification run. Startup passed with KV=932288 (vs 763392
for the FP8 cache), confirming the single-weight layout keeps KV capacity.
Accuracy and cold-prefill results pending; no candidate has passed Phase C.
