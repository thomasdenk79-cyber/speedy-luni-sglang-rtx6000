# NVFP4 dense `in_proj_qkvz`: prefill paths (read-only donor analysis)

Source: `/tmp/opencode/sglang-donor-nvfp4kv` (donor snapshot 2026-11-24/25). No GPU runs, no edits.

## 0. Current state (facts)

- Packing hook: `nvfp4_dense.py:363-406` (`maybe_pack_dense_nvfp4`) runs at the top of `Fp8LinearMethod.process_weights_after_loading` (`fp8.py:854-861`), i.e. while `layer.weight` is still BF16; it then deletes the BF16 weight and installs `nvfp4_wq/bs/gscale` + `NVFP4DenseLinearMethod`.
- Packed layout (`w4a16_nvfp4_gemv.py:4-12, 449-495`): `wq[N,K/2]` uint8 (low nibble = even k), `bs[N,K/16]` uint8 holding **raw e4m3 bit patterns**, `gscale[1]` fp32 `= amax(|W|)/(448*6)` (ModelOpt convention). The decode kernel folds a 2^14 bit-trick factor into the epilogue (`:60-63, 194-197`); the dequant helper does not (`:512-533`).
- Dispatch: `apply` (`nvfp4_dense.py:222-231`) → GEMV when `_ok` (M <= `_m_limit()`, bf16, contiguous). Otherwise `_wide_m` (`:293-333`): M<=128 loops the GEMV over knob-sized slices (CUDA-graph safe); large M dequantizes in 16384-row chunks to **BF16** and calls `torch.mm` (`:328-333`). That is the regression.
- Relevant shape (from tuning table comments `w8a16_gemv.py:463, 476`): `in_proj_qkvz` N=16384, K=2560, per layer, TP=1.
- Platform gates: `is_blackwell_supported()` includes SM100/110/**120** (`sglang/srt/utils/common.py:278-284`); `is_sm100_supported()` is SM100 only (`:290-294`). FP4 auto backend is `flashinfer_cutedsl` only on SM100, `marlin` for SM80–SM99, else `flashinfer_cutlass` (`fp4_utils.py:143-156`). This fork targets **SM120** (`w4a16_nvfp4_gemv.py:110`), so AUTO FP4 would select `flashinfer_cutlass`, not cutedsl/Marlin.

## 1. NVFP4 -> FP8 e4m3, then an existing FP8 GEMM — YES, cleanest

Nothing in the donor converts *dense* weights NVFP4->FP8 at runtime. `cast_e2m1fn_to_e4m3fn` (`fp8.py:189-231`) is MXFP4/MoE-only: int8 input, fp4 block 32 / fp8 block 128, ue8m0 scales — **not reusable** for our 16-element NVFP4 blocks. `fp8.py:1819-1837` is the analogous load-time *dual-copy* precedent for MoE experts.

But the FP8 GEMM infrastructure is already present and importable:

- `fp8_utils.py:1788-1963` `apply_fp8_linear`: BF16 input -> `sglang_per_token_quant_fp8` (`:1904-1905`) -> if `channelwise_cutlass` (weight_scale.numel() == weight.shape[1], `:1829-1835`) and shape %16==0 -> `fp8_scaled_mm` (`:1955-1962`) or tuned Triton (`:1935-1953`). That is exactly the FP8 prefill GEMM baseline (2x BF16 math).
- `fp8_scaled_mm` is imported from `sgl_kernel` at `fp8_utils.py:204-215` (with fake impl), so a direct call is available: `fp8_scaled_mm(qx[M,K], w8[K,N], x_scale[M,1], w_scale[N], out_dtype=bf16)`.
- FP8 weights must be **[K,N]** (`fp8.py:889`, `fp8_utils.py:1814, 1830`), so the dequantized `[N,K]` tile is transposed before the GEMM.
- Scale choice: per-output-channel `w_scale[n] = amax_n/448` keeps the fast channelwise path; using a single global scale would drop into `torch._scaled_mm` per-tensor and, with per-token activation scales, into the slow unfused fallback (`fp8_utils.py:1965-2028, 2044-2052`).

Concrete minimal change (in `_wide_m`, large-M branch only):
1. Quantize x once: `qx, x_scale = sglang_per_token_quant_fp8(x)` (defined in `sglang/kernels/ops/quantization/fp8_kernel.py`, re-exported via `fp8_utils.py:10-20`, used at `:1905`).
2. Per row chunk: `w = dequantize_nvfp4(wq[r0:r1], bs[r0:r1], gs)` (`w4a16_nvfp4_gemv.py:512-533`), `s = w.abs().amax(-1).clamp(min=1e-8)/448`, `w8 = (w/s[:,None]).clamp(-448,448).to(torch.float8_e4m3fn).T.contiguous()`.
3. `y[:, r0:r1] = fp8_scaled_mm(qx, w8, x_scale, s, out_dtype=torch.bfloat16)`.

Stage 2 (if step-2 torch ops dominate): fuse unpack + e4m3 block scale + global scale + e4m3 requant into one Triton kernel that reads the 22.5 MiB packed tile and writes 40 MiB FP8, reusing `_unpack_dequant` / `_load_block_scales` (`w4a16_nvfp4_gemv.py:73-122`). This removes the 160 MiB fp32 transient.

Transient VRAM: naive torch path ~200 MiB (160 MiB fp32 + 40 MiB FP8); fused kernel ~40 MiB. Current path is already ~240 MiB (160 MiB fp32 + 80 MiB bf16). Persistent VRAM: **unchanged** (only NVFP4 stays resident).

Accuracy caveat: e2m1 (1 explicit mantissa bit) x e4m3 (3) needs up to ~5 bits; e4m3 keeps 3, so requantization is lossy by up to ~6% relative. Must pass the existing acceptance gate before shipping.

## 2. Native NVFP4 Marlin dense GEMM — available, moderate port, not a 2x-FP8 win

`apply_fp4_marlin_linear` (`marlin_utils_fp4.py:78-136`) is a registered custom op calling `gptq_marlin_gemm` with `scalar_types.float4_e2m1f`, FP16/BF16 activations (`:89`), `b_scales`, `global_scale`, workspace. Prep is `prepare_nvfp4_layer_for_marlin` (`:139-212`).

Our raw `wq/bs/gscale` are **not** directly feedable:
- Weight must be padded to N%64/K%128 or N%128/K%64, viewed as int32, transposed, then `gptq_marlin_repack` (`:155-190`). Our `wq` is already packed but in the wrong (unpermuted, un-transposed) layout.
- Scales: `bs.T -> marlin_permute_scales(group_size=16) -> nvfp4_marlin_process_scales` = cast half, `view(-1,4)[:,[0,2,1,3]]`, `*2^7`, bit-shift `<<1`, view e4m3, take `[:,1::2]` (`:192-200`, `:28-43`). Our `bs` is raw e4m3 with no swizzle.
- Global scale: must be fp16/bf16 then multiplied by `2^(exponent_bias-7)` (`:46-58`, `:202-205`). Our `gscale` is fp32 scalar -> one cast + mul at load.
- Needs `layer.workspace` from `marlin_make_workspace` (`:179`).

Wire-up pattern already exists in `modelopt_quant.py:1962-1972` and `:2125-2135`. So: moderate one-time prep at load, plus a repacked second copy if the Triton GEMV must keep reading the original codes. Repacked weight ≈ same byte count as the NVFP4 copy (~22.5 MiB/layer), scales ~2.5 MiB. Marlin is weight-only W4A16 — Activations stay BF16; **no guarantee of FP4 tensor-core throughput**, the gain is mainly fusing the dequant into the GEMM. On SM120 its performance vs the FP8 path is unknown. High risk / medium speed; not recommended as first move.

## 3. Dual copy (FP8 for prefill + NVFP4 for decode) — fastest, but VRAM-heavy

Trivially dispatchable: extend `NVFP4DenseLinearMethod.attach` (`nvfp4_dense.py:203-208`) with an FP8 weight + scale, keep the GEMV branch for M<=limit and call `apply_fp8_linear`/`fp8_scaled_mm` for large M. At load (`nvfp4_dense.py:394`, while `w` is still BF16) reuse `per_token_group_quant_fp8(w, K)` and transpose exactly as `fp8.py:877-890`.

Per `in_proj_qkvz` (N=16384, K=2560): NVFP4 22.5 MiB (20 MiB codes + 2.5 MiB scales), FP8 40 MiB + 64 KiB scale, BF16 original 80 MiB. **+40 MiB per GDN layer**; for L such layers the added residency is `40*L` MiB (e.g. L=48 -> ~1.9 GiB, L=60 -> ~2.3 GiB). The donor notes only ~2 GiB free at `mem_fraction_static=0.935` (`nvfp4_dense.py:153-156`), so this likely requires lowering mem-fraction (less KV cache/concurrency) or is outright OOM. Exact layer count on the deployed model is **unknown**. Speed is the most predictable (exact FP8 baseline GEMM, no dequant in the hot path) — potentially restores the full 14938 tok/s and might exceed it if the old FP8 scale layout was per-tensor.

## 4. Native FP4 CUTLASS / FlashInfer dense GEMM — exists, W4A4, biggest unknown

Yes: `fp4_gemm` at `modelopt_quant.py:141-160` wraps `flashinfer.mm_fp4`, backends remapped in `fp4_utils.py:122-137`. ModelOpt's dense apply is `modelopt_quant.py:2119-2185`: activations are quantized with `fp4_quantize(x, input_scale_inv)` (`:2147`; wrapper + fake at `fp4_utils.py:20-88`), weight is transposed (`:2166-2168`), scales swizzled (`:2033-2071`), and `alpha = input_scale_2 * weight_scale_2` (`:1952-1954`). Alignment N%32/K%32 (`:173, 2034-2036`); our N=16384, K=2560 already satisfy it.

Our ingredients match the ModelOpt convention (per-16 e4m3 block scales + global scale), but:
- `bs` must be block-interleaved/swizzled into `weight_scale_interleaved` (pure-torch reshape/permute at `modelopt_quant.py:2061-2068`, or `swizzle_blockscale_2d` at `:2074-2077`).
- Need an activation scale (`input_scale_inv`); dynamic `global_scale=None` is supported by `fp4_quantize` but the wrapper is only exercised with a checkpoint scale in this fork.
- W4A4 activation quantization (e2m1 x2) is much coarser than FP8; accuracy risk for prefill logits is unmeasured.
- On SM120, AUTO picks `flashinfer_cutlass` (`fp4_utils.py:143-156`); no evidence in the donor that W4A4 CUTLASS beats FP8 on SM120 for M=prefill, N=16384, K=2560. Unknown; would need a kernel-level benchmark.

Port size: moderate-to-large (activation quant + swizzle + alpha + backend validation); persistent VRAM ~+2.5 MiB/layer for the swizzled scale copy (weight transpose can be a view).

## Ranking

| Option | Speed gain (expected) | Impl. risk | Persistent VRAM / layer | Verdict |
|---|---|---|---|---|
| 3. Dual FP8 copy | Highest / most predictable (restores FP8 GEMM exactly) | Low (~15 lines + load-time quant) | **+40 MiB** (≈1.9–2.3 GiB for 48–60 layers) | Best if memory allows; likely needs mem-fraction cut |
| 1. On-the-fly NVFP4->FP8 + `fp8_scaled_mm` | High but uncertain (FP8 tensor cores restored, dequant added); could recover most of the 17–19% | Low–medium (no new kernel needed for v1; accuracy gate) | **0** (40–200 MiB transient) | **Recommended first change** |
| 4. FlashInfer/CUTLASS W4A4 `mm_fp4` | Highest theoretical (2x FP8 math) but completely unmeasured on SM120; activation accuracy risk | Medium–high | ~2.5 MiB (swizzled scales) | Interesting follow-up, benchmark first |
| 2. Marlin FP4 dense | Medium (fuses dequant, likely still BF16-class math) | Medium (repack + scale preprocessing, dual packed copy) | ~22.5 MiB (repacked weight) | Weakest case on SM120 |

## Concrete minimal-change recommendation

1. **Ship Option 1 v1** in `nvfp4_dense.py:328-333`: keep the row-chunk loop and `_FALLBACK_ROW_CHUNK`; replace `dequantize_nvfp4(...).to(torch.bfloat16)` + `torch.mm` with e4m3 requant + `fp8_scaled_mm` (import from `fp8_utils.py:205`). Quantize the activation once outside the loop with `sglang_per_token_quant_fp8` (`fp8_utils.py:1905`). Keep the M<=128 GEMV loop at `:318-327` untouched. Guard with an env flag so the old BF16 path can be compared under the same cold-prefill harness.
2. Validate numerics against the BF16 fallback and the FP8 baseline before benchmarking; if e4m3 requantization fails acceptance, switch to Option 3 (dual copy) — the code diff is similarly small and speed is guaranteed.
3. Only if neither reaches the target, prototype Option 4 behind a flag using `modelopt_quant.py:2033-2071, 2119-2185` as the template; run a standalone `mm_fp4` shape benchmark on SM120 first.

## Explicit unknowns

- Actual number of GDN layers and current free VRAM on the deployed model (VRAM totals for Option 3 are per-layer formulas).
- End-to-end tok/s for Options 1/2/4 — no GPU was used; speed claims are architectural estimates only.
- Whether NVFP4(e2m1) -> e4m3 requantization passes the model's acceptance gate.
- Whether FlashInfer `mm_fp4` cutlass backend is compiled/performant for SM120 and this shape; whether native FP4 tensor cores are used.
- Whether Marlin FP4 on SM120 uses FP4 or BF16 mma internally.
- Whether prefill `_wide_m` is called with M<=128 on some chunks (those keep the GEMV loop and are unaffected).
- Exact cold-prefill shape distribution (M) used for the 14938 -> 12043 measurement.

## 10-line summary

1. `_wide_m` large-M path (`nvfp4_dense.py:328-333`) dequantizes to BF16 and calls `torch.mm`; that is the only prefill regression source.
2. Existing FP8 GEMM (`fp8_utils.py:1788-1963`, `fp8_scaled_mm` at `:205`) can be reused with per-channel FP8 weight scales and `[K,N]` layout (`fp8.py:889`).
3. No dense NVFP4->FP8 helper exists; `cast_e2m1fn_to_e4m3fn` (`fp8.py:189-231`) is MoE/MXFP4-specific and cannot be used.
4. On-the-fly e4m3 requant costs no persistent VRAM but is lossy (~2 mantissa bits) and needs an acceptance check.
5. Marlin FP4 (`marlin_utils_fp4.py:78-212`) needs repacked weights, permuted/swizzled scales and a converted global scale — moderate port, likely BF16-class compute, no proven SM120 win.
6. Dual FP8 copy gives guaranteed FP8 prefill speed and a tiny code diff, but costs **+40 MiB per layer** (~1.9–2.3 GiB for 48–60 GDN layers) against ~2 GiB free.
7. Native W4A4 `mm_fp4` exists (`modelopt_quant.py:141-160, 2119-2185`) and our scales already use the ModelOpt convention, but it needs swizzled scales + FP4 activation quantization.
8. AUTO backend on SM120 selects `flashinfer_cutlass`, not cutedsl/Marlin (`fp4_utils.py:143-156`) — performance is unknown.
9. Ranking: dual-copy (speed) > on-the-fly FP8 (risk/VRAM balance) > W4A4 (theoretical) > Marlin.
10. Recommendation: implement on-the-fly NVFP4->FP8 + `fp8_scaled_mm` behind a flag in `_wide_m` first; fall back to the dual copy if accuracy or latency targets are missed.
