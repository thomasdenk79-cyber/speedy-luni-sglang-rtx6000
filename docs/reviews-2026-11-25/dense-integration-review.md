# Dense NVFP4 linear-attn integration review (read-only)

**Scope:** source + lab notes only. No repo edits, no GPU, no port 8001 touched.
**Sources:** `/tmp/opencode/sglang-donor-nvfp4kv` (feature HEAD `dc333b6` + 2 uncommitted patches), `/tmp/opencode/sglang-donor-nvfp4-run` (run HEAD `f9bf31a`), benchmark evidence `donor-nvfp4-dense-linear-20261125T063614Z`, `N1_LOG.md`, `FG1_GDN_FRONT_OVERLAP.md`, prior audit `analysis/nvfp4-dense-gates.md`.

## 0. TL;DR — 10-line ranked summary

1. **CONFIRMED** gap A: `NVFP4DenseLinearMethod` has no `apply_into_split` in the run tree (`f9bf31a`); the feature tree patch adds it and the underlying GEMV already supports `out2/split_n` (`w4a16_nvfp4_gemv.py:384-411`) — the hook is implementable, not missing kernel capability.
2. **CONFIRMED** gap B: after packing, `in_proj_qkvz.weight` is the `[N,K/2]` uint8 code tensor (`16384,1280`), so `weight.shape[-1]==1280 != 16384`; the original shape gate at `qwen3_5.py:809` necessarily returns False, independent of gap A.
3. Both gaps disable the same single path: `_direct_layout_ok()` → `_forward_input_proj_direct()` + `front_gate_stream()`; the fallback is `_forward_input_proj()` + a separate `fused_qkvzba_split_reshape_cat_contiguous` kernel.
4. `commit f9bf31a..dc333b6` touches none of `nvfp4_dense.py`, `qwen3_5.py`, `fp8.py`, `w4a16_nvfp4_gemv.py`, `qwen4_exp.py`; the run tree lacks both fixes only because they are uncommitted in the feature tree.
5. **Rank 1 (measurable, ~1.2 %):** missing `apply_into_split` — lab-measured fallback cost **+115 µs** of a ~10 ms step (N1_LOG §13), i.e. below the 5 % bar by itself.
6. **Rank 2 (~1.3 %):** `front_gate_stream` is gated on `_direct_layout_ok()` (`qwen3_5.py:862-870`, `qwen4_exp.py:1970`); losing it costs ~3.2 µs × 36 layers ≈ 0.13 ms/step (FG1 §0). Also below 5 %.
7. **Rank 3 (plausibly >5 % of prefill):** `_wide_m()` fallback (`nvfp4_dense.py:293-333`) dequantises the packed weight to BF16 in 16384-row chunks then `torch.mm`, replacing FP8's native `apply_fp8_linear` GEMM at M>32/128 — consistent with the −17..−19 % cold-prefill result, but not a "missing hook" in the interface sense.
8. **Rank 4 (~5.2 % of step, not a hook):** N1_LOG §13 shows the faster qkvz GEMV *uncovers* roofline-bound MoE grouped GEMMs (`+348/+171 µs`) — a stream/overlap consequence of a faster kernel; the only documented mechanism of this size.
9. **Ruled out (0 %):** `apply_norm_gated` (only queried on `out_proj`, which `maybe_pack_dense_nvfp4` deliberately never packs), `apply_weight_name_mapper` (load-time, on `Fp8Config`, not on the runtime method), weight preprocessing (correctly hooked at `fp8.py:854-861` before FP8 quantisation).
10. **Important caveat:** at BB c8 with EAGLE verify, `M = 8 × (steps+1) ≈ 32–64 > 16`, while `_forward_input_proj_direct` and `front_gate_stream` are hard-capped at 16 rows; therefore gaps 1–2 cannot explain the c8 −18 % on their own — ranks 3–4 are the c8/prefill candidates.

## 1. Verification of the two reported gaps

### Gap A — `apply_into_split` absent
- Run tree `f9bf31a` `nvfp4_dense.py`: only `apply`, `apply_into`, `_wide_m`, `_ok`, `attach`. **No `apply_into_split`.**
- Feature tree patch (`nvfp4_dense.py:253-291`) adds it as a two-destination `apply_into` with shape/dtype/contiguity guards, returning `False` to decline.
- The gate that consumes it is `qwen3_5.py:805-806`: `getattr(self.in_proj_qkvz.quant_method, "apply_into_split", None) is not None`. With the run-tree method this is False.
- The GEMV itself already implements the native split store: `w4a16_nvfp4_gemv(..., out2, split_n)` at `w4a16_nvfp4_gemv.py:365-424`, `_out_or_new` split branch at 384-390 and `SPLIT_N` epilogue at 411. So the missing hook is purely a wiring omission.
- The patch is signature-compatible with the FP8 reference (`fp8.py:1081-1118`) and with the only call site (`qwen3_5.py:909-916`, passing `mixed_qkv`, `z.view(M,-1)`, `qkv_dim`); N splits as `16384 = (2*key_dim+value_dim) + value_dim`, so `split_n=qkv_dim` is valid.

### Gap B — shape gate fails on packed weights
- `maybe_pack_dense_nvfp4` (`nvfp4_dense.py:363-406`) quantises `layer.weight` into `wq [N,K/2]`, calls `prealloc`, attaches `nvfp4_wq/bs/gscale`, **deletes** `layer.weight`, then reassigns `layer.weight = wq` (line 401). It also sets `weight_scale=None`, `input_scale=None` and replaces `quant_method` (402-404).
- Runtime log confirms `NVFP4 dense: ...linear_attn.in_proj_qkvz (16384, 1280) packed` for the packed layers.
- Original gate (`qwen3_5.py:809-810`): `weight.shape[-1] == 2*key_dim + 2*value_dim` → `1280 == 16384` → False. FP8 works because Fp8 stores the weight transposed `[K,N]`, so `shape[-1] == N`.
- Feature patch (808-816) adds an NVFP4 branch checking `nvfp4_wq.shape[0] == 2*key_dim + 2*value_dim`, which is true.
- The two gaps are redundant causes: fixing only one leaves `_direct_layout_ok()` False.

### Worktree delta
`git diff f9bf31a..dc333b6 -- <the five relevant files>` is **empty** — the only commit between them is `dc333b6 "launcher: gate fastsafetensors load format behind LAB_FAST_LOAD_FORMAT"`. The fixes exist solely as uncommitted changes in the feature tree.

## 2. Benchmark context (guards against misattribution)

- `plan.json` comparison: argv byte-identical between baseline `donor-cl-base-20261125T050612Z` and the candidate; **the only env difference is `SGLANG_LINEAR_ATTN_NVFP4: None -> 1`**. KV capacity +9.3 % is a side effect of freed dense-weight VRAM, not a second flag.
- Flags that matter are all ON in both arms: `SGLANG_GDN_PROJ_DIRECT_LAYOUT=1`, `SGLANG_OPT_GDN_FRONT_OVERLAP=1`, `SGLANG_GDN_BA_TRITON_GEMV=1`, `SGLANG_GDN_AB_STASH_DIRECT=1`, `SGLANG_FP8_W8A16_GEMV=1`, both GEMV M-limits 32.
- Served model is `qwen4_exp` (`--quantization modelopt_fp4`, `--qwen4-exp-dense-fp8 ...linear_attn...`). `Qwen4ExpLinearDecoderLayer` inherits `Qwen3_5LinearDecoderLayer` and its `linear_attn` is a `Qwen3_5GatedDeltaNet` (`qwen4_exp.py:69, 1945-1957`), so the `qwen3_5.py` gates are the live ones; `qwen4_exp.py:1970` is the only extra consumer (`_attn_gate_stream` → `front_gate_stream`).
- **M caveat:** `_forward_input_proj_direct` refuses `M>16` (`qwen3_5.py:879-882`) and `front_gate_stream` requires `0<num_tokens<=16` (865). EAGLE verify at c8 uses `M = 8×(steps+1) ≈ 32–64` (adaptive config candidate steps 3/7; observed accept len ~3.2). Both arms therefore take the non-direct path at c8 regardless of gaps A/B. The gaps are decisive only at c1–c4 under graph capture.

## 3. Hook-by-hook audit: FP8 path vs NVFP4 dense path

| Hook / gate | FP8 | NVFP4 dense | Live for qkvz? | Verdict |
|---|---|---|---|---|
| `apply(layer,x,bias)` | `fp8.py:1120-1231` (GEMV → mxfp8/block/fp8 GEMM) | `nvfp4_dense.py:222-231` (GEMV → `_wide_m`) | Yes | Present; fallback quality differs (row 8 of summary) |
| `apply_into` | `fp8.py:1005-1032` | `nvfp4_dense.py:233-251` | Only via `apply_into_split` call site | Present |
| `apply_into_split` | `fp8.py:1081-1118` | **absent in run tree**, patch in feature tree | Yes — `_direct_layout_ok` gate + call site | Real gap, but only M≤16 |
| Shape gate on weight layout | `weight.shape[-1]==N` holds for `[K,N]` FP8 | fails for `[N,K/2]` packed codes | Yes | Real gap, redundant with above |
| `apply_norm_gated` | `fp8.py:1034-1079` | absent | **No** — only queried on `self.out_proj` at `qwen3_5.py:1101`; `out_proj` matched by no category in `_cat_flags()` and is deliberately excluded (`nvfp4_dense.py:366-369`) | No effect |
| `apply_weight_name_mapper` | `fp8.py:434-438`, on the **config/method factory**, applied at load time to `ignored_layers` | absent | No — NVFP4 method is installed after load; `qwen4_exp.py:2504-2548` maps checkpoint names into the FP8 layer before packing | No effect |
| Weight preprocessing / packing | `process_weights_after_loading` (`fp8.py:854-861`) calls `maybe_pack_dense_nvfp4` **first, on the still-BF16 fused tensor**, and returns early | packed in that hook (`nvfp4_dense.py:394-405`) | Yes, correctly ordered (before FP8 quant, before graph capture) | No missing hook |
| Post-pack attribute integrity | `weight_scale`, `input_scale`, per-channel scale remain | explicitly set to `None`; `quant_method` replaced | `_linear_accepts_fp8_tuple` (`qwen3_5.py:264-269`) would misroute tuple input, but that needs `_use_aiter` (ROCm); this run is CUDA | No effect here |
| CUDA-graph capture safety | `prealloc` in `process_weights_after_loading` (`fp8.py:981-987`) | `prealloc(wq.device)` at pack time (`nvfp4_dense.py:392-395`); `_m_limit()` uses capture-aware `get_gemv_m_limit` (`forward_context.py:82-87`); `_wide_m` small-M loop has no syncs/host copies (293-327) | Yes | No missing hook found |
| Dual-stream overlap (non-direct) | `_forward_input_proj` 690-701, `_in_proj_ba_skinny` 716-732 | unchanged; qkvz goes through `apply` | Yes, still active at all M | No gap |
| Dual-stream/fused split (direct) | `_direct_use_alt` 842-854, `_forward_input_proj_direct` 872-943, `front_gate_stream` 856-870 | all gated off by `_direct_layout_ok()` | M≤16 only | Real gap (summary 1–2) |
| Medium/large-M fast GEMM | FP8 falls through to `apply_fp8_linear` for M>32 | dequantize-to-BF16 + `torch.mm` in `_wide_m` (`_FALLBACK_ROW_CHUNK=16384`, `_CHUNK_M_MAX=128`) | Yes at c8/prefill | Quality gap, plausible >5 % on prefill |
| Stream cover for MoE GEMMs | 88.5 % exclusive qkvz GEMV hides expert GEMMs | faster GEMV → 75.4 % exclusive, exposes +348/+171 µs (N1_LOG §13) | Yes | Consequence, not a hook |

## 4. Cost estimate and ranking

| Rank | Item | Mechanism | Plausible cost | Clears 5 %? |
|---|---|---|---|---|
| 1 | `apply_into_split` missing | extra split/reshape/cat kernel per GDN layer; N1 measured **+115 µs/step** | ~1.2 % of 10 ms step | No |
| 2 | `front_gate_stream` lost (same gate) | 3.2 µs × 36 layers; FG1 §0 | ~0.13 ms/step ≈ 1.3 % | No |
| 3 | `_wide_m` dequant+bf16 GEMM vs FP8 GEMM | M>128: materialise 16384×1280 BF16 + GEMM before every GDN qkvz | plausibly >5 % of **prefill**; explains −17..−19 % cold prefill | Yes (prefill) |
| 4 | MoE-exposure from faster GEMV | +348/+171 µs on roofline-bound MoE GEMMs (N1_LOG §13) | ~5.2 % of step | Yes (but not a missing hook) |
| 5 | `apply_norm_gated` | unused on packed layers | 0 % | No |
| 6 | `apply_weight_name_mapper` | load-time only | 0 % | No |
| 7 | weight preprocessing | correct hook, correct order | 0 % | No |
| 8 | CUDA-graph / capture interactions | prealloc + capture-aware M-limit present | 0 % | No |

## 5. Interpretation / recommendation for the parent agent

- The two patches are **correct and worth applying**, but they restore only ~2.5 % of step time at M≤16; they are not, by the code and lab evidence, a sufficient explanation for a c8 −18 % E2E loss.
- At c8 the direct path is ineligible in both arms (M>16), so the regression must come from either (a) the `_wide_m` fallback quality at M=32–64 / prefill, or (b) the documented MoE-exposure effect of a faster qkvz kernel, or (c) a second-order effect outside these hooks (e.g. spec acceptance / noise — E2E tok/s is ±20 % at 2 repeats per N1_LOG §14; the BB ran 3 repeats with 20 runs/level and conf 0.95, so this is less likely but not excluded).
- Before attributing the −18 %, instrument the candidate for: `M` at `Qwen3_5GatedDeltaNet.forward`, fraction of qkvz calls taking `_wide_m` vs GEMV, and per-family CUPTI exclusive times (qkvz, MoE GEMM1/2, split kernel). That separates ranks 1–4 in one pass.
- Do **not** expect the feature-tree patches to fix the cold-prefill regression: the direct path is a decode/verify-only optimisation.
