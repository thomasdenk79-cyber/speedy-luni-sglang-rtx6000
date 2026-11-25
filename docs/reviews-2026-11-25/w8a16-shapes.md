# W8A16 GEMV shapes — verification report

**Scope:** read-only. No repo edits, no GPU use, port-8001 benchmark untouched (no listener present at check time).
**Sources:** safetensors header reads (first 8-byte length + JSON header, no tensor loads) under
`/srv/llm-models/Qwen3.8-Flash-Next-NVFP4/`; `config.json`; `model.safetensors.index.json`;
loader fusion in `/tmp/opencode/sglang-donor-nvfp4kv/python/sglang/srt/models/qwen4_exp.py`;
GDN module in `.../srt/models/qwen3_5.py`; kernel planner in
`.../srt/layers/quantization/w8a16_gemv.py`; NVFP4 packing hook in
`.../srt/layers/quantization/nvfp4_dense.py`; trace re-aggregation of
`/srv/llm-cache/qwen38-flashnext/profile/c1-20261125T041807Z/c1-1795580295.0387323-TP-0.trace.json.gz`
and `c4-20261125T041823Z/c4-1795580311.7414348-TP-0.trace.json.gz`.

## Verdict on the doc

`DONOR-NVFP4-PROFILE.md`'s shape column is **correct for all five rows**. No doc shape needs correction.
The hypothesis in the task prompt that q/k/v are `[7552, 2560]` each is **wrong**: the checkpoint has
`q_proj [12288, 2560]`, `k_proj [512, 2560]`, `v_proj [512, 2560]`; fused qkv is `13312 x 2560`
(the doc's number). The grid-derived shapes are consistent with the real tensors and with the
kernel planner's BLOCK_N / SPLITS tables.

## Corrected / confirmed shape table

Checkpoint tensors are BF16; dense linears are converted to FP8 E4M3 at load by the
`--qwen4-exp-dense-fp8 shared_expert,attn,linear_attn,lm_head,mtp_dense` arm. `w8a16_gemv` always
receives an `[N, K]` row-contiguous weight, grid = `(ceil(N/BLOCK_N), SPLITS)`.

| Trace grid | Doc claim (N x K) | Actual checkpoint tensor(s) | Fused runtime N x K | Grid math (BLOCK_N / SPLITS) | Verdict |
|---|---|---|---|---|---|
| 256 | GDN `in_proj_qkvz` 16384 x 2560 | `in_proj_qkv [10240,2560]` + `in_proj_z [6144,2560]` (layers 0.. etc.) | 16384 x 2560 | 16384/64 = 256, splits 1 (`_BY_SHAPE_M16` 64x128) | **Correct** |
| 40 x 4 | GDN `out_proj` 2560 x 6144 | `out_proj [2560,6144]` | 2560 x 6144 (N=2560 out, K=6144 in) | 2560/64 = 40, splits 4 (`_BY_SHAPE_M32` 64x128) | **Correct** |
| 416 | attn `q/k/v` 13312 x 2560 | `q [12288,2560]`, `k [512,2560]`, `v [512,2560]` | 13312 x 2560 | 13312/32 = 416, splits 1 (`_BY_SHAPE_M16` 32x128) | **Correct** (per-head guess 7552 wrong) |
| 3 | GDN `in_proj_ba` 96 x 2560 | `in_proj_b [48,2560]` + `in_proj_a [48,2560]` | 96 x 2560 (order `[b; a]`) | 96/32 = 3, splits 1 (`_BY_SHAPE` bf16 32x512) | **Correct** |
| 1940 | `lm_head` 248320 x 2560 | `lm_head.weight [248320,2560]` | 248320 x 2560 | 248320/128 = 1940, splits 1 (`_BY_SHAPE` 128x256) | **Correct** |

Config cross-check (`text_config`): `hidden_size=2560`, `linear_num_key_heads=16`,
`linear_key_head_dim=128` => key_dim 2048; `linear_num_value_heads=48`,
`linear_value_head_dim=128` => value_dim 6144. So `in_proj_qkv = 2*2048+6144 = 10240`,
`in_proj_z = 6144` => qkvz 16384; `in_proj_a/b = 48` each => ba 96; GDN `out_proj`
`[value_dim, hidden] = [2560,6144]`. `num_attention_heads=24`, `head_dim=256`,
`num_key_value_heads=2` gives q 12288 / k 512 / v 512; `vocab_size=248320` gives lm_head.

**Fusion is loader-side, not in the checkpoint:** the index has no `in_proj_qkvz`, `in_proj_ba`,
or `qkv_proj` keys. `qwen4_exp.py:2505-2517` maps `in_proj_qkv (0,1,2)` + `in_proj_z (3)`
into `in_proj_qkvz`, `in_proj_b (0)` + `in_proj_a (1)` into `in_proj_ba`, and q/k/v into
`qkv_proj`; `finalize_fused_in_proj()` is called at `qwen4_exp.py:2839-2841`.

## Trace re-aggregation by grid (exact call counts)

| Grid | c1 calls / 40 steps | c1 calls/step | c1 avg us | c4 calls / 40 steps | c4 calls/step | c4 avg us |
|---|---:|---:|---:|---:|---:|---:|
| 256 | 1440 | 36.0 | 29.78 | 1440 | 36.0 | 29.63 |
| 3 | 1440 | 36.0 | 10.22 | 1440 | 36.0 | 10.72 |
| 40 x 4 | 1440 | 36.0 | 14.62 | 1440 | 36.0 | 16.07 |
| 416 | 668 | 16.7 | 25.38 | 600 | 15.0 | 26.62 |
| 1940 | 40 | 1.0 | 400.4 | 40 | 1.0 | 401.6 |
| 80 x 10 | 2108 | 52.7 | 5.55 | 2040 | 51.0 | 5.47 |
| 32 x 10 | 2108 | 52.7 | 4.14 | 2040 | 51.0 | 4.29 |
| 80 x 1 | 1960 | 49.0 | 4.22 | 1960 | 49.0 | 4.45 |
| 80 x 6 / 160 x 6 / 160 x 3 | 520/148/148 | 13.0/3.7/3.7 | 12.9/10.7/6.6 | 600/0/0 | 15.0 | 13.4 |

This confirms the doc's per-step counts and timings. Note: the trace header says
`record_shapes: 1`, but GPU-kernel `args` contain only grid/block/stream, **no tensor shapes**;
shape evidence here is from headers + planner, not from recorded kernel shapes.
`_w4a16_nvfp4_gemv_kernel` appears with grids 1024 (c1) / 512 (c4) and 768 (both) — these are
**not** qkvz/qkv/out_proj; those dense linears ran through `_w8a16_gemv_kernel` (FP8).

## Which of the three optimizations is feasible given the real layouts

1. **Fuse grid-3 into grid-256 (N 16480): mathematically valid, but blocked in the measured arm.**
   Both weights are `[*, 2560]` and consume the same `hidden_states`, so stacking is legal; the
   code already implements it in `qwen3_5.py:625-653` (`finalize_fused_in_proj`) and uses it in
   `_forward_input_proj` (`qwen3_5.py:655-678`). However it is gated on **both** linears being
   `UnquantizedLinearMethod` + BF16 (`qwen3_5.py:640-648`). With
   `--qwen4-exp-dense-fp8 ...,linear_attn,...`, `in_proj_qkvz` is FP8 (category mapping only
   covers `in_proj_qkvz`/`out_proj`, `qwen4_exp_dense_fp8.py:82-84`) while `in_proj_ba` stays BF16,
   so the fusion is silently skipped and two launches remain. The existing direct-layout path
   (`SGLANG_GDN_PROJ_DIRECT_LAYOUT=1` in `serve-fast.sh:256`, `qwen3_5.py:864-935`) already overlaps
   the 3-CTA ba GEMV with the qkvz GEMV on an alt stream and writes b/a into two destinations via
   `split_n=48`; it does **not** merge them into one launch. Making one launch requires either
   dropping FP8 on qkvz (bf16 path) or adding `in_proj_ba` to the FP8 category and a fused
   FP8 weight loader — a code change, not a config flip. Split-K for ba alone was measured
   *slower* (9.2 us vs 7.9 us; `w8a16_gemv.py:453-455`), so that sub-option is not viable.

2. **Tune `out_proj` split-K: feasible in principle, little headroom in the observed buckets.**
   Real layout `[N=2560, K=6144]`, FP8; K is divisible by 128/256, so split-K/BLOCK_K tuning is
   well-defined. But the planner already pins `(64,128,4)` for M=32 (`_BY_SHAPE_M32` line 478,
   14.3 us vs ~18 us fallback) and `(16,256,6)` for M=1 (`_BY_SHAPE_M1` line 486); the trace's
   `(40,4)` and `(160,6)` grids are exactly those pinned tiles. The doc's "M=16 verify rows"
   target does not appear as an out_proj grid in these traces. Also `out_proj` is deliberately
   excluded from NVFP4 packing (`nvfp4_dense.py:327-329`, measured 1.08-1.13x slower), so tuning
   is the only lever there — but it is a micro-tuning of an already-tuned shape.

3. **NVFP4 packing via `SGLANG_LINEAR_ATTN_NVFP4` / `SGLANG_ATTN_NVFP4`: feasible — this is the
   only ready-to-use switch.** `nvfp4_dense.py:311-320` maps the flags to `in_proj_qkvz` and
   `qkv_proj`; `maybe_pack_dense_nvfp4` is called from `Fp8LinearMethod.process_weights_after_loading`
   (`fp8.py:857-860`) **before** FP8 quantization, so it sees the true fused BF16 tensors. Both
   have K=2560, a multiple of the 16-element FP4 scale block (`nvfp4_dense.py:348-350`), and N
   has no constraint; it installs `NVFP4DenseLinearMethod` and the `_w4a16_nvfp4_gemv_kernel`.
   Neither flag is set by `serve-fast.sh` or present in the traced arm, so this optimization is
   **not yet active** — that is why qkvz/qkv show up as W8A16. Caveats: `NVFP4DenseLinearMethod`
   has no `apply_into_split`, so `_direct_layout_ok()` (`qwen3_5.py:804-805`) turns false and the
   two-destination qkvz/ba overlap is lost; and `out_proj` is explicitly not packable by this
   hook. Feasibility is proven by the code path, but the end-to-end win still needs a shape-recorded microbench (the doc's own caveat).

**Bottom line:** only #3 is directly feasible with existing switches and the real layouts.
#1 needs a dtype/category change (bf16-vs-fp8 mismatch), and #2 is already tuned at the M
buckets present in these traces. No shape in the doc is wrong; the q=7552 hypothesis is wrong.

## Shapes flagged as wrong / at risk

- Task-prompt hypothesis `q/k/v each [7552,2560]` — **wrong**; actual 12288/512/512, fused 13312.
- Doc "grid 416 = 17 calls/step" — exact trace is 668/40 = 16.7 (c1) and 600/40 = 15.0 (c4);
  rounding is fine for c1, slightly stale for c4.
- Doc "record-shapes was NOT enabled" — trace header says `record_shapes: 1`, but no GPU-kernel
  shape payloads are present; inference method remains valid.
- None of the five N x K shapes in the doc require correction.
