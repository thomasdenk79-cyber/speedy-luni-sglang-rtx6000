# Dense NVFP4 with integration fixes — verdict (2026-11-25)

Contents: [setup](#setup) · [results](#results) · [interpretation](#interpretation) · [verdict](#verdict)

## Setup

Single candidate: commit `64e0fbe` on top of the qualified dense arm
`donor-nvfp4-dense-linear-20261125T063614Z`. Two integration gaps fixed:

- `apply_into_split` was missing, so `_direct_layout_ok()` disabled the
  fused direct-projection path (dual-stream overlap, split into mixed_qkv/z/b/a).
- The qkvz shape gate compared `weight.shape[-1]` (K/2 for packed NVFP4) to
  the output dimension; now uses `nvfp4_wq.shape[0]`.

Evidence: `donor-nvfp4-dense-fixed-20261125T075405Z`, engine `4fd47b8`,
status `qualified`, 39 artifacts, no fallback. Env identical to the earlier
dense arm except the fixes; server stayed on `:8001`.

## Results

| Metric | Baseline | Dense fixed | Delta |
|---|---:|---:|---:|
| GSM8K | 242/250 | 242/250 | 0 |
| MMLU | 218/250 | 218/250 | 0 |
| KV capacity | 866 496 | 947 392 | **+9.3 %** |
| Cold prefill, 8K–64K (tok/s) | 14 939–15 528 | 11 864–12 750 | **−19.0 %** |
| BB prefill `pp_tps`, 8K–64K | 14 759–15 176 | 11 823–12 632 | −19.5 % |
| Combined decode median | 402–408 | 405/419 | ~0 |
| BB c8 aggregate, all repeats | 824.7 / 833.8 / 824.8 | 667.9 / 803.1 / 812.4 | median −2.6 % |
| TTFT p50 | 151–158 ms | 205 / 319 / (boost) | bimodal |
| Single-call total, all categories | 1163 ms | 1173 ms | +0.8 % |

The earlier dense arm's GSM8K/MMLU delta (238/224) did **not** reproduce: the
accuracy loss came from the broken fallback path, not NVFP4 numerics.

## Interpretation

- Decode is **not** regressed. The c8 aggregate drop in individual repeats is
  prefill-sensitive: `aggregate_tps` includes prefill time at high concurrency.
  Comparing medians over all three repeats gives only −2.6 %.
- TTFT is bimodal across repeats (baseline 151–158 ms; dense either ~200 ms
  or 307–341 ms). A single repeat cannot be compared. Report medians and
  `pp_tps` alongside `aggregate_tps`.
- The real regression is **cold-prefill throughput at −19 %**, consistent
  between the isolated cold-prefill harness and BetterBench `pp_tps`. This is
  the NVFP4→BF16 dequant + `torch.mm` large-M path; the FP8 baseline uses an
  FP8 GEMM.
- For short agent/chat prompts the end-to-end call time is nearly unchanged
  (+0.8 %). For long-context calls (32K–64K input) the −19 % prefill dominates
  and total latency rises roughly 15–20 %.

## Verdict

Qualified as a correctness result: +9.3 % KV at baseline accuracy and baseline
decode. **Not** a free speed win on prefill. Next candidate is P0.2
(`SGLANG_NVFP4_DENSE_PREFILL_FP8=1`, NVFP4→FP8 e4m3 + `fp8_scaled_mm`), which
targets exactly the −19 % prefill gap; switch-gated and still GPU-unvalidated
as of this note.
