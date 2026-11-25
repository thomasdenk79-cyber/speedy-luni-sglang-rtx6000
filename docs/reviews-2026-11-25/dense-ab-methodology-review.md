# Dense-NVFP4 A/B methodology review (read-only)

**Question:** is `SGLANG_LINEAR_ATTN_NVFP4=1` a −13..−18 % regression, based on
`donor-nvfp4-cake-sm120-20261125T014250Z` (baseline) vs
`donor-nvfp4-dense-linear-20261125T063614Z` (dense)?
**Verdict: CONFOUNDED.** The direction (flag is not a win) is probably right, but the
−18 % headline is not a clean apples-to-apples number. See top 3 confounders below.

## (a) Env / serving flags

All three arms have **byte-identical `argv`** in `plan.json`: `mem-fraction-static 0.96`
then `0.886778242678`, `max-running-requests 4` then `8`, `chunked-prefill-size 8192`,
`speculative-num-steps 3` then `15`, `num-draft-tokens 4` then `16`, adaptive spec config
`w16_c4.json`, `kv-cache-dtype nvfp4`, no HiCache, TP1, context 262144. No power-limit or
clock-lock command recorded in either arm.

Env diff is exactly **two variables**:

| arm | only-here env |
|---|---|
| baseline | `SGLANG_DONOR_SM120_TRACE=1` |
| dense | `SGLANG_LINEAR_ATTN_NVFP4=1` |

`SGLANG_DONOR_SM120_TRACE` only logs at most 96 unique `(rows, capture, route)` tuples
(`moe_runner/flashinfer_cutlass.py:41-52`) — negligible overhead; it does **not** explain
the result in either direction. `launcher.patch` differs (baseline empty; dense/cl-base
add inert `LAB_FAST_LOAD_FORMAT` parsing; env not set → no effect).

**Material derived difference:** baseline `max_total_num_tokens=866496`,
`available_gpu_mem=7.45 GB`; dense `947392`, `7.65 GB` (+9.3 % KV pool, from packing
freeing weights — same nominal mem-fraction). Serving state is therefore not identical.

GPU telemetry recorded in BB `results.json` (clock / temp-ish field / power):

| run | baseline | dense |
|---|---|---|
| 1 | 2760 MHz / 46.08 / 482.37 W | 2812 MHz / 46.80 / 398.18 W |
| 2 | 2827 / 75.15 / 387.66 W | 2835 / 76.82 / 373.41 W |
| 3 | 2827 / 76.20 / 384.44 W | 2842 / 75.63 / **290.14 W** |

## (b) Code identity

`source-commit.txt` = `f9bf31ad…` for all three; `engine_sha` matches; `tool_sha`
`38992a77…` for baseline and cl-base. **But** the dense arm has no
`benchmark-identity.json`, so its tool/dirty state is unrecorded; baseline
`source_repo.dirty_count=2`, cl-base `=3`. Same HEAD is *not* the same effective tree.

Decisive: commit `64e0fbe` (08:38, after dense BB ended 08:24) fixes two bugs in the
measured code: (1) `NVFP4DenseLinearMethod.apply_into_split` is missing, so
`_direct_layout_ok()` disables the fused direct-projection path **and dual-stream
overlap**, forcing a split/reshape/cat fallback; (2) the qkvz shape gate compares packed
`weight.shape[-1]` (K/2) to the output dimension. The benchmark measured the **pre-fix,
degraded fallback path**, not the intended NVFP4 path.

Flag-firing check (sceptical in the other direction): dense log shows 36
`in_proj_qkvz … packed` lines at layers 0,1,2,4,5,6,…,46 — exactly the expected 3-of-4
GDN layer pattern. So this is **not** a partial-application confounder. The 98
`has no .prefix` warnings are other module families; the `CP delta rule … falling back to
non-CP` warning appears in **both** arms, so it does not differentiate them.

## (c) Warmup / order / fallback

Runs were sequential, not interleaved: baseline 02:42–03:55 → cl-base 06:06–06:12
(startup-only) → dense 07:36–08:37+. The established N1 A/B method explicitly
interleaves off/on pairs because single pairs confound drift; this A/B did not.

The dense arm is **incomplete**: `test.log` stops at `MODE START agent`; no
`summary.json`, no `benchmark-identity.json`, no telemetry mode. It was never qualified.

Within-arm drift is the largest measurement problem:

| BB run | baseline c8 t/s | dense c8 t/s | delta | dense TTFT p50 c8 |
|---|---:|---:|---:|---:|
| 1 (cold) | 824.7 | 807.1 | **−2.1 %** | 198 ms |
| 2 | 833.8 | 672.9 | −19.3 % | 310 ms |
| 3 | 824.8 | 657.6 | −20.3 % | 341 ms |
| **median (reported)** | **824.8** | **672.9** | **−18.4 %** | — |

Baseline is flat (±1 %) across repeats; dense collapses monotonically while its power
draw falls 398→290 W and TTFT grows 198→341 ms. Speculative acceptance
(`tokens_per_update ≈ 3.0`) is unchanged, so the collapse is prefill/queue-side, not a
uniform decode penalty. The reported −18 % is simply dense run 2.

Counter-evidence that some regression is real: cold-prefill (12 unique-prefix repeats,
before BB) is stably slower in dense at every depth — 8000: 14939→12043 (−19.4 %),
16000: 15069→12055 (−20.0 %), 32000: 15528→12858 (−17.2 %), 64000: 15525→12906
(−16.9 %). BB's own prefill sweep agrees (64k: ~15000 vs ~12600 in *all three* runs).
So a genuine prefill-path cost of roughly −16…−20 % is supported; the extra BB decode
collapse is not.

## (d) Accuracy

gsm8k 96.8 % → 95.2 % (−1.6 pp); mmlu 87.2 % → 89.6 % (**+2.4 pp**, dense is the best
of every arm in the cache). Across comparable baseline arms the empirical spread is
gsm8k 94.8–97.6 % and mmlu 84.8–89.2 %. With n=250, the 1σ SE of the difference is
≈1.8 pp (gsm8k) and ≈2.9 pp (mmlu): both differences are <1σ, i.e. within noise.
Accuracy gives **no** support for a quality regression.

## (e) Session provenance

The quoted BB medians match the baseline arm's own 3-run medians exactly (verified by
recomputing from `bb-025948/run-{1,2,3}/results.json`); they did not come from
`donor-nvfp4-best-profile-20261125T041040Z`, which has profiler enabled and only
doctor/probe (no BB). The startup-only `donor-cl-base-20261125T050612Z` is a different
session (06:06, ready after 350 s) from best-profile (05:10, 340 s); the quoted "338 s"
does not match either log. So there is no cross-session mixing of the BB numbers, but
the startup comparison does use different sessions.

## Top 3 confounders

1. **Within-arm run drift / thermal-power state.** Dense alone degrades across BB
   repeats (−2 % → −19 % → −20 % at c8) with power 398→290 W and TTFT +70 %; the median
   is just the degraded middle run. Baseline is stable. This inflates −18 % by roughly
   12–16 pp.
2. **Pre-fix code at f9bf31a.** The measured arm lacked `apply_into_split` and had the
   broken qkvz shape gate (both fixed in `64e0fbe` after the run). The intended fused,
   dual-stream NVFP4 path was disabled — the arm measures flag + two integration bugs.
3. **Not interleaved + non-identical derived config.** Arms ran ~5 h apart, sequential;
   KV pool differs 866496 vs 947392 tokens; GPU clock/power at start differs; dense run
   never produced identity/summary/telemetry and the working-tree dirty counts differ.

## Bottom line / what would be needed

- Do **not** report "−18 % regression" as a property of `SGLANG_LINEAR_ATTN_NVFP4=1`.
- Defensible statement: *there is a stable ~16–20 % cold-prefill penalty and a small
  (~2 %) cold decode penalty; the large BB median gap is confounded by within-arm
  drift and by pre-fix fallback code.*
- Required re-run: cherry-pick `64e0fbe` (or later), then interleave off/on/off/on pairs
  in one GPU lock/thermal window with identical derived KV size, and qualify both arms
  (identity + summary). Use first-run/cold numbers as well as medians and report the
  per-run trend, not just the median.
