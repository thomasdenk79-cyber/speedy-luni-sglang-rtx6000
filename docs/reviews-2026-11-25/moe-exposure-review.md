# Skeptical review of the "MoE exposure" explanation for the NVFP4 dense step-C result

Read-only. No repo edits, no GPU, port 8001 untouched. Sources: `N1_LOG.md` §7/§11/§13,
`EXCLUSIVE_TIME_MAP.md` §1/§5/§5.3, `prof/exclusive_time.py`, CUPTI traces
`profile/c1-20261125T041807Z` and `profile/c4-20261125T041823Z` (40 steps each),
`ab-night/donor-nvfp4-dense-linear-20261125T063614Z`, `env-compare.txt`,
`sglang-donor-nvfp4kv/.../models/qwen3_5.py`, `/tmp/opencode/analysis/w8a16-shapes.md`.

## Verdict: (c) the MoE-exposure explanation is likely wrong

The "covering" story is conceptually inverted, is contradicted by the retained traces,
and rests on numbers that have no trace provenance in the surviving artifacts. The real
code-backed mechanism is loss of the GDN dual-stream/direct-layout fusion, and the large
E2E regression is dominated by run-to-run session drift plus a spec-decode/acceptance
effect, not by MoE GEMMs becoming exposed.

## 1. Is CUPTI "exclusive time" a coverage metric? No.

`EXCLUSIVE_TIME_MAP.md` §1 and `exclusive_time.py:11-21, 212-241` define it precisely:

```
exclusive(k) = wall time during which k is the ONLY GPU-resident kernel   (deletion metric)
sum(exclusive) + shared(N>=2) + gap(no kernel) == wall                     (identity)
```

Overlap time is charged to *nobody*. A kernel with **low** exclusive fraction is the one
that is covered; a kernel with **high** exclusive fraction is the one running alone.
`N1_LOG.md` §418 says qkvz "went from 88.5 % exclusive to 75.4 % — the NVFP4 kernel
overlaps more — so the MoE grouped GEMMs ... became exposed." That is backwards: qkvz
being *less* exclusive in the NVFP4 arm cannot make a *different* kernel more exposed
unless the two actually overlap — and, per the identity, making one kernel shorter with
a flat wall simply moves microseconds between `exclusive`, `shared`, and `gap`. In
`N1_LOG` §13 the family deltas sum to -459+348+171+115-175 = **0 µs**, i.e. the +348/+171
are a *partition consequence* of a flat wall, not independent evidence of added work.

The same document that N1 cites (`EXCLUSIVE_TIME_MAP.md` §5.1/§0) puts the MoE grouped
GEMMs at **96-99 % exclusive** in both arms — they are essentially never overlapped with
anything, so they could not have been "hidden behind" qkvz in the first place.

## 2. Do qkvz and the MoE GEMMs share a stream / overlap? No.

Decoded all kernel events (`cat=kernel`, `ph=X`, `args.stream`, `args.graph`) of both traces:

| check | c1 (1 gen) | c4 (4 gens) |
|---|---:|---:|
| qkvz GEMV (`_w8a16_gemv_kernel`, grid `[256,1,1]`) instances | 1440 | 1440 |
| MoE GEMM (`nvfp4_swiglu_moe::persistent_main`) instances | 1920 | 2040 |
| **overlapping qkvz↔MoE interval pairs** | **0** | **0** |
| overlapping qkvz↔`in_proj_ba` (grid `[3,1,1]`) pairs | 1440 (all) | 1440 (all) |

- Inside the CUDA graph, qkvz (node ~`1168231104544`) and the MoE prologue/persistent
  pair (nodes `...4562/4563`) are strictly sequential within each layer; MoE starts
  ~100 µs after qkvz. They are assigned to the graph's three streams (ids 13/339/340,
  rotating across layers) but are joined by data dependencies and are **never resident
  together**. So no stream structure could let qkvz "cover" them.
- qkvz and `in_proj_ba` **are** on different streams at the same launch timestamp
  (e.g. stream 339 vs 340) and overlap ~10-15 µs per call — that is the *only* dual-stream
  fork involving qkvz, and `ba` is a 3-CTA ~10 µs kernel, not a MoE GEMM.

This is the Nov-25 FP8-arm build (SM120 `nvfp4_swiglu_moe`); the A/B that produced
`cutlass_moe_grouped_gemm1/2` +348/+171 (Sep-06, W4, `qual.sh`) has **no retained trace**
under the named directories, so the exact decomposition cannot be recomputed. But the
structural conclusion (no qkvz↔MoE overlap) is robust for the same dependency chain.

## 3. Are the quoted 88.5 % / 75.4 % even about qkvz?

`75.4` appears exactly once in the retained logs — in the sentence making the claim
(`N1_LOG.md:419`); it is not a reproducible trace value. `88.5 %` in `N1_LOG.md:218` is
the **target lm_head's %-of-roof** for the NVFP4 kernel, not qkvz exclusivity; qkvz's
bench exclusivity was 94.9 %→77.4 % (M=1, §7). The headline quantitative premise of
claim (1) therefore has no auditable source.

## 4. Could +348/+171 be a profiling/zipf artifact? Yes, several ways.

1. **Partition identity** (above): flat wall + one family losing 459 µs forces the rest to gain; this is accounting, not causation.
2. **Workload/routing sensitivity**: MoE GEMM time is ∝ distinct experts D (`EXCLUSIVE_TIME_MAP.md` §5.1, 27 at T=4 / 70 at T=16). The stage-C arm also packs `attn qkv`, changing draft-token logits and therefore spec-decode acceptance and the routed-token/zipf mix. Different D changes MoE *duration* and its overlap partners with no "coverage" mechanism at all.
3. **Graph-structure change**: with the NVFP4 method installed, `_direct_layout_ok()` goes false (§5 below), adding a launch and changing fork/join placement, which shifts which neighbours overlap MoE and therefore its exclusive partition.
4. **Noise**: `N1_LOG.md` §11 itself says whole-step wall "moved ±5 % round to round"; §13 is a single off/on pair. A ±5 % measurement cannot resolve a 0.5 % step change into causal family attributions.

## 5. What can actually make a faster qkvz kernel slow the whole step?

- **Real, code-proven:** `NVFP4DenseLinearMethod` has no `apply_into_split`, so
  `_direct_layout_ok()` (`qwen3_5.py:804-805`) is false. The whole direct-layout path
  (`qwen3_5.py:872-943`) and its `front_gate_stream` overlap are bypassed, and a separate
  `fused_qkvzba_split_reshape_cat` launch is added (**+115 µs** in §13).
  `w8a16-shapes.md:100-103` states exactly this: "the two-destination qkvz/ba overlap is
  lost." This is a genuine structural regression, and it is fixable; it is *not* a MoE
  kernel issue.
- **Real, not measured here:** logits from NVFP4 qkvz/qkv change speculative acceptance.
  The arms run `--speculative-adaptive` with `w16_c4.json` (draft steps up to 15); even a
  small accuracy shift moves tokens-per-verify and E2E t/s far more than 0.46 ms/step can
  move a 10.5 ms step. This is the most plausible route to a *double-digit* E2E effect.
- **Measurement drift (dominant for the quoted -18 %):** the dense arm BB c=8 aggregate
  t/s was 807.1 / 672.9 / 657.6 across three runs (07:55-08:24), while the baseline was
  824.7 / 824.8 / 833.8. The "−18 %" is the median of a declining sequence: **run 1 was
  only −2 %**, runs 2-3 collapsed. `EXCLUSIVE_TIME_MAP.md` §1 and `OVERHEAD_REPORT.md` F4
  document Max-Q power-cap/thermal throttling and 0.5-1.5 ms preemption spikes making
  this box's absolute numbers session-dependent (22.48 ms vs 17.32 ms for the same W16
  config). A 1.53× kernel on a 1.07 ms family can save at most ~0.46 ms ≈ 4 % of a
  10.5 ms step — it cannot, by itself, produce −18 % E2E.
- Secondary: per-request caching/acceptance mix differences (`kv_used_max` 52608 vs 66304;
  probe tok/s shifted ~20 % at 07:44 before BB started) confound the comparison further.

## 6. Is there a multi-stream structure in decode that could explain "covering"?

Yes, but only for qkvz∥ba, never qkvz∥MoE:

- `qwen3_5.py:680-714` forks `in_proj_qkvz` (main) and `in_proj_ba` (alt) inside CUDA-graph
  capture when `num_tokens < 1024`; `:918-934` runs the direct-layout variant, with
  `_GDN_FRONT_OVERLAP=1` putting qkvz on the alt stream and joining the HC combine gate
  (`front_gate_stream`, `:856-870`). Trace: 1440/1440 qkvz↔ba overlaps, 0 qkvz↔MoE.
- `EXCLUSIVE_TIME_MAP.md` §5.3 names three deliberate dual-stream sites: QSA indexer ∥ QKV
  projection, GDN qkvz ∥ ba, shared expert ∥ router+routed experts. Verify-step *k* ∥
  draft-step *k+1* is ruled out on dependency grounds; MTP `plan_stream` / `forward_stream`
  (`eagle_worker_v2.py:2819`, `model_runner.py:621`) do not create a qkvz↔MoE fork.
- Both c1 and c4 (1 and 4 concurrent generations) show the same zero qkvz↔MoE overlap.

## Bottom line

Do not use "MoE exposure" to explain the stage-C result. Recheck with: a retained CUPTI
A/B in the *same* session/power state; per-kernel medians plus MoE distinct-expert counts
and accepted-tokens-per-step; `apply_into_split` restored; headless/interleaved ABBA. Until
then the honest summary is: a 1.53× qkvz kernel was bought at the price of losing the
direct-layout two-destination fusion (+115 µs), and the −18 % E2E figure is largely a
session-drift/acceptance artifact, not a GPU critical-path exposure effect.
