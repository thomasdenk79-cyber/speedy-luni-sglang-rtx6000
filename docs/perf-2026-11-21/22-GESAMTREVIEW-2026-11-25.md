# Overall review 2026-11-25 — NVFP4/SM120 base, patch plan

> **Owner-direction update (2026-11-25):** For future work, follow [24 — Upstream-first SM120 / NVFP4 handover](24-UPSTREAM-FIRST-SM120-HANDOVER-2026-11-25.md). The donor remains the frozen LKG, official upstream becomes the intended next main engine, P0.2/P0.2b are diagnostic only, and native NVFP4 plus controlled tokenmap ablations take precedence. The measured historical results below remain unchanged. GPU/operator approvals still apply.

## Contents
1. Scope and priority
2. Corrected interpretation of the dense-NVFP4 arm
3. Consolidated review findings (with severity)
4. Upstream / stack strategy
5. Ranked patch plan
6. Evidence standard and gates
7. Handover adoption (p16)
8. Open items and next steps

## 1. Scope and priority

Owner priority: **speed > quality > VRAM > RAM > SSD**. HiCache/NIXL is a later topic and is
not planned here. Goal is a future-proof base: take SGLang/FlashInfer/Triton upstream
(including nightly) where it gives an elegant SM120 solution, then re-apply our own
improvements on top. Full NVFP4 support is wanted **if the summed benefit justifies keeping
it**: NVFP4 is a sweet spot for decode speed and VRAM, quality deltas so far are noise, but
professionals often leave BF16/FP8 in dense paths — for lack of a kernel or because of
measured quality loss. Both reasons are checked explicitly below.

KV cache is already well optimized for c4 operation (NVFP4 KV + Cake decode, 866496 tokens)
and is not part of this patch plan.

## 2. Corrected interpretation of the dense-NVFP4 arm

Arm `donor-nvfp4-dense-linear-20261125T063614Z` (`SGLANG_LINEAR_ATTN_NVFP4=1`, run tree
`f9bf31a`, full benchmark set) vs best arm `donor-nvfp4-cake-sm120-20261125T014250Z`:

| Metric | Best arm | Dense arm | Corrected reading |
|---|---:|---:|---|
| BB c1 / c8 | 346.5 / 824.8 | 340.6 / 672.9 | c8 headline -18 % is **drift**, not a stable effect |
| BB c8 single runs | 825 / 825 / 834 | 807 / 673 / 658 | run 1 only -2 %; runs 2/3 degraded with power 398->290 W, TTFT 198->341 ms |
| Cold prefill 8k..64k | 14939..15525 | 12043..12906 | **stable -17..-20 % at every depth** — the real regression |
| Cold decode | — | ~-2 % | within noise |
| GSM8K / MMLU | 96.8 / 87.2 | 95.2 / 89.6 | within noise (MMLU +2.4 pp is dense arm's best result) |
| KV tokens | 866496 | 947392 | +9.3 % capacity (real, but not useful under context_len 262144) |

**Verdict on the arm:** the flag is not a win *as integrated*, but the -18 % headline is
confounded. The one stable, real regression is cold prefill. The first "verdict: reject"
was premature; the arm must be re-measured with the integration fixes (below).

## 3. Consolidated review findings (with severity)

| ID | Finding | Severity | Status |
|---|---|---|---|
| A | `NVFP4DenseLinearMethod` had no `apply_into_split`; `_direct_layout_ok()` false, fused direct projection + dual stream + `front_gate_stream` disabled. Measured fallback +115 us/step (~1.2 %) and ~3.2 us x 36 layers (~1.3 %). | Major (speed, small) | **Fixed**, commit `64e0fbe` in feature worktree |
| B | Packed `in_proj_qkvz.weight` is `[N,K/2]` = `[16384,1280]`; shape gate `weight.shape[-1]==16384` fails independently. `2*key+2*value = 16384 = nvfp4_wq.shape[0]`, `split_n = 10240`. | Major (blocks A) | **Fixed**, `64e0fbe` |
| C | `_wide_m` (large M) dequantizes NVFP4 to BF16 and calls `torch.mm`; FP8 baseline uses a native FP8 GEMM (2x BF16 compute). This is the only prefill regression source. | **Blocker for speed** | Open — top patch item |
| D | "MoE GEMMs got exposed by the faster qkvz GEMV" is **wrong**: CUPTI exclusivity interpretation was inverted; 0 overlapping qkvz/MoE intervals in 1440 instances; MoE 96-99 % exclusive in both arms; the quoted 88.5/75.4 % have no trace provenance. | Informational | Discard explanation |
| E | Within-arm thermal/session drift (power 398->290 W, TTFT 198->341 ms) inflates BB deltas by ~12-16 pp. Arms were sequential, ~5 h apart, not interleaved. | Methodological | Fix protocol |
| F | Accuracy deltas are within noise; no quality regression measured. BF16 source at the loader hook avoids FP8->NVFP4 compounding (~2 argmax points). | Informational | Keep BF16 source |
| G | `sglang-kernel 0.4.11` pins torch 2.14.1; 0.4.6.post1 has c10::cuda error on 2.14.1; flashinfer 0.7.5 ships no `csrc/*.cu`; current stack 0.6.17 / 0.4.6.post1 / torch 2.13.0 is the only clean path. | Blocker for upgrade | Do not upgrade in place |

## 4. Upstream / stack strategy

- **No rebase:** `git merge-base 8edd1fa upstream/main` is empty. Cherry-pick/port only.
- **Do not upgrade FlashInfer in place** (finding G). Our 5-file Cake overlay is byte-identical
  to 0.7.5, so it stays necessary either way. `tools/assemble_flashinfer.py` hard-codes
  version 0.6.17 with a hash-validated manifest; any upgrade means regenerating it and
  finishing the stalled 9-conflict rebase in `/tmp/opencode/fi-rebase` first. Hold.
- **Selective correctness ports** (quality first): `0f1509829b` NVFP4 TRT-LLM hidden padding
  to 256/512 (otherwise silently wrong outputs); GDN fixes `03d4f49f22`, `ff68baae6d`,
  `077acd49a1`, `304a6016b3`. Do **not** take bundle `775db052b6` — it was reverted by
  `3981de0825` on main. Drop `a8ac86595c` (Beam-only) and draft PRs #51051-#51068 (unmerged).
- **If a FlashInfer rebase is ever resumed:** reject-not-corrupt guards for FI #9555
  (N=64, M=1..32 silently garbage) and #9556 (M<128) must reject before launch, never pad or
  fall back to cutile. Zero conflict markers, sha-matched manifest and private caches are
  acceptance gates. This is scratch until then.
- **QSA prefill #162/#45791 is already in our branch** — the job is measurement, not porting.
  It targets long cached prefixes (94.9 % of prompt tokens), not cold unique-prefix prefill.
  Test: `max_row_keys=None/0`, cached 50K/150K/200K/500K at C1/C2/C4/C6, 3 repeats.
- **SM120 SwiGLU kernel:** upstream has nothing newer; newest ref `d5a4bc5a35` is byte-identical
  to the donor `.cuh`. Tuning is local. The 512-row cap (`SWIGLU_MAX_TOKENS=512`) sends
  >512-row prefill to general CUTLASS.

## 5. Ranked patch plan

Benefit is stated as an unmeasured target where no A/B exists — these are candidates, not
accepted numbers. Every item is switch-gated, off by default, and must leave the switches-off
build identical to the donor FP8 baseline.

### P0 — restore the base (speed, low risk)

| ID | Change | Expected | Risk / gate |
|---|---|---|---|
| P0.1 | Apply `64e0fbe` (findings A+B) to the run worktree and re-run the dense arm with identical env (`/tmp/opencode/rerun-dense-fixed.sh`). | +1.2..2.5 % of step | Bitwise/CUDA-graph tests; identity check with all switches off |
| P0.2 | `_wide_m`: NVFP4 -> FP8 e4m3 on the fly, then `fp8_scaled_mm` (`fp8_utils.py:1788-1963`, `:205`) behind a new flag. Keep BF16 `torch.mm` as fallback. | Recovers most of the stable -17..-20 % cold prefill | Accuracy gate (GSM8K/MMLU within noise band, <=1 % @ k=7); FP8 quantization adds ~2 mantissa bits loss — measure before shipping. No extra persistent VRAM. |
| P0.3 | SM120 MoE tuning: parameterize `kUnit` (32 -> 16 -> 8) and GEMM2 stages (4 -> 5) via `-D` defines fed from env; mirror Python workspace literals. | `kUnit=32` at M=16 gives mcount 1-2, ~29 % MMA utilization — largest unused gap; 5 stages prefetch all 5 GEMM2 K-chunks at identical smem | Bitwise equality vs `cutlass_fused_moe(use_fused_finalize=False)` incl. skipped experts and CUDA-graph replay; process restart per variant; one measurement method for all |

P0.2 alternative if accuracy or latency misses: keep a second FP8 copy of `in_proj_qkvz`
(~1.9-2.3 GiB for 48-60 GDN layers). Guarantees FP8 prefill speed but trades away the VRAM
advantage — only if VRAM headroom stays acceptable (VRAM is medium priority).

### P1 — quality / correctness (high priority)

| ID | Change | Expected | Risk / gate |
|---|---|---|---|
| P1.1 | Port `0f1509829b` NVFP4 hidden-padding fix. | Removes silent wrong outputs on non-256 dims | Correctness test on affected shapes |
| P1.2 | Port the four granular GDN fixes. | Correctness | Per-fix tests; no bundle |
| P1.3 | QFN guard backport (~150 lines / 4 files): `G_UP_OFF` draft-MoE scale, 0197 `k % 256` guard (both gates), 0200 SM120 check, 0198/0201/0202 bounds. | Robustness; donor already has the 4 functional perf patches | No QFN feature/bench code, no PDL stub, no `sm120_bf16_skinny_gemv` into the W8A16 path |
| P1.4 | Instrument spec-decode acceptance: D counts, accepted tokens/step, logits shift under NVFP4. | Distinguishes a real quality effect from session drift | Same-session CUPTI, not whole-step wall |

### P2 — further speed (after P0 is measured)

| ID | Change | Expected | Risk / gate |
|---|---|---|---|
| P2.1 | Lift the 512-row fused SwiGLU cap with a preallocated workspace buffer (8192 rows -> 446.3 MiB); lower `tune_max_num_tokens`; keep chunked prefill 8192. | Prefill >512 rows currently forced to CUTLASS | Workspace formulas mirrored Python/C++; OOM reproduction at 0.92 mem-fraction; 840 MiB request must come from the general path, not the fused kernel |
| P2.2 | Verify `front_gate_stream` and dual-stream overlap are actually restored after P0.1. | ~1.3 % | Trace `route=` and stream intervals |
| P2.3 | Hyper-connection fusion (12.7 % c1 / 9.6 % c4 of decode). | Low headroom — current gates `HC_GATE_EARLY=2`, `HC_APPLY_MIX_FUSED=1`, `HC_LAYER_APPLY_FUSED=1` already active | Only after P0/P1; expect little |

### P3 — future-proofing (parallel, no GPU)

- Watch `upstream/main` and draft PRs; keep a one-command cherry-pick shortlist.
- Keep the FlashInfer rebase as unresolved scratch with reject-not-corrupt guards; do not
  rebuild during a benchmark.
- p16 (32 threads / 188 GiB, sm_89) does editing, CPU tests, manifest/compile preparation;
  t15 (sm_120) does GPU validation only. Never edit the shared `main` tree from both machines.

### P4 — later (not now)

HiCache `file` + NVFP4 sidecars (K/V E4M3 block scales, MTP draft, packed row size,
write-complete via fsync), pinned host tier + page-cache experiment (note: `drop_caches`
does not evict ZFS ARC), NIXL/GDS only if cold reload is measured to be the bottleneck.

## 6. Evidence standard and gates

- Matched control arm, >=3 repeats, **alternating A/B (ABBA)**, same checkpoint/prompts/power,
  warmup 3, report the spread. An advantage inside normal scatter is **not** a confirmed speedup.
- Separate cold unique-prefix prefill from long cached-prefix prefill; never mix warm-prefix
  and cold-prefill numbers. 1-3 % per server start and +/-5 % wall are the noise floor;
  read CUPTI medians for kernel decisions, not whole-step wall.
- **Missing evidence = HOLD, not a zero delta.** Reject-not-corrupt, fail-closed.
- No `gpu_validated: true` without end-to-end evidence; CPU unit tests alone never release a
  GPU kernel (owner rule). Keep `:30000` down and `:8001` as the only benchmark lane.
- Push to fork `main` needs `LUNI_PUSH_GATE=1 LUNI_PUSH_REASON=...`. Never push from a sparse
  checkout (past incident deleted 3200 tracked files); check `git diff --name-status HEAD
  origin/main | head` and file count before pushing.

## 7. Handover adoption (p16)

Canonical handover: `luni-knowledge-as-code@origin/main:docs/exec-plans/active/review-handover-p16-2026-11-25.md`.
It is not duplicated here (single source of truth). Adopted into this plan:

- 512-row SwiGLU cap and workspace numbers (P2.1)
- FlashInfer guard strategy and private-cache isolation (section 4)
- Evidence standard, HOLD-not-zero, owner gates (section 6)
- p16/t15 work split (P3)

Corrected by our reviews:

- The "MoE exposure" mechanism is wrong (finding D), so it is not a patch item.
- The -18 % BB headline is drift-inflated (finding E); cold prefill is the real regression.
- The dense arm measured flag + two integration bugs (findings A+B), so it is re-measured.

## 8. Open items and next steps

1. Re-run the dense arm with `64e0fbe` (P0.1) after the current arm finishes.
2. Implement P0.2 behind a flag; A/B with accuracy gate.
3. Prepare P0.3 variants as parameterized builds; bitwise tests first, GPU later.
4. Port P1.1/P1.2 (correctness) as separate small commits.
5. Measure #162 cached-prefix prefill on a clean GPU window.
6. Run the switches-off identity check with the full benchmark set.
7. Hold FlashInfer rebase and HiCache until P0/P1 are qualified.

*Status: 2026-11-25. Reviews: `/tmp/opencode/analysis/{dense-integration-review,dense-ab-methodology-review,moe-exposure-review,nvfp4-prefill-paths,upstream-cherry-picks,flashinfer-upgrade-audit,sm120-moe-tuning,qfn-lowrow-portability,w8a16-shapes,qsa-prefill-kernels,hicache-nvfp4-design,nvfp4-dense-gates}.md`.*
