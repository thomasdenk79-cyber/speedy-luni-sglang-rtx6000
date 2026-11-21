# Release train: Flash-Next on SM120

[Baseline](#baseline) | [Patch decisions](#patch-decisions) | [Promotion](#promotion)

## Baseline

Decision on 2026-11-22: **prepare candidates; no default change or release yet**.
The [source matrix](02-PATCH-PORT-MATRIX.md) records the browser review; the
[QFN register](../PATCHES.md) identifies individual commits, flags and tests.
Base here means a Git tree, not proof of what bytes a running image loaded.

| Review claim | Reproducible evidence in `sglang-upstream` | Consequence |
|---|---|---|
| Opus: pending HyperBurst work is not in rc2 | `git rev-list --left-right --count a090492091...4eb9a38b69` = `5 10`; vs `7ee989227b` = `5 20` | Keep GPU status pending. Do not start a new port. |
| The experiment still needs to be assembled from scratch | `git show --stat 11eccd3d7e` includes the kernels, switches, tests and four microbenchmarks. `git log --oneline fe914780fb..a090492091` lists the R10 HiCache clear fix only. | Reuse the experiment on an isolated candidate containing R10. Neither the integration tip nor the dirty rc2 checkout is a release image. |
| NVFP4 host support needs a blanket port | `git merge-base --is-ancestor 47045ecad3 a090492091` succeeds. | Test sidecar fidelity and actual NIXL runtime; ancestry proves neither. |
| The two draft templates are benchmark evidence | `git show fe914780fb:docs/experiments/draft-w4a16.env` and its `draft-w4a4.env` sibling use `REPEAT=1`, point at a worktree and toggle a dispatch flag. | Pin the build and prove the selected kernel; a flag name does not prove a quantization mode. |

`speedy-baseline` records image digest, loaded source SHA/dirty state, model and
tokenmap revisions, launch configuration, harness revision, seeds and power
settings. Abort a comparison if attribution is missing or conditions drift.
Run CPU preparation concurrently. Only the lease holder runs GPU jobs, serially
in the approved window; no restart, cache reset or load test of the serving engine.

## Patch decisions

Each row is a separate comparison against the same preceding accepted base.
Ranges include associated guards and tests; do not strip them for a smaller diff.
Task names omit the common `2026-11-21-` prefix.

| Cut / candidate | Expected benefit and current evidence | Operator owner / deciding check |
|---|---|---|
| R1a: draft MoE QFN-0193/0198/0201/0202 | 49.4 to 29.8 us/call **donor**, not local throughput | `hyperburst-gpu-ab`: trace actual draft dispatch, W4A16 vs baseline, acceptance and end-to-end speed |
| R1b: router QFN-0194/0197/0200 | Low-row speed, not measured here | Same task: biased/unbiased parity, unsupported shapes fall back |
| R1c: GDN norm QFN-0195/0199 | Removes a launch, end-to-end gain unknown | Same task: compare unfused output, eager and captured |
| R1d: MTP FC QFN-0196 | Small-batch decode speed, unmeasured | Same task: verify entry projection at M=1..16 representative shapes |
| R1e: PDL vs singleton pruning QFN-0203..0214 | Launch overlap vs approximate routing, both unmeasured | `speedy-hyperburst-pdl`: first PDL only with tau=0, compare PDL off/on; separately evaluate nonzero tau, never silently ship output drift |
| R2a: NVFP4 KV + host sidecars QFN-0136..0140 | Pool 12.16 to 7.15 GiB **historical**; not a NIXL result | `nvfp4-hicache-nixl`: byte/scales, QSA/Mamba/draft state, partial pages, eviction and restart; fail closed on missing sidecars |
| R2b: async HiCache; anchor clamp #50236; #47621/#48421 separately | Correctness/availability, not tok/s claims | `speedy-port-50236` + `hicache-nonblocking-audit`: reproduce pressure, async backup while warm, never evict last good copy; no stalled active decode |
| R2c: NIXL restore | Correct restore without blocking, no claimed bandwidth win | `speedy-hicache-nixl-ab`: RAM-backed disposable fixture, no SSD experiment; verify backend, failure paths and ITL under concurrent restores |
| R3: residency admission | Less TB4 refill and queue latency; not measured | `sched-tier-admission`: compare FCFS/LPM vs VRAM > genuinely resident host > oldest fitting request, with headroom, aging and no queue head blocking. Staged data counts only when ready. |
| Later speed: QSA #162 vs #45791; vLLM #57097 and #59533 | Donor or other architecture; applicability open | `speedy-pennyroyal-qsa162`, `speedy-vllm-qsa-fusions`: check semantic overlap; isolate each fusion |
| Later attention: FlashInfer #6804 vs #54846; #64199 reference; rc3 split-K and skinny GEMV vs #33 | Kernel speed vs numerical and scratch cost, local benefit unverified | `speedy-nvfp4-flashinfer`: single winner per shape vs current kernel; rc3 is not a drop-in release |
| Later draft: #43/#55, #118 width sweep, QFN-0170..0192 | Controller correctness / smaller head vs added complexity | `speedy-pennyroyal-draft43-55` audits closed ideas; `speedy-agentload-c1-c10` compares widths, fixed vs dynamic map. DFlash2 historical C1 0.883x: keep off; DFly2 research only. |

Hold #38 (FP8 staging), #75 (test semantics, not a closed diff), #49158
(write-back only); #73 is an alignment reference, not an SSD project.
PLE stays packed in pinned host RAM; record pinned allocation, duplicate copies,
reload growth and peak host RAM. Reject any arm that silently switches to SSD.

## Promotion

1. **R1 decision first.** `speedy-release-integration` consumes baseline, quality,
   agent-load, low-row and PDL reports. A completed CPU task or queued GPU job is
   not a passed gate. Mark each arm ship/hold/drop with raw evidence; a held arm
   stays off and need not delay independently proven winners.
2. **Measure one variable.** Record start/end with timezone, base/candidate
   SHAs and flag, alternating A/B with at least three repeats per arm, raw data
   and median/spread. C1/C2/C4/C6 first, C8 for scale; compare C4 ~710 and
   C8 ~750 tok/s as goals only. Include TTFT/ITL p95/p99, accept length,
   quality, KV headroom and peak VRAM/RAM. The primary agent-load KPI is
   [generated tokens per wall-clock hour under matched arrivals including large prefills](10-PREFILL-DECODE-HOUR.md).
   Neither a microbench win nor decode-only tok/s alone can ship a patch.
3. **Independent review.** For speed, require repeatable end-to-end delta beyond
   observed spread with no quality or tail regression outside the predeclared
   tolerances. For correctness, require a failing baseline case and passing
   candidate, including the no-regression load test. Record evidence or HOLD
   in [RESULTS](RESULTS.md). Test the exact combined winner SHA again.
4. **R2 then R3 have separate decisions.** Do not mix scheduler, attention
   kernel and HiCache changes in one experiment. R2 requires sidecar round-trip
   and nonblocking overlap evidence; R3 requires queue/admission traces and
   starvation checks under pressure. No promotion by dependency completion alone.
5. **Rollback before rollout.** Retain known-good image digest, launcher/profile
   and source SHA. On parity failure, wrong-sidecar restore, stalls or unexplained
   regressions, disable the candidate flag or select the previous image/profile
   in the approved lifecycle. No automatic live promotion. Update the QFN register
   only for the exact reviewed and measured candidate.
