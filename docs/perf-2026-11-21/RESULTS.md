# Local release evidence

No campaign A/B run is entered here yet. This is not a claim that no historical
runs exist. Donor references and historical results are in the [roadmap](08-INTEGRATION-ROADMAP.md).

Append one row per isolated arm (including negative results). Link raw logs/JSON,
the reproducible launch manifest and independent review; do not publish secrets
or prompt payloads. Missing evidence means HOLD, not a zero delta.

| Patch / flag | Start / end (TZ) | Base / candidate SHA + image | Load / repeats | Baseline vs candidate, delta + spread | Quality / tails / VRAM / RAM | Raw evidence / review | Decision |
|---|---|---|---|---|---|---|---|

## Baseline reference rows (task `2026-11-21-speedy-baseline`, 2026-11-22)

No local campaign number exists: the operator had no GPU lease (`bin/gpu_window.py status` ->
rc=1) and the owner approval `engine-hold-lift-rc2-2026-11-21` assigns `gpuq add/run` to the
owner, so no arm was measured. Rows below are **reference points only**; a future run replaces
them, it does not edit them.

| Metric | Reference point | Class | Spread | Evidence | Source condition |
|---|---|---|---|---|---|
| decode tps C1 | 369.9 | donor, author-reported | not stated | donor table `2026-10-27.md:132` (no raw artifact) | cap **unknown** (same HW/config, no 350 W in any local record) |
| aggregate tps C4 | 710.2 | donor, author-reported | not stated | level-4 `aggregate_tps`, `:136` | cap unknown |
| TTFT p50 | 101 ms | donor, author-reported | not stated | `:138` | cap unknown |
| prefill tps | 12796 | donor, 32k single sequence | not stated | `:142` | cap unknown; not contract-reachable locally (chunk 8192, budget 8192) |
| decode tps C1 | 355.93 med / 373.78 mean / 307.50 p1 | local artifact, provenance thin | `throughput_p1` | `hyperburst-rtxpro6000/results/betterbench-1007/results.json` | no `betterbench_version`; `-pl` value unknown |
| decode tps C1 | - | **not measured** | - | - | plan + guard in place, run open |
| decode tps C6 | - | **not measured**; donor level 6 was empty | - | - | arm-vs-arm only, never vs donor C4 |
| HiCache host tier | - | **not measured** | - | - | needs owner-approved config change, task 661 |
| token map / MTP head | - | **not shipped** | - | - | donor private (420.9 / 423.7 tok/s) |

| Patch / flag | Start / end (TZ) | Base / candidate SHA + image | Load / repeats | Baseline vs candidate, delta + spread | Quality / tails / VRAM / RAM | Raw evidence / review | Decision |
|---|---|---|---|---|---|---|---|
| baseline docs (this task) | 2026-11-22 00:0x-01:45 CET | fork `05f84d8` + follow-ups, docs only | none | no claim | n/a | `reports/report-2026-11-21-speedy-baseline.md` | docs landed, run open |
