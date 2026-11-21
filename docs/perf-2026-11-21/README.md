# Perf campaign 2026-11-21

## Contents
- [Order and review](#order-and-review)
- [Deviations from the order](#deviations-from-the-order)
- [Artifacts](#artifacts)

## Order and review
- Order (data, German, verbatim): [AUFTRAG-2026-11-21-browseragent.md](AUFTRAG-2026-11-21-browseragent.md).
- Reviewed 2026-11-22 by Opus against `gh pr view`, `sglang-upstream` branches and
  `luni-ai-operator@dev`. Result: the direction is sound and matches the owner priorities
  (speed > NVFP4-KV with HiCache/NIXL > VRAM > RAM > SSD, GPU-resident-first scheduling,
  non-blocking HiCache, pinned PLE, no SSD experiments).
- Verified PR states: [02-PATCH-PORT-MATRIX.md](02-PATCH-PORT-MATRIX.md).
- Operator and task migration: [09-OPERATOR-TASK-MIGRATION.md](09-OPERATOR-TASK-MIGRATION.md).

## Deviations from the order
| Order says | Finding | Decision |
|---|---|---|
| Pennyroyal #38, #43, #55, #75 as port sources | All four are **CLOSED, not merged** | Idea sources only; re-derive on our tree, HOLD until explorer check |
| SSD tier 3 in the scheduler | Owner: no new SSD offload experiments | Keep SSD hits as lowest admission class (scheduling only); no new SSD work |
| Operator bug 1 (stale task path) and 2 (`report:` unused) | Fixed on `dev` (`luni-operator.sh` sets `task=TASKS/work/<slug>.md`, reads `report:`) | No work; GitHub default branch `master` is 1221 commits behind `dev`, the live operator runs `dev` |
| Operator bug 3 (OMO busy blocks all) | `agent_busy()` is defined but not called by the dispatcher | No work; concurrency comes from runner caps |
| Old tasks `mixed-chunk`, `mamba-pool-groesser` in `TASKS/todo` | Archived in `TASKS/archive/2026-11-18-reset/` | Re-evaluate inside `speedy-agentload-c1-c10`, not revived 1:1 |
| HyperBurst ref C1 369.9 / C4 710.2 tok/s | Donor numbers, not reproduced locally | Label as donor until `speedy-baseline` reproduces them |

## Artifacts
| File | Status | Owner task |
|---|---|---|
| `00-SOURCE-INVENTORY.md` | open | `2026-11-21-speedy-baseline` |
| `01-UPSTREAM-AND-FORK-DIFF.md` | open | `2026-11-21-speedy-baseline` |
| `02-PATCH-PORT-MATRIX.md` | started (PR states verified) | all port tasks update their row |
| `03-NVFP4-KERNEL-COMPARISON.md` | open | `2026-11-21-speedy-nvfp4-flashinfer` |
| `04-HICACHE-NIXL-AUDIT.md` | open | `2026-11-21-hicache-nonblocking-audit`, `2026-11-21-speedy-hicache-nixl-ab` |
| `05-GPU-RESIDENT-SCHEDULER-DESIGN.md` | open | `2026-11-21-sched-tier-admission` |
| `06-BENCHMARK-PLAN.md` | open | `2026-11-21-speedy-agentload-c1-c10` |
| `07-QUALITY-GATES.md` | open | `2026-11-21-speedy-quality-regression` |
| `08-INTEGRATION-ROADMAP.md` | open | `2026-11-21-speedy-release-integration` |
| `09-OPERATOR-TASK-MIGRATION.md` | done | - |
| `RESULTS.md` | open | every benchmark task appends; no numbers without raw data |
