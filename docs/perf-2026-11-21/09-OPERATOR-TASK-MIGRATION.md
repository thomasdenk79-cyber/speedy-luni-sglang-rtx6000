# Operator task migration

Operator repo `luni-ai-operator`, live branch `dev` (GitHub default `master` is stale).

## Operator bug claims (order section 14)
| # | Claim | Finding on `dev` | Action |
|---|---|---|---|
| 1 | runner reads pre-claim path | `luni-operator.sh` sets `task=TASKS/work/<slug>.md` after `mv` | none |
| 2 | `report:` unused, `budget:` not enforced | `report:` read and used; `budget:` is a token budget, wall clock is capped globally by `RUN_TIMEOUT_S` (rc 124/137 booked as ceiling) | none; per-task wall clock optional later |
| 3 | any OMO session blocks dispatch | `agent_busy()` defined, not used by `refill()` | none |
| 4 | head-of-line blocking on gate | `refill()` checks the gate per candidate and continues | none |
| 5 | missing metrics read as 0 % KV | `operator_config.py gate()` returns `engine-unreachable (hold)` | none |
| 6 | safe execution | covered by escalation/failure paths; engine-down watch added 2026-11-21 | none |

## Task mapping (existing plan tasks kept, no duplicates)
| Order file | Existing task | Change |
|---|---|---|
| `speedy-hyperburst-gemv` | `2026-11-21-hyperburst-gpu-ab` (critical) | M sweep 1..16, graph vs eager, BetterBench reproduction |
| `speedy-gpu-resident-scheduler` | `2026-11-21-sched-tier-admission` | aging, skip non-fitting, buffer_only staging rule |
| `speedy-hicache-async` | `2026-11-21-hicache-nonblocking-audit` | invariants A-F, #47621/#50236/#48421 |
| (NVFP4 sidecars) | `2026-11-21-nvfp4-hicache-nixl` | unchanged |
| `speedy-baseline` | new | |
| `speedy-hyperburst-pdl` | new | |
| `speedy-pennyroyal-qsa162` | new | |
| `speedy-pennyroyal-draft43-55` | new | closed PRs as ideas |
| `speedy-nvfp4-flashinfer` | new | |
| `speedy-vllm-qsa-fusions` | new | |
| `speedy-hicache-nixl-ab` | new | |
| `speedy-agentload-c1-c10` | new | includes mixed-chunk + mamba-pool re-evaluation |
| `speedy-quality-regression` | new | |
| `speedy-release-integration` | new | |
| - | new `speedy-port-50236` | absent in rc2, applies cleanly |

GPU-heavy tasks state `GPU exclusive: yes` and run one at a time only inside an owner GPU window
(`state/gpu-window/lease.json`); CPU-only parts (analysis, ports, unit tests, review) run while t15 serves.

## Release claim recheck (Astra, 2026-11-22)

The six historical claims above are not an end-to-end dispatch assurance.
On the committed `dev` snapshot `f55fca0`, the initial `gate-deps` block in
`bin/luni-operator.sh` follows an unconditional `continue` inside the depth
error branch; refill does not recheck after taking its queue snapshot.
In addition, `task_queue.py` on that commit treats brackets in `[a, b]` as
part of the filenames while `task_fm.py` accepts name prefixes. The shared
checkout already has local parser changes: do not replace them wholesale.

The offline candidate lives in the operator worktree
`/srv/community-repos/speedy-operator-release-claim` (based on `f55fca0`).
It shares the exact-name check between enumeration and admission and rechecks
at both initial and refill moves. Synthetic replay of the actual four-prerequisite
R1 card: held with four missing; admitted with four exact done files; held
when one was replaced with a `-review` sibling. Checker failure also holds.
This narrows the check/move race; it is not an atomic transaction.

**Disposition: not deployed.** Integrate with operator task
`2026-11-21-operator-archived-dependency-gate` after its fixture-isolation and
owner-review gate. Do not treat arbitrary archived cards as completed. Review
the watchdog fallback separately (it may skip dependency-bearing tasks), run
presubmit in isolation, then use a runner-free lifecycle window and verify a
new claim before calling the live operator hardened.
