# Large prefill and generated tokens per wall-clock hour

[Observed](#observed) | [Interpretation](#interpretation) | [Deciding experiment](#deciding-experiment)

## Observed

Read-only rc2 observation, **2026-11-21 20:00 to 2026-11-22 00:29 CET**.
The [November 18 BetterBench 0.6.0 reference](../releases/speedy-rc2.md)
measured NVFP4 median prefill **13,523 tok/s at 8K**, 13,045 at 16K,
13,666 at 32K and 13,329 at 64K, at **350 W**. Even that reference
measured only 7,182 at 2K; 13K is not a target for every prompt shape.

| rc2 log class | Count | Input tok/s p10 / p50 / p90 |
|---|---:|---:|
| Consecutive full 8,192-new-token prefill batches, one sequence, zero running requests, no replay | 25 | 9,019 / **9,107** / 10,307 |
| First full batch of such a streak (includes preceding non-prefill time) | 11 | 535 / 1,405 / 1,656 |
| All batches with fewer than 2,048 new tokens (includes warm turns) | 1,398 | 6 / 132 / 1,232 |

Source: `podman logs --timestamps --since 2026-11-21T20:00:00+01:00 --until
2026-11-22T00:29:00+01:00 qwen38-rc2`, filtered to scheduler `Prefill batch`
and `Decode batch` messages in timestamp order. Of 1,893 prefill lines,
the full-chunk class above excludes the first qualifying line after any
decode or nonqualifying prefill. One of the 25 was at or below 8,500.
These are **batch log intervals**, not 25 independent cold prompts or GPU timings.

Prometheus `increase(sglang:realtime_tokens_total{instance="host.containers.internal:30000",mode="decode"}[1h])`
at the right edge of each hour; same selector for `mode="prefill_compute"`.
Rounded counts are scrape-based estimates, not demand-normalized capacity.

| Hour (Nov 21, CET) | Generated tokens | Fresh prefill compute tokens | Decode samples / counter resets |
|---|---:|---:|---:|
| 20:00-21:00 | 311,683 | 728,449 | 238 / 0 |
| 21:00-22:00 | 658,247 | 2,015,450 | 240 / 0 |
| 22:00-23:00 | 584,118 | 2,732,987 | 240 / 0 |
| 23:00-00:00 | 659,841 | 1,889,349 | 240 / 0 |

## Interpretation

- **A real comparison mismatch:** min and max of `nvidia_smi_enforced_power_limit_watts`
  over the 269-minute window were both **300 W**, not the reference 350 W.
  Fleet power draw p90 was 300.26 W on the single GPU. This makes power a
  plausible contributor, not proof it caused the gap; those samples are not
  restricted to the 25 batches.
- **Decode displacement:** `/get_server_info` reported chunk size 8,192,
  max prefill 16,384, mixed chunk off, interval 1, FCFS, EAGLE (7 steps),
  NVFP4 KV. Separate prefill forwards occupy compute that would otherwise
  serve ready decodes. With no decode waiting, faster prefill mainly improves TTFT.
- **Metric mismatch:** `sglang-upstream@a090492091`, `scheduler_components/metrics_reporter.py`,
  `report_prefill_stats()`: `(new + replay tokens) / time since previous prefill report`.
  This includes intervening decode/idle; it is not the prefill kernel time.
  The consecutive-full class reduces this bias but does not control prefix depth,
  clocks, graph route, or runtime build. A kernel regression is **not established**.
- At fixed work, 8K to 12K tok/s (**+50%**) cuts that prefill duration by
  **one third**, not half. For 64,000 fresh tokens that is 8.0 to 5.33 s
  (illustrative, not measured). Only the fraction reclaimed while decode has
  eligible work can increase generated tokens per hour.

## Deciding experiment

`speedy-baseline` owns the cold reference; `speedy-agentload-c1-c10` owns
displacement and the release KPI. Reuse operator investigation `153-sglang-prefill-stalls`
for stall diagnostics; tasks 169/190 address **short** chunks, not this cold comparison.

1. In the approved GPU window reproduce the historical harness, prompt lengths,
   cold state, model, NVFP4 pool and seeds on the reference and candidate build
   **at the same approved power limit**, starting at 300 W. Record actual clocks,
   power, device forward elapsed time, backend/graph route and TTFT. Compare truly
   fresh 8K/16K/32K/64K separately from 8K extensions on a 200K resident prefix;
   alternate arms, at least three repeats, include spread. No live power changes.
   The power cap alone does not explain the gap: the owner reports donor forks
   (e.g. Pennyroyal) also power-limit the RTX PRO 6000 and still publish better
   numbers; verifying each donor's actual cap is part of this task. Attribute
   the delta only after matched-power, matched-build measurement; until then
   the 300 W cap is a confounder, not a proven cause.
2. Replay identical arrivals and outputs: steady ready decodes at C1/C4/C6 plus
   large new sessions. Measure emitted decode tokens per **whole elapsed hour**
   (including prefill/idle), prefill duty from device timing, decode gap and ITL
   p95/p99, TTFT and queue wait. Also run a cold-only control. Do not equate
   inter-log gaps with GPU duty or subtract prompt tokens from generated tokens.
3. If both builds are slow at matched power, isolate prefix depth, chunk size
   then attention route one factor at a time; #162/#45791 is a **long cached
   prefix** candidate, not an assumed fix for cold prompts. Test mixed chunks
   separately: the source allows EAGLE, but mixed steps use a plain one-token
   decode, so improved ITL can cost speculative throughput. Verify actual route.
   Promote only with repeatable decode/hour gain at fixed demand, acceptable
   tails, unchanged correctness and no headroom regression.
