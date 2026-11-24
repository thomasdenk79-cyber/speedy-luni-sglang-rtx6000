# HyperBurst donor: three-run BetterBench verification

Follow-up to [the tagged donor reference](HB-REFERENCE-RC1.md). This is not a Speedy or 48 Ki LRU measurement. The donor remained on the same process and command line, production stayed down, and all 1,580 chat HTTP requests (including warmups and interleaved probes) returned 200. BetterBench 0.6.0, corpus 1.0, effective greedy config and 48 requests per level matched the earlier method.

| Client concurrency | Aggregate min / median / max (tok/s) | TTFT p50 / p95 (ms) | Decode p50 (tok/s per request) | Peak running / waiting |
|---:|---:|---:|---:|---:|
| 1 | 333.2 / 334.4 / 339.2 | 166 / 197 | 430.1 | 1 / 0 |
| 2 | 492.2 / 494.3 / 494.4 | 174 / 198 | 298.7 | 2 / 0 |
| 3 | 588.3 / 589.6 / 589.7 | 178 / 202 | 234.9 | 3 / 0 |
| 4 | 655.0 / 666.8 / 668.9 | 180 / 333 | 199.4 | 4 / 0 |
| 5 | 660.7 / 665.3 / 670.9 | 678 / 1,475 | 198.4 | 4 / 2 |
| 6 | 661.1 / 661.4 / 668.4 | 1,294 / 2,125 | 198.0 | 4 / 3 |

Per-request percentiles pool 144 successful requests per level (nearest-rank p95). Running/waiting maxima are independently sampled Prometheus gauges in the interior of each sweep stage; they need not occur in the same sample. Decode logs saw queue maxima of 1/2 for c5/c6. The c4 aggregate median differs from v1.0 by −0.08%; paired within-run changes at c5/c6 were within ±1.2%. This tests queues in front of four server slots, not a higher server cap.

| Prefill target | Actual prompt tokens, median | Prefill min / median / max (tok/s, 24 samples) |
|---:|---:|---:|
| 8k | 5,892 | 14,318 / 14,573 / 14,918 |
| 16k | 11,800 | 14,698 / 14,856 / 15,039 |
| 32k | 23,548 | 14,957 / 15,181 / 15,364 |
| 64k | 47,014 | 14,910 / 15,104 / 15,247 |

Server logs reported no cached tokens in pre-sweep batches in any of the three runs. BetterBench prefill is still prompt tokens divided by TTFT, not isolated compute throughput. Across the campaign the sampled KV peak was 47,232 / 633,344 tokens and the sampled retraction gauge stayed zero. Host-wide swap counters rose by 65,057 pages in and 5,601 pages out; this cannot be attributed to the engine or equated to KV swapping. GPU temperature peaked at 92.77 °C, with no causal assignment to throughput variation.

## Raw evidence

`/srv/llm-cache/qwen38-flashnext/ab-night/hb-bb-verify-20261124T140837Z/` contains the results, `verification-analysis.txt`, `live-metrics.jsonl`, `gpu.csv` and process-identity snapshots. Result JSON SHA-256, in run order:

| Result | SHA-256 |
|---|---|
| `bb-150838/run-1/results.json` | `050953ed5920a85d04ad07fb93d8ed0f2b2966490d82eb8e68d7f2bc319ffac6` |
| `bb-151714/run-1/results.json` | `830bbf4f1437cf9a78ffe3a0afb054a9a58c4847e09cddad0b16680819aa1396` |
| `bb-152550/run-1/results.json` | `214edd56822a11ede0124968390246eb4847465affb0282d6285a27bcb0ac252` |
