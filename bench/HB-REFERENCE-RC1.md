# Speedy benchmark reference v1.1-rc1

**Scope:** accepted HyperBurst donor reference for Speedy development. This tag versions benchmark evidence and the next experiment, not Speedy engine qualification. The donor stays unchanged.

## Donor reference

BetterBench 0.6.0, greedy, corpus 1.0 (`e332dceff176d033`), 48 total requests per level. Donor SGLang `8edd1fab4b6c2d7ff11f0bbfc8e21a195bd41a07` on RTX PRO 6000 Workstation Edition (620.70.06), four server slots, 20 Mamba-cache slots, decode graphs 1–4, FP8 KV, HiCache off. Use the [canonical driver](CANONICAL-DRIVER.md).

| Client concurrency | Baseline aggregate min / median / max (tok/s, 3 runs) | Rerun aggregate tok/s | Rerun TTFT p50 / p95 | Rerun peak running / waiting |
|---:|---:|---:|---:|---:|
| 1 | 329.8 / 333.3 / 336.8 | 339.2 | 168 / 188 ms | 1 / 0 |
| 2 | 490.6 / 491.1 / 494.0 | 492.2 | 172 / 198 ms | 2 / 0 |
| 3 | 589.3 / 590.2 / 593.0 | 589.7 | 175 / 298 ms | 3 / 0 |
| 4 | 655.0 / 667.3 / 668.1 | 666.8 | 179 / 333 ms | 4 / 0 |
| 5 | — | 670.9 | 641 / 1,509 ms | 4 / 1 |
| 6 | — | 668.4 | 1,293 / 2,007 ms | 4 / 2 |

Rerun: first completed sweep, 48/48 at every level. c5/c6 measure queueing with four server slots, not five or six active server slots. BetterBench prefill medians at 8k/16k/32k/64k target: **14,527 / 14,927 / 15,245 / 15,115 tok/s** (24 samples each). Target depth is a label, not the actual prompt-token count or a raw compute benchmark.

Raw data under `/srv/llm-cache/qwen38-flashnext/ab-night/`: baseline `0635-donor/bb-c1234*/results.json`; rerun `hb-bb-verify-20261124T140837Z/bb-150838/run-1/results.json` (SHA-256 `050953ed5920a85d04ad07fb93d8ed0f2b2966490d82eb8e68d7f2bc319ffac6`). The rerun root also holds `live-metrics.jsonl`.

## Next experiment: 32k adaptive draft vocabulary at eight slots

1. Use the same Speedy build and model in both arms: `--max-running-requests 8`, `--max-mamba-cache-size 48`, `--cuda-graph-bs-decode 1 2 3 4 5 6 7 8`, `--cuda-graph-max-bs-decode 8`. Disable HiCache and SSD streaming for this isolated test. Confirm eight simultaneous server requests, BS8 decode graph use, and Mamba/KV pool headroom before comparing results.
2. Fix the draft head at **32,768 total rows**: a 24,576-ID seed plus `SGLANG_SPEC_DYNAMIC_VOCAB_SLOTS=8192`. Use a frequency-ranked seed and keep its ranking explicit; the shipped 49,152-ID file is ID-sorted, so slicing it is not a frequency top-k. Adding 32,768 slots to the shipped map would instead produce 81,920 rows.
3. Use `SGLANG_SPEC_DYNAMIC_VOCAB_POLICY=lfu` in both arms. Control: `SGLANG_SPEC_DYNAMIC_VOCAB_ADAPTIVE_STATIC=0`. Candidate: `ADAPTIVE_STATIC=1` and `STATIC_CORE=8192` under the same prefix; this protects 8,192 ranked seed IDs and lets other seed IDs be replaced. `STATIC_CORE=0` is a separate fully replaceable variant, not the first arm. Keep profile state isolated and start both arms without carried-over learned state.
4. Run repeated interleaved BetterBench c1–c8 and the same representative agent workload. Report early versus steady aggregate tok/s, TTFT, acceptance, and queue. In a separate instrumented pass, enable `SGLANG_SPEC_DYNAMIC_VOCAB_METRICS=1` in both arms: require nonzero `demote` to demonstrate original seed displacement (`evict` alone is not proof). Instrumented measurements are not the headline throughput run.

Do not start this comparison on the donor GPU without an authorized GPU handoff. No Speedy performance claim is made by this reference tag.
