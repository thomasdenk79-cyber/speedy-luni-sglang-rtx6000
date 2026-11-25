# Donor-LRU8: GPU qualification and BetterBench

## Runtime

- Engine `0248b5a15ec0c3a5e289e6d4d97c723c9e0140c2`, based on donor `8edd1fab4b`; canonical driver `38992a77017269b19026a1215edf139ab2b495bd`.
- RTX PRO 6000 Blackwell; Qwen3.8-Flash-Next-NVFP4, served as `flash-next` on `127.0.0.1:8001`; FP8 E4M3 KV, no HiCache/NIXL, PLE in host RAM. Adaptive `w16_c4.json`, 8,192-token chunks, 262,144 context. Effective KV pool 509,760 tokens (donor: 633,344).
- Startup and `HB-OK` passed on PID 2432391. Primary head: 49,152 replaceable rows, zero pinned (48,896 NVFP4 + 256 FP8); recurrent head: 33,024 replaceable rows. 48 Mamba slots, eight running requests, decode graphs BS1–8.
- Pressure with 12,000 distinct non-seed input IDs produced logged seed demotions `[12002, 12004]` across the two heads. Validation counters then disabled themselves without disabling LRU. The admission probe completed 8/8, sampled eight active, and logged eight in one decode batch with `cuda graph: True`. Probe actual accepted drafts per verify (no bonus): 1.35 for the pressure prompt (73/54), 4.37 pooled across the eight requests (3,341/765); not representative throughput or coding measures.

## Three interleaved sweeps

BetterBench 0.6.0, corpus 1.0, greedy, 48/48 successful concurrent requests per level per run. Donor comparison uses the [existing matched three-sweep result](HB-RERUN-3X.md); no FIFO rerun.

| Clients | Aggregate min / median / max (tok/s) | Median vs FIFO | TTFT p50 / p95 (ms) | Request decode p50 (tok/s) | Peak running / waiting | Logged accept length p50 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 331.7 / 332.6 / 336.5 | −0.5% | 171 / 193 | 423.7 | 1 / 0 | 3.60 |
| 2 | 480.0 / 484.7 / 488.6 | −1.9% | 180 / 205 | 287.5 | 2 / 0 | 2.95 |
| 3 | 580.5 / 584.0 / 585.5 | −0.9% | 181 / 200 | 232.0 | 3 / 0 | 2.93 |
| 4 | 649.5 / 653.1 / 660.2 | −2.1% | 186 / 349 | 194.3 | 4 / 0 | 2.95 |
| 5 | 661.0 / 664.9 / 665.7 | −0.1% | 191 / 344 | 156.0 | 5 / 0 | 2.93 |
| 6 | 709.2 / 712.8 / 718.7 | +7.8% | 197 / 363 | 141.7 | 6 / 0 | 2.93 |
| 7 | 737.7 / 744.7 / 760.1 | — | 201 / 343 | 127.3 | 7 / 0 | 2.95 |
| 8 | 775.8 / 777.2 / 784.2 | — | 206 / 362 | 119.1 | 8 / 1 | 2.92 |

TTFT pools 144 requests per level (nearest-rank p95). Running/waiting maxima are independently sampled in the interiors of stages reconstructed from result timestamps and stage durations. Accept length is a median of interior decode-log windows, **including** the bonus token, not per-request draft acceptance. At c5/c6 the four-slot donor was queue-limited: its TTFT p50 was 678/1,294 ms and peak waiting 2/3. No c7/c8 donor data exist. These results describe the four-change package, not LRU in isolation.

| Prefill target | Actual prompt tokens p50 | BetterBench prefill min / median / max (tok/s; 24 samples) |
|---:|---:|---:|
| 2k | 1,514 | 6,773 / 7,738 / 8,307 |
| 8k | 5,892 | 14,148 / 14,416 / 14,765 |
| 16k | 11,800 | 14,571 / 14,766 / 14,951 |
| 32k | 23,548 | 14,926 / 15,044 / 15,257 |
| 64k | 47,014 | 14,801 / 14,959 / 15,101 |

At matched 8k–64k depths, BetterBench prefill medians were 0.6–1.1% below donor. Logs reported no cached tokens in pre-sweep batches. This metric is prompt tokens divided by TTFT, not isolated compute throughput. Accuracy, dedicated cold prefill, agent load, ramp and coding replay are separate runs and not included in this report.

## Evidence

`/srv/llm-cache/qwen38-flashnext/ab-night/donor-lru8-20261124T155916Z/`: `plan.json`, `probe.json`, `validation-bench.log`, `live.jsonl`, and `betterbench/benchmark-identity.json`. The driver's first-occurrence CLI detection says four slots due to an earlier default flag; the scheduler and runtime probe confirm the effective eight. Result JSON SHA-256 under `betterbench/`:

| Path | SHA-256 |
|---|---|
| `bb-170707/run-1/results.json` | `e5e832c9f08bd27de741ceb548e4cec87756a61075ed430fe168666adbd6b004` |
| `bb-171640/run-1/results.json` | `b15e3daa0dd4757b67e531fe3e0097b7fd362122d1e45a921331b82b2ccadfec` |
| `bb-172609/run-1/results.json` | `a0c0c7dea1e94adb4c42b287e96c30c64fbf4c11d8aaef0f6fcbd9919b8efaf3` |
