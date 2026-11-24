# Donor NVFP4 KV: first GPU arm

## Contents
- [Runtime](#runtime)
- [Switches](#switches)
- [Results](#results)
- [Assessment](#assessment)
- [Evidence](#evidence)

## Runtime
- Engine `091ed7386fd2dcbd9b884d6bd7e2f54b94df0007` (branch `donor-nvfp4-kv` on donor `0248b5a`); driver `38992a77017269b19026a1215edf139ab2b495bd`.
- Identical to the FIFO/FP8 baseline except `--kv-cache-dtype nvfp4` and `--speculative-draft-kv-cache-dtype fp8_e4m3`: 49,152 static + 256 FP8 FIFO rows, 8 running requests, 48 Mamba slots, pinned host PLE, no HiCache/NIXL, port 8001.
- All other new switches off: `SGLANG_QSA_NVFP4_SCRATCH=bf16` (default), no split-K, no Cake, no `LAB_FAST_*`, no SM120 SwiGLU.
- Startup fix needed: `13b8c13` (hybrid pools passed host scale 1.0; CUDA graph capture rejected the pageable H2D copy).

## Switches
| Switch | Default | This arm |
|---|---|---|
| `LAB_KV_DTYPE` / `--kv-cache-dtype` | `fp8_e4m3` | `nvfp4` |
| `LAB_DRAFT_KV_DTYPE` / `--speculative-draft-kv-cache-dtype` | unset | `fp8_e4m3` |
| `SGLANG_QSA_NVFP4_SCRATCH` | `bf16` | `bf16` |
| `SGLANG_OPT_TRITON_DECODE_ATTN` | off | off |
| `SGLANG_QSA_NVFP4_DECODE` | empty | empty |

## Results
KV pool 509,760 -> **866,496 tokens (+70 %)**.

| Metric | FP8 | NVFP4 |
|---|---:|---:|
| GSM8K 250 | 97.6 % | 97.2 % |
| MMLU 250 | 85.2 % | 86.8 % |

BetterBench 0.6.0, three runs, median aggregate tok/s. FP8 reference c1-c4: [HB-RERUN-3X](HB-RERUN-3X.md) (FIFO, 4 slots, no queueing up to c4); c5-c8: [LRU8](DONOR-LRU8-QUALIFICATION.md) (8 slots).

| c | FP8 | NVFP4 | Delta | TTFT p50 FP8 / NVFP4 (ms) |
|---:|---:|---:|---:|---|
| 1 | 334.4 | 321.7 | -3.8 % | 166 / 164 |
| 2 | 494.3 | 476.4 | -3.6 % | 174 / 174 |
| 3 | 589.6 | 572.1 | -3.0 % | 178 / 177 |
| 4 | 666.8 | 643.6 | -3.5 % | 180 / 181 |
| 5 | 664.9 | 656.5 | -1.3 % | 191 / 187 |
| 6 | 712.8 | 699.9 | -1.8 % | 197 / 193 |
| 7 | 744.7 | 742.7 | -0.3 % | 201 / 196 |
| 8 | 777.2 | 780.1 | +0.4 % | 206 / 198 |

| Prefill | BB FP8 | BB NVFP4 | Cold FP8 (LRU run) | Cold NVFP4 |
|---|---:|---:|---:|---:|
| 2k | 8,469 | 8,528 | - | - |
| 8k | 14,573 | 14,556 | 14,721 | 11,904 (-19 %) |
| 16k | 14,856 | 14,811 | 14,818 | 13,222 (-11 %) |
| 32k | 15,181 | 15,041 | 15,473 | 14,380 (-7 %) |
| 64k | 15,104 | 14,902 | 15,632 | 14,944 (-4 %) |

Agent sim, 4 sessions, matched context 57,348-114,696, 900 s:

| Class | n FP8 / NVFP4 | TTFT p50 (s) | Decode p50 (tok/s) |
|---|---|---|---|
| tool | 48 / 50 | 0.733 / 0.819 | 405.5 / 398.1 |
| refill | 32 / 32 | 3.297 / 3.521 | 419.8 / 380.0 |
| compact | 14 / 15 | 0.942 / 0.921 | 352.0 / 368.6 |

Ramp levels have 4-14 requests each; too few for a verdict.

## Assessment
- Capacity +70 % at unchanged accuracy; decode -3 to -4 % at c1-c4, neutral at c7-c8.
- Cold prefill loses most at short depths: the bf16 prefix dequant per chunk. Next arm: `SGLANG_QSA_NVFP4_SCRATCH=fp8` + split-K, then `SGLANG_QSA_NVFP4_DECODE=cake`.
- An unmatched agent run (NVFP4 auto-capped to 150k-195k context) is kept as evidence but not compared.

## Evidence
- `/srv/llm-cache/qwen38-flashnext/ab-night/donor-nvfp4-20261124T185149Z/` (`kv4/`, `matched-load/`, `plan.json`, `logs/`).
- FP8 load baseline: `/srv/llm-cache/qwen38-flashnext/ab-night/donor-fifo8-load-20261124T180815Z/`.
- Failed starts: `donor-nvfp4-20261124T173445Z` (ready timeout during forced MoE JIT rebuild), `donor-nvfp4-20261124T183842Z` (graph-capture H2D scale, fixed by `13b8c13`).

## Arm 2: FP8 scratch + split-K
`SGLANG_QSA_NVFP4_SCRATCH=fp8`, `SGLANG_OPT_TRITON_DECODE_ATTN=1`; everything else as arm 1. Evidence `donor-nvfp4-fp8scratch-20261124T203652Z/`.

- Split-K never ran on decode: with MTP every decode step is a verify (`rows 16 != batch 1`), so it falls back to TRT-LLM on the FP8 scratch. The split-K switch is a no-op with speculative decoding.
- Cold prefill recovered to FP8-KV level: 14,858 / 15,110 / 15,562 / 15,634 tok/s at 8k/16k/32k/64k (bf16 scratch: 11,904 / 13,222 / 14,380 / 14,944).
- BetterBench vs arm 1: c1-c8 -0.4 to -3.4 % (c3 -3.4 %, c4 -2.7 %); BB prefill +0.5 to +1.3 %.
- Accuracy GSM8K 96.8 %, MMLU 84.8 % (arm 1: 97.2 / 86.8; FP8 KV: 97.6 / 85.2).
- Agent sim (matched context): tool TTFT p50 0.739 s, decode 387 tok/s; refill decode 312 vs 380 tok/s in arm 1.
- Verdict: FP8 scratch fixes prefill but costs decode; candidate split is FP8 scratch for prefill and BF16 or Cake for decode.
