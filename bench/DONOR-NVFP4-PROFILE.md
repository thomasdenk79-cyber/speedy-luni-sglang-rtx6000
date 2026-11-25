# Donor NVFP4 best arm: decode GPU profile

## Contents
- [Setup](#setup)
- [Kernel shares](#kernel-shares)
- [W8A16 GEMV detail](#w8a16-gemv-detail)
- [Next kernel targets](#next-kernel-targets)

## Setup
- Engine `f9bf31a`, NVFP4 KV + Cake + SM120 SwiGLU (`FUSED_FINALIZE=0`), MTP on, CUDA graphs on.
- Torch profiler via `/start_profile`, 40 scheduler steps, CPU+GPU, during 1 and 4 concurrent 3000-token generations. Driver `/tmp/opencode/profile-decode.sh`.
- Traces: `/srv/llm-cache/qwen38-flashnext/profile/c1-20261125T041807Z/`, `c4-20261125T041823Z/`.

## Kernel shares
| Kernel group | c1 | c4 |
|---|---:|---:|
| `_w8a16_gemv_kernel` (FP8 dense linears, Triton) | 36.6 % | 27.2 % |
| SM120 MoE `persistent_main` + `prologue` | 21.8 % | 37.0 % |
| hyper-connection `_hc_up/_hc_down/_hc_branch_stats/hc_combine_gate` | 12.7 % | 9.6 % |
| GDN decode (CUTLASS DSL) | 4.2 % | 6.7 % |
| QSA indexer/top-k | 3.7 % | 2.4 % |
| Cake attention | 1.0 % | 1.1 % |
| singleton prune + router softmax | <1 % | 2.5 % |

## W8A16 GEMV detail
Per call, c1 (12,356 calls in 40 steps, 152 ms):

| Grid | Calls/step | avg µs | ms/step | Likely shape (N x K) |
|---|---:|---:|---:|---|
| 256 | 36 | 29.8 | 1.07 | GDN `in_proj_qkvz` 16384 x 2560 (~1.4 TB/s, near roofline) |
| 40 x 4 | 36 | 14.6 | 0.53 | GDN `out_proj` 2560 x 6144 (~1.1 TB/s) |
| 416 | 17 | 25.4 | 0.42 | attention `q/k/v` fused 13312 x 2560 |
| 1940 | 1 | 400 | 0.40 | `lm_head` 248320 x 2560 (~1.6 TB/s) |
| 3 | 36 | 10.2 | 0.37 | GDN `in_proj_ba` 96 x 2560: 3 CTAs, latency-bound |
| 80 x 10, 32 x 10, 80 | 50-53 | 4-5.5 | 0.72 | shared expert, router gate, small projections |

## Next kernel targets
1. **Small-N latency calls** (`in_proj_ba`, grid 3, ~0.37 ms/step): fuse into `in_proj_qkvz` (one GEMV, N 16480) or split-K to fill SMs. Expected ~3 % at c1.
2. **`out_proj` 2560 x 6144** at 1.1 TB/s: tune split-K / BLOCK_K for M=16 verify rows. Expected 1-2 %.
3. **Hyper-connection kernels** (~10-13 %): check fusion of `_hc_up`/`_hc_down`/`branch_stats` per layer.
4. SM120 MoE dominates at c4 and is new; its tile/split tuning for 16-128 rows is the largest lever at higher concurrency.
- Shapes are inferred from grid sizes and checkpoint tensors; confirm with a shape-recorded microbench before changing code.
