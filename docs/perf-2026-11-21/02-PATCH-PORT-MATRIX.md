# Patch port matrix

States checked 2026-11-22 with `gh pr view` (merge SHA = merge commit, first 10 chars).
"In rc2" = reverse-apply of the PR diff against `rc/qwen38-rc2` (`git apply --cached -R --check`):
`present` reverses cleanly, `absent` applies cleanly forward, `diverged` = neither, needs a content check.
Benchmark and quality columns stay `-` until a task records raw data in [RESULTS.md](RESULTS.md).

| Source | PR | Merge | SHA | Topic | In rc2 | Branch | Expected benefit (label) | Bench | Quality | Decision |
|---|---|---|---|---|---|---|---|---|---|---|
| jpezzulli/sglang-rtxpro6000 | 162 | merged 2026-11-06 | dfd469c3a7 | QSA prefill for long cached prefixes | check vs #45791 | - | ~26 % C1 / 33 % C4 fresh prefill after 200K cached (donor) | - | - | PORT if missing |
| sgl-project/sglang | 45791 | merged 2026-10-26 | 642e9c1448 | QSA prefill indexer, long cached prefixes | diverged | - | upstream variant of #162 (donor) | - | - | content check first |
| jpezzulli/sglang-rtxpro6000 | 38 | **closed** | - | fused QSA prefill KV gather (FP8 staging) | - | - | fewer launches (not measured) | - | - | HOLD, NVFP4 layout differs |
| jpezzulli/sglang-rtxpro6000 | 43 | **closed** | - | reuse draft-step seq lens | - | - | C1/C2 draft overhead (not measured) | - | - | HOLD, re-derive |
| jpezzulli/sglang-rtxpro6000 | 55 | **closed** | - | adaptive feedback matched to forward | - | - | correctness under overlap | - | - | HOLD, audit our controller |
| jpezzulli/sglang-rtxpro6000 | 33 | merged 2026-10-07 | d4bf0c3dee | SM120 FP8 low-row output head | - | - | overlaps QFN-0159..0169 | - | - | KEEP one winner per shape |
| jpezzulli/sglang-rtxpro6000 | 75 | **closed** | - | wait for layer load before QSA compressed keys | - | - | correctness | - | - | HOLD, write equivalent test |
| jpezzulli/sglang-rtxpro6000 | 73 | merged 2026-10-17 | 6d5e00a39c | page-aligned Mamba host strides (O_DIRECT) | - | - | avoids bounce buffers (not measured) | - | - | check vs upstream |
| jpezzulli/sglang-rtxpro6000 | 118 | merged 2026-10-23 | 16bf12873e | native MTP width policy (C1 W8, C2+ W4) | - | - | spec tuning (donor) | - | - | evaluate in W-sweep |
| sgl-project/sglang | 47621 | open | - | buffer_only oversize storage hit trim | diverged | - | 200K-500K restores (not measured) | - | - | evaluate |
| sgl-project/sglang | 50236 | open | - | clamp buffer_only anchor-lock cap | **absent, applies cleanly** | - | avoids scheduler stall (correctness) | - | - | PORT (quick win) |
| sgl-project/sglang | 48421 | merged 2026-11-05 | e838360ffd | reject partial L3 prefetch, hybrid | diverged | - | correctness | - | - | content check |
| sgl-project/sglang | 49158 | open | - | write_back admission | diverged | - | only write_back | - | - | DROP for buffer_only+write_through |
| flashinfer-ai/flashinfer | 6804 | merged 2026-10-21 | 53f175d7d7 | NVFP4 QSA sparse paged attention | - | - | direct NVFP4 consume (not measured) | - | - | compare |
| vllm-project/vllm | 64199 | open | - | NVFP4 main KV via FlashInfer | - | - | reference integration | - | - | compare |
| vllm-project/vllm | 54846 | open | - | Triton FP8/NVFP4 QSA (SM120) | - | - | alt kernel | - | - | compare |
| vllm-project/vllm | 57097 | merged 2026-10-01 | c9578f10ab | fuse QK-norm/RoPE/gate/KV write | - | - | -2 launches (donor) | - | - | evaluate |
| vllm-project/vllm | 59533 | merged 2026-10-05 | 042ab0305c | merge QKVG + indexer QK GEMM | - | - | 1.45-1.68x kernel, non-SM120 (donor) | - | - | evaluate on SM120 |

Own patches (QFN register, all GPU `pending`): HyperBurst QFN-0193..0202 (`perf/hb-lowrow-kernels`),
QFN-0203..0214 (`perf/hb-prune-pdl`), draft vocab buckets QFN-0170..0192. Validate, never re-create.
