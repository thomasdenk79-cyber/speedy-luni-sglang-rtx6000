# Donor-first vocabulary ablation

## Reference

Keep the measured [HyperBurst donor](HB-RERUN-3X.md) unchanged. Its SGLang `8edd1fab4b` startup log shows 49,152 fixed NVFP4 draft-head rows **plus 256 FP8 FIFO rows** (49,408 total). The draft-head formats are separate from the donor's FP8 KV cache. The existing context-vocabulary option implements FIFO with a fixed hot prefix, not fully replaceable LRU.

The Speedy rc5-p1 c1–c8 run combined a different build/model profile, NVFP4 KV, the eight-slot profile and a 49,152-row fully replaceable LRU map. It is a functional result, not a feature ablation; it does not replace the donor baseline.

## Requested candidate

Run the four changes together as one candidate on the donor engine and model profile:

- 49,152 total draft-vocabulary rows, fully replaceable by last-used (LRU), with no pinned IDs.
- `--max-running-requests 8`.
- `--cuda-graph-bs-decode 1 2 3 4 5 6 7 8` (maximum decode graph batch size 8).
- `--max-mamba-cache-size 48`.

Keep the remaining donor settings, including FP8 KV, the existing draft-head representation, loopback port `8001`, and HiCache/NIXL off. Reuse the existing FIFO baseline; do not run it again and do not stage an intermediate four-slot LRU profile. Do not substitute rc5 for the donor candidate.

## Donor release source

Base: `8edd1fab4b6c2d7ff11f0bbfc8e21a195bd41a07`. Apply [the donor LRU patch](../patches/donor-lru/0001-spec-add-fully-replaceable-LRU-to-the-donor-mixed-dr.patch) in an isolated checkout with `git am`. The reference checkout is not modified. The new path is opt-in with `SGLANG_DONOR_DRAFT_LRU=1`; the flag-off path remains the donor's FIFO implementation.

The primary head keeps a mixed representation: 48,896 packed NVFP4 rows plus 256 FP8 rows. The recurrent decode head keeps its smaller width and mixed representation. Each head has its own LRU map, updated from the same observations, so the recurrent subset need not share the primary FP8 tail. Packed replacements come from a BF16-derived source bank loaded once at startup, not from disk during inference. A single bank-wide scale replaces the donor's subset-specific scale for the packed heads; record this implementation difference when interpreting acceptance and throughput.

Set `CONC=8`, `MAMBA_SLOTS=48`, and pin `ADAPTIVE_CONFIG` to the donor's `adaptive/w16_c4.json` to avoid silently switching adaptive policies. The donor launcher derives decode buckets from `CONC`. Optional `SGLANG_DONOR_DRAFT_LRU_METRICS=1` logs seed evictions and switches its counters off after both heads show demotion. Retain the original allocator and memory fraction, model, FP8 KV, and no HiCache or NIXL.

The CPU oracle matched sequential LRU over 512 randomized batches. A synthetic GPU test replaced both FP8 and NVFP4 rows and verified that a captured graph read the updated IDs, weights and scales at the original addresses. This is not a full-model startup or BS8 replay result.

Verify seed eviction, eight simultaneous admissions and BS8 replay before the three interleaved BetterBench c1–c8 runs. Report min/median/max, prefill, TTFT, acceptance and active/queued requests. Compare common client levels to the recorded donor. Attribute the result to the four-change package, not to LRU alone.
