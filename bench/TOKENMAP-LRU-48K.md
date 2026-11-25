# Speedy benchmark: fully replaceable 48 Ki draft token map

This plan supersedes the **32k experiment** in [the donor reference](HB-REFERENCE-RC1.md), not the measured donor baseline. It is an isolated Speedy benchmark, not a production default.

## Fixed size and last-used policy

- Match the shipped static head size: **49,152 total rows** (48 Ki, about 50k). Use a 40,960-ID seed and the 8,192 extra slots from [the candidate environment](tokenmap-lru-48k.env). Appending the slots to the uncut 49,152-ID file would instead produce 57,344 rows.
- `POLICY=lru`: prompt and accepted-output occurrences refresh recency. The least recently used entry is the eviction candidate. The rc5-p1 default is FIFO, which does not refresh on reuse. **Pass the environment to the Speedy test server and confirm `lru` in its startup log**; merely setting the slot count does not activate LRU.
- `ADAPTIVE_STATIC=1`, `STATIC_CORE=0`: all seed IDs are replaceable; none are pinned. `STATIC_HORIZON=0` gives seed entries no positive recency bonus. This is eligibility for eviction, not a requirement to flush useful seed IDs. Keep persisted learning disabled for the clean-start comparison.

Build the seed offline with the exporter from the pinned Speedy SGLang source. Provide `--rank` if a separate frequency ranking is available; otherwise the reproducible seed uses the shipped ID order and is **not** a frequency top-k. Record the seed hash and ordering in the run identity. Use the same seed in both matched arms.

```bash
python /path/to/sglang/srt/speculative/dynamic_draft_vocab.py \
  --token-map tokenmaps/hyudryu_top49152.json --k 40960 --out /path/to/seed-40960.pt
```

## Compare and verify

1. Both arms use the same Speedy build/model and test-only flags: `--max-running-requests 8`, `--max-mamba-cache-size 48`, `--cuda-graph-bs-decode 1 2 3 4 5 6 7 8`, `--cuda-graph-max-bs-decode 8`; HiCache and SSD streaming off. Verify eight active server requests and BS8 decode graph use.
2. Pass the seed via `--speculative-token-map` in both arms. Control: override only `SGLANG_SPEC_DYNAMIC_VOCAB_ADAPTIVE_STATIC=0` (40,960 pinned seed rows + 8,192 LRU rows). Candidate: unmodified environment file (0 pinned + 49,152 LRU rows). Optionally measure the shipped 49,152 static-only map as a separate reference, not as the matched control.
3. Use the [canonical BetterBench driver](CANONICAL-DRIVER.md), c1–c8, followed by matched agent traffic. Report early and steady throughput, TTFT, queue, and draft acceptance. In a separate validation pass enable `SGLANG_SPEC_DYNAMIC_VOCAB_METRICS=1`, confirm startup `lru` and `0 static + 0 ring + 49152 pool`, and require nonzero `demote` to prove eviction of an original seed ID. `evict` alone is not sufficient. A short run with no vocabulary pressure cannot validate replacement.

Do not change the running HyperBurst donor or start Speedy on its GPU without an authorized handoff.

**CPU preflight (rc5-p1 `e4e0b0e3d5`):** the generated 40,960-ID seed and this environment produced 49,152 unique rows, zero pinned rows and 64 seed demotions under synthetic vocabulary pressure. A separate deterministic recency check confirmed that reusing a token retained it while the older token was displaced. GPU and server-start validation remain open.
