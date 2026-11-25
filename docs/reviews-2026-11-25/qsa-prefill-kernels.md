# QSA prefill kernels — relevance to OUR workload (read-only, 2026-11-25)

**Method:** `git show` / `git grep` / file reads only. No repo or worktree edits, no GPU use, no
benchmark touched (port 8001 left alone). Local refs inspected:
`sglang-jpezzulli-pennyroyal` @ `202f266be2d` (detached; **does not contain** `dfd469c3a7`),
`sglang-upstream` @ `fp4local-port` / `108c00266f`, rc2 `a090492091`, `perf/qsa-relative-prefill-20261122`.

---

## 1. Path separation — do not conflate decode and prefill

| Fact | Evidence |
|---|---|
| **Cake attention is only 1.0 % (C1) / 1.1 % (C4) of decode GPU time.** Even deleting it entirely caps decode upside at ~1 %. | `speedy-luni-sglang-rtx6000/bench/DONOR-NVFP4-PROFILE.md:22` (Torch profile, 40 steps, 1 and 4 concurrent 3000-token generations; donor build `f9bf31a`). |
| The NVFP4 kernel comparison (FlashInfer Cake `#6804`, vLLM `#54846`/`#64199`, our CUDA split-K decode) is overwhelmingly a **decode** study. Its own prefill recommendation is "our Triton path A as fallback"; it names Cake/vLLM as **ideas sources, not measured SM120 tempo**. | `docs/perf-2026-11-21/03-NVFP4-KERNEL-COMPARISON.md:25-27, 92-107, 303-318`. |
| Prefill is a separate forward path. The QSA prefill indexer above 1024 tokens runs on an overlap stream; its optimization target is **indexer scoring launches**, not the decode attention kernel. | `02-PATCH-PORT-MATRIX.md:82-88`; dispatch code `qsa_indexer.py:438-512` at rc2. |
| **94.9 % of 2.31 B prompt tokens were served from the prefix cache** — long cached prefixes dominate token volume, even though cold prefill is the visible KPI. | `docs/releases/speedy-rc2.md:116, 251`. |
| Our cold prefill (task-given ~14.9–15.5 K tok/s, 8K–64K unique, no hits). Docs record rc2 BetterBench 13.0–13.7 K tok/s at 8K–64K (350 W) and live 300 W full-8K batches p50 ~9.1 K; 300 W vs 350 W is a **confounder, not a proven cause**. | task context; `speedy-rc2.md:120, 223-227`; `10-PREFILL-DECODE-HOUR.md:9-16, 39-43`. |

**Consequence:** a decode-kernel win (Cake, FlashInfer, vLLM FP8/NVFP4 QSA decode) says nothing about
our prefill. Only the QSA **prefill indexer** work touches a path our workload actually spends prefill time in.

---

## 2. Ranked relevance to OUR workload

| Rank | Candidate | Relevance | Why |
|---|---|---|---|
| **1** | **Pennyroyal #162 / upstream #45791 — QSA request-relative prefill indexer** | **Highest — directly targets our dominant case** | Scores each new query row only against its own cached-prefix section (`[rows, max_row_keys]` instead of `[rows, all batch compressed keys]`), removing thousands of one-CTA launches when a batch holds long prefixes. Selection/top-k is unchanged. Exact fit for long cached agent prefixes (94.9 % of tokens). Donor claim ~26 % C1 / 33 % C4 **fresh prefill after 200K cached** — donor-only, never measured on t15. Explicitly **not** a cold-unique-prefix fix (`10-PREFILL-DECODE-HOUR.md:81-84`). |
| 2 | vLLM **#54846** (open) — Triton FP8/NVFP4 QSA for SM120 | Ideas source, medium-low | Same architecture model as our Triton gather→scratch→`tl.dot` path; prefill exists but is decode-centric and unmerged. Borrow only per-shape `num_stages`/`BLOCK_N` and swizzled-V-scale ideas; **do not** change our token-linear scale layout without an accuracy test (FlashInfer wants token-linear). No SM120 speed evidence. |
| 3 | vLLM **#64199** (open) — NVFP4 main KV via FlashInfer | Low | Routes into `flashinfer.sparse.qsa_sparse_paged_attention`, a facade → Cake on 12.0 / reference on 8.x–9.0. Cake's wheel has **0 `csrc/` lines** (not loadable); reference is SM80/90-only. Decode/reference integration, blocked by packaging, not a prefill win. |
| 4 | FlashInfer **#6804** / Cake | Lowest | Decode kernels; Cake = 1 % of decode; SM80/90 guard or unloadable wheel. Numerics anchor only. |

**Bottom line:** #162/#45791 is the only candidate that targets the workload we actually run — but it is
**already in our branch**, so the job is *measurement*, not porting. The other three are decode-side or
blocked ideas sources and must not be credited with prefill gains.

---

## 3. Is #162 already in our branch? — YES (substance), one correctness guard is the only delta

`#162` is a **fork-side backport of upstream `#45791`**, not an independent patch: 175 of its 198 added
`mqa.py` lines (88 %) are line-identical to #45791's 183; the 23 #162-only lines are a
`get_device_core_count` import and hardcoded `T.bfloat16`; its `_prefill_max_row_keys` is a fork shim
for a metadata class lacking the upstream method. Against rc2, #45791 coverage is **794/802 = 99.0 %**;
#162 is **210/250 = 84.0 %** (remainder = weaker variants + fork shims).
(`docs/perf-2026-11-21/03-QSA-PREFILL-CONTENT-CHECK.md:10-45`.)

| Component | rc2 `a090492091` | our branch `fp4local-port` |
|---|---|---|
| `tilelang_qsa_mqa_prefill_relative` / `qsa_mqa_prefill_relative` | present (6 / 13 hits) | present (6 / 13 hits) |
| `sparse_gqa_paged_prefill_triton` (NVFP4 paged-prefill fast path) | present (4 hits) | present (4 hits) |
| `QSAIndexerMetadata.prefill_max_row_keys` / `_prefill_key_splits` | present (3 / 2) | present (3 / 2) |
| empty-key-set guard in `torch_qsa_mqa_prefill_relative` | **absent** | **absent** |

- Dispatch: `relative = not _is_npu and max_row_keys is not None and max_row_keys > 0 and q.is_cuda
  and HAS_TILELANG` (`qsa_indexer.py:465-471`); `prefill_max_row_keys(forward_batch.seq_lens_cpu)`
  is passed at `:667` and `:799`. rc2 carries the stronger fp8-preserving scoring variant, not #162's
  bf16 hardcode.
- **Only missing delta:** empty key set indexed column 0 of a `[rows, 0]` score matrix. Ported on
  `perf/qsa-relative-prefill-20261122` @ **`baca63ae46`** (parent `a090492091`), 2 files / +54−1:
  all-`-inf` `(rows, width)` fp32 return with `width = max(int(width), 1)`, plus a CPU-only regression
  test (`test_relative_prefill_reference_handles_empty_key_set`). Evidence: rc2 `1 failed, 1 passed`
  (`RuntimeError: index 0 out of bounds`); patched test `PASSED`; the 36 other failures are identical
  environmental errors (no CUDA GPU, no `vllm`). Presubmit rc=0.
- `baca63ae46` is **not** an ancestor of `fp4local-port` (checked with `git merge-base --is-ancestor`);
  that branch is based on rc2, so re-porting to `fp4local-port` would be needed if it is wanted there.
- The guard is a **correctness/robustness fix for direct callers**, not a performance change (the indexer
  short-circuits empty compressed keys anyway — see `qsa_indexer.py:481-487`). Upstream #45791 lacks it too.

**Verdict:** no performance patch to port. The measured benefit (~26 %/33 %) remains `donor`/`not-measured`;
no GPU window existed for these docs, and this analysis used no GPU.

---

## 4. Clean isolated test (staged, owner's gpuq window only)

**A/B axis.** Since the relative path is already present, isolate it by toggling the dispatch rather than
porting: candidate worktree = rc2 with a one-line `relative=False` (force `max_row_keys=None/0` in
`select_prefill_tokens`), baseline = untouched rc2. Top-k selection is documented as identical on both
paths, so an index-equality check is a built-in correctness control.

**Two workload arms, never mixed:**
1. **Cold-only:** fresh 8K / 16K / 32K / 64K with `unique_nonce`, nothing resident → tests the case #162
   is *not* supposed to fix (expect no change).
2. **Cached-extension:** resident prefix 50K+512, 150K+1K, **200K+4K**, 500K+1K at C1/C2/C4/C6 → tests
   the donor claim. Measure `prompt_tokens − cached_tokens` per turn from replay with
   `--enable-cache-report`; record `prefill[depth].pp_tps`, device forward elapsed and TTFT.

**Controls (from `30-RUN-COMMANDS.md` §1–§5 and `10-PREFILL-DECODE-HOUR.md` §Deciding experiment):**
- Open gpuq lease (`bin/gpu_window.py active` is the go/no-go signal), production `qwen38-rc2` unchanged,
  idle gate (running=0/waiting=0), single server at a time (one GPU cannot host two ~95 GiB endpoints).
- Pin image `lmsysorg/sglang:nightly-dev-cu13-20261117-25c1c439`, BetterBench `v0.6.0` pinned venv,
  corpus hash `e332dceff176d033`, campaign card sha256 `ef946cf2…`, seed 20261121, 48 req/level.
- Same launch args in both arms (chunked prefill 8192, NVFP4 KV, page 64, context 524288), **power cap
  asserted at 300 W before and after every run** — 300 W vs the donor's claimed 350 W is an open confounder.
- 3 repeats per arm with distinct seeds; arms run sequentially in r1→r2→r3 order to remove thermal drift;
  report median + spread. Record clocks, backend/graph route, and confirm no silent BF16 fallback.
- Do not read decode tok/s as prefill: separate metrics (`decode_weighted_tps` vs `prefill[depth].pp_tps`);
  no live/mixed-load numbers as comparative claims.

**Promotion gate:** cached-extension gain repeatable at ≥2 of C1/C2/C4/C6, cold-only unchanged, top-k
index equality, unchanged correctness/tails; otherwise log as not-measured. Known pre-flight gaps
`todo/661` (8192 hardcode vs 16384) and `todo/662` (level-6 flattening) apply.

---

## 5. Sources
- `docs/perf-2026-11-21/02-PATCH-PORT-MATRIX.md` (rows 162, 45791, 54846, 64199; evidence logs)
- `docs/perf-2026-11-21/03-QSA-PREFILL-CONTENT-CHECK.md` (verdict, coverage, ported delta, evidence)
- `docs/perf-2026-11-21/03-NVFP4-KERNEL-COMPARISON.md` (kernel families, decode focus, M1–M12)
- `docs/perf-2026-11-21/10-PREFILL-DECODE-HOUR.md` (cold vs cached decision, power confounder)
- `docs/perf-2026-11-21/30-RUN-COMMANDS.md` (pins, preflight, baselines, abort rules)
- `bench/DONOR-NVFP4-PROFILE.md:22` (Cake 1.0/1.1 % of decode)
- `docs/releases/speedy-rc2.md:116, 223-227, 251` (94.9 % cache hits, prefill rates)
- `git show 642e9c1448`, `git show baca63ae46`, `git grep` on `a090492091` / `fp4local-port`
