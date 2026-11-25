# QSA long-cached-prefix prefill: content check #162 vs #45791 vs rc2

Answers the `PORT if missing` / `content check first` decision of
[02-PATCH-PORT-MATRIX.md](02-PATCH-PORT-MATRIX.md) rows 162 and 45791. Written 2026-11-22 by the
`2026-11-21-speedy-pennyroyal-qsa162` run. No GPU was used: no `state/gpu-window/lease.json`
existed and the device was busy, so every benefit label here stays `donor` or `not-measured`.

## Verdict

`#162` is a **fork-side backport of upstream `#45791`**, not an independent patch. Measured: of the
198 added lines #162 puts into `qsa/mqa.py`, 175 (88 %) are line-identical to the 183 lines #45791
adds there; the 23 #162-only lines are the `get_device_core_count` import and the hardcoded
`T.bfloat16` operand declarations. In `qsa_indexer.py` the overlap is 22 of 64 lines and the
#162-only block is the `_prefill_max_row_keys` shim, whose own docstring reads "Ported from upstream
``QSAIndexerMetadata.prefill_max_row_keys`` (#45791)" (pr162.diff:288). `rc/qwen38-rc2` already
carries the complete request-relative prefill path — kernel, dispatch, fused compress, padded
metadata fast path and the NVFP4 paged-prefill table. One defect was genuinely missing and is now
ported.

| Item | State in `rc/qwen38-rc2` (`a090492091`) | Action |
|---|---|---|
| relative TileLang prefill kernel + `*_relative` exports | present, stronger variant (`_TILELANG_DTYPES` mqa.py:115, `_scoring_dtype` :121 keep fp8; #162 hardcodes `T.bfloat16`) | nothing to port |
| dispatch + fused compress + padded metadata | present (`_num_sms` :561 instead of #162's `get_device_core_count`) | nothing to port |
| `prefill_max_row_keys` sizing | method `QSAIndexerMetadata.prefill_max_row_keys(host_sequence_lengths=None)` (`metadata.py:130`), both call sites pass `forward_batch.seq_lens_cpu` (`qsa_indexer.py:667`, `:799`); #162's free function `_prefill_max_row_keys` is a fork shim for a metadata class that lacks it | nothing to port |
| NVFP4 paged-prefill table | present in `qwen_sparse_attn_backend.py:1849` (`sparse_gqa_paged_prefill_triton` fast path) with local `layer_router.read(...).key()` / `kv_shard_slot_table` adaptations | nothing to port |
| empty-key-set guard in `torch_qsa_mqa_prefill_relative` | **absent** — reference indexed column 0 of a `[rows, 0]` score matrix | **ported**, see below |

Upstream duplicate check: the guard is *not* in #45791 either (upstream leaves the empty key set
crashing), so porting it does not re-apply upstream work.

## Coverage measurement

Added lines of each PR diff matched line-by-line against the worktree tree
(`/tmp/opencode/qsa162/hunkcheck.py`, tool of this run). Numbers below are from the patched tree,
so the guard counts as covered.

| Diff | Added lines | Present in rc2 | Coverage | Note |
|---|---|---|---|---|
| #45791 (`642e9c1448`) | 802 | 794 | 99.0 % | 8 remaining are fork-context lines (`pool.get_key_buffer`, monkeypatch args) |
| #162 (`dfd469c3a7`) source | 250 | 210 | 84.0 % | remainder = weaker variants + fork shims above; on the untouched rc2 tip it is 83.6 % |

All four #162-only names (`_DeviceRows`, `_dispatched_row_counts`, `relative_scorer`,
`absolute_scorer`) are test-local helpers, not shipped code. Both diffs fail
`git apply --check` forward *and* `--reverse` against rc2 (`rc=1`), which is why the row said
`diverged` and a content check was required.

## Ported delta

Branch `perf/qsa-relative-prefill-20261122`, commit `baca63ae46` (parent `a090492091`), worktree
`/srv/community-repos/sglang-wt-qsa162` — 2 files, +54/-1:

- `python/sglang/srt/layers/attention/qsa/mqa.py`: `torch_qsa_mqa_prefill_relative` returns the
  all `-inf` `(rows, width)` fp32 layout when the key block is empty and applies the kernel
  helper's floor `width = max(int(width), 1)`; the absolute reference keeps validating operands,
  so an empty key set is checked exactly like a full one.
- `test/registered/kernels/ops/attention/qsa/test_qsa_indexer.py`:
  `test_relative_prefill_reference_handles_empty_key_set` — CPU-only, no device needed, covers
  shape/dtype, the width floor, zero rows, and that a wrong head dimension still raises.

Attribution to #162 (`dfd469c3a774789ad0b3476386fdfd9df5a42298`, merged 2026-11-06) sits in the
code comment and the test docstring.

## Evidence (red/green, no GPU)

Command (host `t15`, borrowed venv has torch 2.13.0+cu130 and tilelang 0.1.12, `pytest` installed
into `/tmp/opencode/qsa162/pylibs`):

```bash
bash /tmp/opencode/qsa162/run-cpu-tests.sh /srv/community-repos/sglang-wt-qsa162 \
  test_relative_prefill_reference_handles_empty_key_set
```

| Tree | Result |
|---|---|
| rc2 tip `a090492091` (no guard) | `1 failed, 1 passed` — `RuntimeError: index 0 is out of bounds for dimension 1 with size 0` at `mqa.py:474` |
| patched `baca63ae46` | new test `PASSED` |
| full file, base | `37 failed, 1 passed` |
| full file, patched | `36 failed, 2 passed` |

The 36 failures are identical on both trees and purely environmental: `RuntimeError: No CUDA GPUs
are available` (device-gated parity tests, 8 of them for the relative path) and
`ModuleNotFoundError: No module named 'vllm'` (reference imports of the borrowed venv). Failure
sets differ only by the new test, so nothing regressed. Presubmit at landing: `rc=0`
(`presubmit code --staged --root /srv/community-repos/sglang-wt-qsa162`, 15 checks, 1 warning for
the `test/` layout with the goal suite proven here).

## GPU part: still open

The cached-prefix matrix (50K+512, 150K+1K, 200K+4K, 500K+1K at C1/C2/C4/C6) and the donor claim
of ~26 % (C1) / 33 % (C4) stay `not-measured`. [30-RUN-COMMANDS.md](30-RUN-COMMANDS.md) is the
executable form; card `bench/campaign-20261121.json`
(sha256 `ef946cf2…`), jobs in `bench/jobs/*.env`, driver `bench/bb-campaign.sh`. Pre-flight:
`python3 bin/gpu_window.py status` must show an open lease, `nvidia-smi -pl 300` per plan,
`qwen38-rc2` untouched. Known gpuq gaps that bite before any of these runs: `todo/661` (8192
hardcode vs 16384 context) and `todo/662` (level-6 flattening). Do not write numbers into
[RESULTS.md](RESULTS.md) without raw data behind them.
