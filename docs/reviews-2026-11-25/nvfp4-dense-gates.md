# NVFP4 dense switches — read-only correctness / benefit audit

**Scope:** source + lab logs only. No repo edits, no GPU, no server, port-8001 benchmark untouched.
**Sources:** `sglang-donor-nvfp4kv/.../quantization/nvfp4_dense.py`, `w4a16_nvfp4_gemv.py`, `fp8.py:854-861`, `linear.py:993-997`, model prefix constructors; `/srv/community-repos/hyperburst-rtxpro6000/docs/lab-notes/N1_LOG.md`, `EXCLUSIVE_TIME_MAP.md`; `/tmp/opencode/analysis/w8a16-shapes.md`.
`N1_LOG.md` is not in `/tmp/opencode` — found under `/srv/community-repos/.../docs/lab-notes/N1_LOG.md`.

## 1. Exact module prefixes matched

`_cat_flags()` (nvfp4_dense.py:311-320) parses env with `on()` = lower-case in `{"1","true","yes","on"}`; default off.

| Switch | Category string `c` |
|---|---|
| `SGLANG_LINEAR_ATTN_NVFP4=1` | `"in_proj_qkvz"` |
| `SGLANG_ATTN_NVFP4=1` | `"qkv_proj"` |

Both on → `cats = ("in_proj_qkvz", "qkv_proj")`. Match rule (line 343):

```python
if not any(prefix.endswith(c) or f".{c}" in prefix for c in cats): return False
```

- `in_proj_qkvz` → `model.layers.N.linear_attn.in_proj_qkvz` (built via `add_prefix("in_proj_qkvz", prefix)`, qwen3_next.py:144). Matches by `endswith`; the `".{c}" in prefix` arm also catches a wrapped prefix with a trailing component. **Only** `SGLANG_LINEAR_ATTN_NVFP4` matches it.
- `qkv_proj` → `model.layers.N.self_attn.qkv_proj` (`add_prefix("qkv_proj", prefix)`, qwen3_next.py:678-692; fused from q/k/v by the loader, qwen4_exp.py:2506-2508). **Only** `SGLANG_ATTN_NVFP4` matches it — `"in_proj_qkvz"` neither ends with `"qkv_proj"` nor contains `".qkv_proj"`, so the two flags are disjoint.
- `out_proj` (`model.layers.N.linear_attn.out_proj`) matches neither category → deliberately excluded. Code comment (lines 327-329): 1.08-1.13× slower, 47-52 % of roof, 0.22 % of the W16 step, below break-even (N1_LOG §7/§9).
- Additional guards: `w.dim()==2`, dtype bf16/fp16 (checkpoint-FP8 weights are skipped), `K % 16 == 0` (both K=2560 pass; 2560/16=160).
- **Prefix availability is a hard prerequisite for the qkv flag:** `QKVParallelLinear` did not set `self.prefix` originally, so all attention qkv layers presented `""` and were silently skipped (N1_LOG §14.4). Fix is present in this donor (`linear.py:993-997`, comment cites that exact bug), so `SGLANG_ATTN_NVFP4` will now actually fire. N1's standing lesson (§14): *check the flag fired before reading any A/B number.*

## 2. Layers with no `.prefix`

`prefix = getattr(layer, "prefix", "") or ""`; if empty → `logger.warning("NVFP4 dense: %s has no .prefix, cannot match it against %s", ...)` and `return False` → layer stays FP8 (lines 334-342). It is a warning, not an exception; the comment says "Never silently skip", but the operational effect is still a skip with a log line.

- N1_LOG §13: 99 no-prefix layers logged (`RowParallelLinear`, `ParallelLMHead`) — the o_proj/out_proj/down_proj family that N1 deliberately skips, so nothing intended was missed.
- N1_LOG §14.4: the same gap silently cost attention qkv its first run; after the `linear.py` fix the matcher warns instead of quietly declining.

## 3. Cache key derivation and stale-hit risk

Cache is **not used by either switching gate**. `maybe_pack_dense_nvfp4` calls `quantize_nvfp4(w.data)` directly on the in-memory BF16 fused tensor at the top of `Fp8LinearMethod.process_weights_after_loading` (`fp8.py:854-861`), before FP8 quantisation; no `build_or_load`, no disk key. The disk cache (`nvfp4_dense.py:41-171`) serves only lm_head callers: `eagle_worker_v2` draft head (`rows=self.hot_token_id`), target head (`rows=None`), `mtp_recur_head` (`token_ids`), `draft_lru_vocab` (`hot_ids`).

- Cache file: `$SGLANG_NVFP4_CACHE_DIR` (default `~/.cache/flash-next-fast/nvfp4`) / `{tensor_name with /→_}.{key}.pt`, written via tmp + `os.replace` (atomic).
- Key (lines 82-100): `sha256("|".join([realpath(model_dir), tensor_name, shard_sig, rows_sig]))[:32]`
  - `shard_sig = f"{basename}:{st_size}:{int(st_mtime)}"` of the shard found via `model.safetensors.index.json` weight_map (or header scan fallback).
  - `rows_sig = "all"` if `rows is None`, else `f"{numel}-{sha256(rows.int64 bytes)[:16]}"`.
- **Current benchmark arm (`SGLANG_LINEAR_ATTN_NVFP4=1`) has zero cache interaction** → no stale hit is possible for its dense packing. `~/.cache/flash-next-fast/nvfp4` was empty at audit time.
- If lm_head flags are added: benchmark model dir `/srv/llm-models/Qwen3.8-Flash-Next-NVFP4/` with token map `hot2_49152` (49152 rows) reproduces the same key as prior arms — that is the *intended* warm hit and is correct as long as shard bytes are unchanged. The key changes with dir path, tensor name, shard basename/size/mtime, or row indices. Residual weakness: shard identity is size + integer-second mtime, not a content hash; a same-size rewrite within one second would evade it. `mtp_recur_head` additionally re-opens the shard on a hit and requires BF16 dtype + exact shape, and requires a private cache dir.

## 4. Accuracy risk

- **Source-of-quantisation is correct for both switches.** Code comment lines 11-17 and N1_LOG §4: quantising the already-FP8 head into NVFP4 compounds two lossy steps and costs **~2 points of argmax agreement** vs BF16 (0.8164 from BF16 vs 0.7969 from FP8, Δ≈0.0195). The stage-C hook sees the true BF16 fused tensor (loader-side fusion of `in_proj_qkv`+`in_proj_z` and `q/k/v`), so it avoids that loss; a post-load re-read would have to replicate the fusion mapping.
- N1_LOG §4 probe (real `lm_head`, 49152 hot rows, 256 random hidden states): fp8 0.9492 / nvfp4-from-BF16 0.8164 / fp8+nvfp4 0.7969 argmax agreement; NVFP4 puts **3.6× the logit noise** on a head. These are random-input flip rates, not a served-acceptance prediction.
- N1_LOG §6 acceptance gate (≤1 % loss at k=7) ran on private data; **no result published** → unquantified acceptance risk on head paths. Stage C does not touch the heads, so that gate does not directly transfer.
- Kernel numerics are sound: bit-exact dequant (0/5632 one-hot mismatches), GEMV RMS 1.66e-3 vs FP8 1.65e-3 (§7); deterministic packer matched the offline emulator bit-exactly after the global-scale-expression and ties-to-even fixes (§2).
- Stage-C quality evidence (§13): needle PASS, greedy agreement **1.000** (code-edit, first 256 tokens) for the qkvz arm. The NO-GO there was performance, **not accuracy**.

## 5. Expected speedup vs the stated profile

Profile (raw per-step, trace re-aggregation `w8a16-shapes.md` + task figures; W4 step ≈10 ms):

| Family | grid | calls/step | µs/call | ms/step |
|---|---|---:|---:|---:|
| GDN `in_proj_qkvz` | 256 | 36 | 29.7 | **1.07** |
| attn `qkv_proj` | 416 | ≈16.7 | 25.4 | **0.42** |

Measured kernel speedups (N1_LOG §7, §13): qkvz **1.269×** (M=16) to **1.449×** (M=1), in-server **1.530×** (29.28→19.14 µs); attn qkv **1.176×** (M=16) to **1.319×** (M=1). No in-server measurement isolates qkv: `bench/n1/{ab,qual}.sh` arm `c` sets **both** `SGLANG_LINEAR_ATTN_NVFP4=1 SGLANG_ATTN_NVFP4=1` together, so the measured stage-C result already includes qkv_proj packing (after the prefix fix made the flag fire).

- **Ideal, if fully on the critical path:** qkvz 1.07→0.70-0.85 ms (save 0.22-0.37); qkv 0.42→0.32-0.36 (save 0.06-0.10); combined save ≈**0.28-0.47 ms = 2.8-4.7 %** of a 10 ms step.
- **Realized (measured stage-C arm = both flags, §13):** qkvz −459 µs exclusive, but step wall **10497→10550 µs (≈0, slightly worse)**. Since the arm included qkv_proj too, this is already the combined verdict for adding the ATTN flag to the running LINEAR_ATTN arm. Two causes: (1) the qkvz GEMV had been *covering* the MoE GEMMs — its exclusivity fell 88.5 %→75.4 %, exposing +348/+171 µs on GEMM1/GEMM2, which are at the expert-weight roofline (§0/§5); (2) `NVFP4DenseLinearMethod` has **no `apply_into_split`** (confirmed: only `apply`/`apply_into`/`_wide_m`), so the two-destination qkvz/ba fusion falls back to a separate split/reshape/cat kernel, **+115 µs**. (2) is fixable; 115 µs ≪ 519 µs of MoE exposure.
- Net expectation for adding the qkv flag: at most ~1 % of step in the ideal case, realistically ≈0 or negative, below A/B noise (one server start moves ms/step 1-3 %, whole-step wall ±5 %, §11/§13).

## 6. Go / no-go for `SGLANG_ATTN_NVFP4` as a follow-up arm

**NO-GO as a ship arm.** N1's final recommendation is Stage C (qkvz + attn qkv, both flags) = **NO-GO, ≈0 gain** (§463, §470-472), and that measured arm already had `SGLANG_ATTN_NVFP4=1` on after the prefix fix — so adding qkv to the current LINEAR_ATTN-only benchmark has direct negative evidence, not just a projection. qkv is the smaller half of C (0.42 vs 1.07 ms), its absolute ceiling (~1 % of step) is below measured noise, and the combined arm showed zero step benefit despite a 1.53× kernel win. If the line is ever revisited: first restore `apply_into_split`, verify the flag fired, and read per-kernel CUPTI medians — not whole-step wall clock.
