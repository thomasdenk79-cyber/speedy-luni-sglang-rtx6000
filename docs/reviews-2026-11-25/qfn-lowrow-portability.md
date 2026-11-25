# QFN-0193..0202 → donor HEAD: portability assessment

Status: read-only analysis, 2026-11-25. No repo/worktree edited, no GPU used, port 8001 untouched.

## Scope and evidence

| Item | Value |
|---|---|
| Donor | `/tmp/opencode/sglang-donor-nvfp4kv` (the task's `/tmp/opencode/sglang-donor-nvfp4-kv` does not exist), branch `donor-nvfp4-kv`, HEAD `dc333b6fc6980c25ca88ff6b54f5d00bb494ef1e`, clean worktree |
| Qualified arm | NVFP4 KV + Cake decode + SM120 SwiGLU, `SGLANG_FLASHINFER_MOE_FUSED_FINALIZE=0` at runtime; `serve-fast.sh` defaults `SGLANG_FP8_W8A16_GEMV=1`, `SGLANG_ROUTER_GEMV=1`, `SGLANG_NORM_INTO_GEMV=1`, `SGLANG_OPT_DRAFT_MOE_GEMV=1`, `SGLANG_MTP_FC_GEMV=1` (last one gated by `HB_BUNDLE2` + profile conditions, `serve-fast.sh:345`) |
| QFN registry | `qwen38-flashnext-nvfp4-kv/docs/PATCHES.md:228-237` and `:1211-1450` |
| Real QFN commits | present in `/srv/community-repos/sglang-upstream`, branch `perf/hb-lowrow-kernels`, not in the qwen38 checkout (registry only). Verified with `git show`/`git diff` on all ten hashes: `26c45b4dd1ed`, `5ddbc4809b2a`, `e2c0661b9c55`, `ae7a43029fe0`, `3d55b2262c4f`, `ef6828485882`, `171f40d10c34`, `dd2725ba417a`, `731e3b146d10`, `4eb9a38b69db`. No checkout performed. |

## Headline finding

The donor **already carries functional equivalents of all four perf patches**, because the QFN series is itself a re-port of the HyperBurst donor work onto the qwen38 tree:

| QFN feature | Donor equivalent | Evidence |
|---|---|---|
| 0193 draft MoE W4A16 GEMV | `72de55f DG1: one-token draft MoE …` + `python/sglang/srt/layers/moe/draft_moe_gemv.py` (451 lines), `quantization/w4a16_nvfp4_gemv.py` (533 lines), `modelopt_quant.py` runner attr + CUTLASS hook, `qwen2_moe.py:436` enable call, `environ.py:1170` flag | donor git log + file reads |
| 0194 MoE router GEMV | `router_gemv.py` (96 lines) + `qwen2_moe.py:747-780 _forward_router_logits`, which already does QFN's final split: bias-free → tuned skinny `bf16_gemv`, biased → dedicated FP32 `router_gemv`; `environ.py:324` flag | `qwen2_moe.py:758-775`, `bench/glue_e/bench_router_gemv.py` |
| 0195 GDN norm into out_proj GEMV | `w8a16_gemv.py:712-850` fused-norm prologue + `_norm_plan` + `w8a16_gemv_norm_gated*`, `fp8.py:1034 Fp8LinearMethod.apply_norm_gated`, `qwen3_5.py:1082 _norm_out_proj_fused`; flag is a module-level read (`w8a16_gemv.py:733`, `qwen3_5.py:39`), enabled by `serve-fast.sh:262` | donor file reads; `bench/h1/h1_check.py` |
| 0196 MTP entry fc skinny GEMV | `qwen4_exp_mtp.py:33 _MTP_FC_GEMV` + `:311-328` calls `w8a16_gemv.bf16_gemv` for both fc projections; `serve-fast.sh:345` exports the flag | donor file reads |

**Therefore this is mostly a guard backport, not a feature port.** The portable value is:

1. **Six correctness fixes absent from donor** (0197, 0198, 0199, 0200, 0201, 0202) — all small, low-conflict.
2. **One genuine numerical delta from 0193**: `G_UP_OFF`. Donor's `_compute_gemm1_alphas` (`modelopt_quant.py:2333-2356`) splits `w13_weight_scale_2[:, 0]` (gate) and `[:, 1]` (up) for [E,2] checkpoints, but donor's draft GEMV applies column 0 to *both* halves (`draft_moe_gemv.py:19,166,214`). QFN passes `G_UP_OFF=1` when the scale has ≥2 columns. If the served checkpoint has distinct gate/up global scales, this is a real correctness gain; if it is 1-D/[E,1], it is a no-op.

What must **not** be ported blindly: QFN's PDL no-op stub (donor has a live `python/sglang/kernels/triton_pdl.py`; QFN explicitly stubs it because its tree lacks it), QFN's `sm120_bf16_skinny_gemv` kernel (donor's equivalent is `w8a16_gemv.bf16_gemv`, no such file/flag), and QFN's MXFP8 norm module (donor's served path is W8A16 FP8).

## Priority by expected benefit (decode profile)

Given w8a16 GEMV already dominates at 27-37 % of GPU time and is already harvested by donor's flags, these patches only remove *adjacent* launches. Ranking uses QFN/donor measured numbers (all pending GPU A/B in the QFN registry; treat as estimates):

| Rank | Item | Measured claim | Donor state | Action |
|---|---|---|---|---|
| P0 | 0193 draft MoE GEMV | 49.4→29.8 µs/call; 54.5→30.4 µs behind L2 flush; **-0.27..-0.34 ms / 15-step verify** | ON by default, kernel present | Backport `G_UP_OFF` + fixes 0198/0201/0202. Highest payoff, but changes draft tokens/acceptance |
| P1 | 0194 router GEMV | 6.18→4.42 µs/call @M=16; **-86 µs/step, 63 fewer kernels** (49-61 calls/step) | ON by default | Backport 0197 + 0200 only |
| P2 | 0195 fused norm | -0.95 µs mean / -1.06 µs median per call; **-36 µs/step, 36 fewer kernels** | ON by default on W8A16 path | Backport 0199 analogue only; skip MXFP8 module unless checkpoint proves MXFP8 |
| P3 | 0196 MTP fc | 1.52 µs @M=1, 1.93 µs @M=4 flushed; **-6.5..-14 µs/draft step** | ON under `HB_BUNDLE2` profile conditions | No port; flag already independent of router |
| — | hyperconnection ~10 % | none of QFN-0193..0202 targets hyperconnection | unaffected | out of scope |

Correctness fixes have no speed claim but should ride with the kernel they guard, in the order above.

## Per-QFN detail

### QFN-0193 — draft MoE W4A16 GEMV (8 files, +1104)

- **Touches**: `docs/docs/references/environment_variables.mdx`, `srt/environ.py`, `srt/layers/moe/draft_moe_gemv.py` (new 506), `srt/layers/quantization/modelopt_quant.py`, `srt/layers/quantization/w4a16_nvfp4_gemv.py` (new 49), `srt/models/qwen2_moe.py`, `scripts/bench_draft_moe_gemv.py` (new 110), `test/registered/unit/moe/test_draft_moe_gemv.py` (new 415).
- **Donor**: all six runtime files exist; the two new kernel files exist with more content (451/533 lines), the env flag, the `modelopt_quant` runner attr + CUTLASS dispatch hook, and the `qwen2_moe` enable call are already present. `scripts/bench_*` and `test/registered/unit/moe/` do not exist; donor has `bench/n1/bench_nvfp4_gemv.py` and `test/srt/layers/test_w8a16_gemv_silu.py`.
- **Conflicts**: (a) QFN deletes `from sglang.kernels.triton_pdl import …` and defines no-op `pdl_wait`/`pdl_trigger`; donor imports the real module and passes `launch_pdl=PDL` — porting the stub is a regression (donor measured PDL ≈0.2 µs of the 19.6 µs saving). (b) `G_UP_OFF` constexpr, kernel signature, call site and docstring are absent in donor. (c) cosmetic: `Optional` vs `| None`, warning text, `DEFAULT_CONFIG` comment. (d) bench/test paths.
- **Minimal port**: keep donor's kernel; apply only the `G_UP_OFF` hunk (add constexpr to `_k1`, load `g_ptr + e*g_stride + G_UP_OFF`, pass `1` when `w13_weight_scale_2.dim()==2 and shape[1]>=2` else `0`) after verifying the checkpoint scale shape. Add CPU tests under `test/srt/layers/` or create `test/registered/unit/moe/`; skip the bench script (reuse `bench/n1/`).
- **Redundant**: 7 of 8 files; the kernel, primitives, flag and both hooks are donor-native.

### QFN-0194 — router gate as GEMV (6 files, +695/-3)

- **Touches**: mdx, `environ.py`, `router_gemv.py` (new 223), `qwen2_moe.py`, `scripts/bench_router_gemv.py`, `test/registered/unit/moe/test_router_gemv.py`.
- **Donor**: `router_gemv.py` exists (kernel only, no `router_gemv_shape_ok`/`applies`/`reference` helpers or `MAX_M`/`BLOCK_*` constants); `qwen2_moe.py` already inlines the same predicate and routes bias-free to `bf16_gemv`, biased to `router_gemv`; flag exists; `bench/glue_e/bench_router_gemv.py` is the donor bench.
- **Conflicts**: porting QFN's helpers requires rewriting donor's `_forward_router_logits` (which also gained `SGLANG_ROUTER_FAST_TOPK`, packed 32-bit keys and shared-gate-early paths since). QFN's test module path does not exist.
- **Minimal port**: none for the feature. Fold 0197 + 0200 into donor's existing code (see below); optionally lift `router_gemv_applies`/`reference` as pure helpers for CPU tests without changing the hook.
- **Redundant**: feature 100 %. Bench is a different tree but measures the same thing.

### QFN-0195 — GDN norm into out_proj GEMV (7 files, +1436)

- **Touches**: mdx, `environ.py`, `fp8.py` (+76), `mxfp8_norm_gated_gemv.py` (new 511), `qwen3_5.py` (+58), `scripts/bench_mxfp8_norm_gated_gemv.py`, `test/registered/unit/layers/test_mxfp8_norm_gated_gemv.py`.
- **Donor**: no `mxfp8_norm_gated_gemv.py`, but the equivalent fold lives in `w8a16_gemv.py:712-850` with its own `_norm_plan`, tuned shapes `(4|16, 2560, 6144, 128)` and server A/B table; `fp8.py:1034 apply_norm_gated` and `qwen3_5.py:1082 _norm_out_proj_fused` already implement the hook; `bench/h1/h1_check.py` covers the kernel.
- **Conflicts**: QFN's `apply_norm_gated` takes `activation: str = "swish"` and requires MXFP8 e4m3 + UE8M0 `weight_scale_inv` with a flashinfer-cutlass/cutedsl dense backend; donor's takes `sigmoid_gate: bool` and targets W8A16 FP8 with an fp32 per-channel scale. Cherry-picking either hunk replaces a working path and breaks the signature at the `qwen3_5.py` call site. QFN also registers a new env var in `environ.py`; donor reads `SGLANG_NORM_INTO_GEMV` via `os.environ` at import.
- **Minimal port**: for the qualified (W8A16) arm, port nothing but the 0199 stride guard into `w8a16_gemv_norm_gated_supported`. If a genuinely MXFP8 dense checkpoint is served, add the module under a new name and branch `Fp8LinearMethod.apply_norm_gated` by `use_mxfp8`/backend instead of overriding it; do not double-register.
- **Redundant**: high for the served arm; QFN's value is extra MXFP8 coverage only.

### QFN-0196 — MTP entry fc skinny GEMV (6 files, +384/-7)

- **Touches**: mdx, `environ.py`, `unquant.py` (+39), `qwen4_exp_mtp.py` (+42/-7), `scripts/bench_mtp_fc_gemv.py`, `test/registered/unit/models/test_qwen4_exp_mtp_fc_gemv.py`.
- **Donor**: no `SGLANG_SM120_BF16_SKINNY_GEMV`/`sm120_bf16_skinny_gemv.py`; `qwen4_exp_mtp.py:33` already reads `SGLANG_MTP_FC_GEMV` at import and `:311-328` already calls `bf16_gemv` with the same bias/compile/batch-invariant guards. `serve-fast.sh:345` sets the flag independently of the router. `test/registered/unit/models/` exists; `scripts/` has no bench gemv.
- **Conflicts**: QFN refactors `unquant.py` into `_bf16_skinny_linear_if` + `maybe_mtp_fc_skinny_linear` keyed on `envs.SGLANG_MTP_FC_GEMV` and an `is_sm120()` helper that donor's `srt` does not have. Donor's MTP entry also has `SGLANG_MTP_ENTRY_FUSED` + `mtp_entry_fused` ahead of the GEMV path, which QFN's `_fc()` abstraction does not know.
- **Minimal port**: none for speed. If central flag registration is wanted, add `SGLANG_MTP_FC_GEMV = EnvBool(False)` to `environ.py` without changing the model hook; note donor reads it via `os.environ` at import, so an EnvBool addition alone does not change behaviour.
- **Redundant**: ~100 %.

### Fixes (QFN-0197..0202)

| QFN | QFN file(s) | Donor state / analogue | Likely conflict | Minimal port |
|---|---|---|---|---|
| 0197 router K alignment | `router_gemv.py` (+22/-5 in file + test) | Same bug: donor `router_gemv.py:57` requires `k % 128 == 0` while `BLOCK_K=256` at launch; `qwen2_moe.py:760` also checks `% 128` | QFN patched only the helper (its `qwen2_moe` delegates to `router_gemv_applies`); donor has two independent gates | Change both: `k % 256` in `router_gemv.py`, `hidden_states.shape[1] % 256` in `qwen2_moe.py`; introduce a local `BLOCK_K` constant if desired. No production shape affected (K=2560) |
| 0198 routes + grad | `draft_moe_gemv.py` (+90/-1) | Donor `_is_one_token_call` (`:335`), `__init__` (`:352`), `_get_workspace` (`:383`) lack all three guards | Function names identical; only `Optional`→`| None` cosmetics | Apply in QFN order: grad conjunct in predicate; `_warned_routes` + `routes > act.shape[0]` early return in `_get_workspace`. Hunk maps 1:1 |
| 0199 row-contiguous weights | `mxfp8_norm_gated_gemv.py` (+25) | Donor analogue `w8a16_gemv.py:790 w8a16_gemv_norm_gated_supported` checks `w.stride(1)==1` but not `w.stride(0)==K` | Different file/function; verify served weight layout before assuming `stride(0)==K` is always true | Add `and w.stride(0) == K` to the donor predicate (plus doc note). If MXFP8 module is added, apply QFN hunk verbatim there |
| 0200 biased router sm_120-only | `qwen2_moe.py` (+28/-1) | Donor's `_forward_router_logits` has **no** arch check in either branch; donor `srt` has no `is_sm120()` helper | QFN imports `is_sm120` from `kernels/ops/gemm/sm120_bf16_skinny_gemv.py` which does not exist in donor | Add arch conjunct to **both** branches using `get_device_capability()`/`torch.cuda.get_device_capability()==(12,0)` or a small local helper; QFN only needed the biased branch because its bias-free branch was already gated |
| 0201 activation width | `draft_moe_gemv.py` (+55) | Donor `maybe_apply` (`:356`) lacks the check | Maps 1:1 | Add `_warned_width` + `hidden_states.shape[1] == layer.w13_weight.shape[2]*2` early return before `_get_workspace` |
| 0202 activation rank | `draft_moe_gemv.py` (+65/-5) | Donor `_is_one_token_call` checks only `x.shape[0]==1`; no topk rank/shape agreement | Maps 1:1 **but must be applied after 0198**, because it rewrites the grad lines introduced there | Add `x.dim()==2`, `topk_ids.dim()==2`, `topk_weights.dim()==2`, `topk_weights.shape == topk_ids.shape` |

## Minimal port sequence (recommended if authorised)

1. No checkout: create a fresh donor worktree/branch off `dc333b6`; keep the port 8001 benchmark and its tree untouched.
2. `draft_moe_gemv.py`: 0198 → 0201 → 0202 (QFN commit order, avoids hunk overlap), then the `G_UP_OFF` subset of 0193 **only if** `w13_weight_scale_2` is [E,2] with unequal columns on the served checkpoint.
3. `router_gemv.py` + `qwen2_moe.py`: 0197 then 0200 (patch both gates; donor differs from QFN here).
4. `w8a16_gemv.py`: 0199 analogue (skip if the norm fold is not actually used).
5. Skip 0194/0195/0196 feature code and all four bench scripts. Optionally add the missing `environment_variables.mdx` rows and `environ.py` declarations for discoverability only.
6. Ports of the CPU tests: `test/registered/unit/moe/` is absent in donor (create it or move to `test/srt/layers/`); `test/registered/unit/{layers,models}/` exist. Keep donor-identical test structure otherwise.
7. Validate offline gates first; any GPU A/B is a separate authorised step — none of these patches has passed GPU validation in the QFN registry, and the A/B campaign in the QFN repo explicitly ended `BLOCKED` with no GPU run.

## Bottom line

Portability is **high but mostly unnecessary**: the donor HEAD is the upstream of this work and already ships all four kernels plus the launch-site wiring and default-on flags. A "port of QFN-0193..0202" onto `dc333b6` should be scoped as six small guard fixes plus one optional `G_UP_OFF` correctness hunk — roughly **150 lines across four files**, not the ~3600 lines of the QFN series. The dominant w8a16 GEMV share (27-37 %) is unaffected by these patches; the expected wins are the already-enabled -0.27..-0.34 ms draft MoE and -86 µs router savings, which can be lost by accidentally replacing donor's PDL kernel, its `bf16_gemv` router path, or its W8A16 fused-norm hook with the QFN variants.
