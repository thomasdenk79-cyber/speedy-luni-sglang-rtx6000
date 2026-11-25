# Upstream cherry-pick shortlist (read-only scan)

**Source:** `/srv/community-repos/sglang-upstream`, `upstream/main` @ `563813c0d6` (2026-11-23).
**Window:** commits dated 2026-11-01 .. 2026-11-23. **Target:** donor branch `/tmp/opencode/sglang-donor-nvfp4kv` @ `dc333b6`, based on fork tip `8edd1fa` (2026-11-12).
**Constraint:** `git merge-base 8edd1fa upstream/main` is empty in the local mirrors — there is no usable common ancestor, so there is no rebase path; only cherry-pick/port. This matches the operator report "Kein Rebase auf upstream/main".
**Method:** `git log` + `git show --stat/--format` only; no checkout, no edits, port 8001 benchmark untouched.
**Difficulty basis:** drift = touched files that already differ between donor base `8edd1fa` and the commit's parent. "needs port" means conflict-prone, not impossible.

## Shortlist (max 10)

| # | Hash | Title (upstream) | Mechanism (one line) | Difficulty |
|---|---|---|---|---|
| 1 | `0f1509829b` | `[Fix] moe: Pad NVFP4 trtllm hidden size to a multiple of 256` | TRT-LLM NVFP4 MoE requires hidden %256 (512 for per-token non-gated); misaligned dims run silently wrong, so pad activations in BF16 and suppress FP4 dispatch when padding. | **needs port** (4 files; 1919+/582- drift in `flashinfer_trtllm.py` + `modelopt_quant.py`) |
| 2 | `03d4f49f22` | `[Perf] gdn: Remove host stalls from GDN prefill` | Prewarms every FlashInfer extend variant (checkpoint mode × seq count) before graph capture and keeps state fixups on-device, removing per-step host synchronization. | **needs port** (8 of 10 touched files drift) |
| 3 | `ff68baae6d` | `[Perf] gdn: Run GDN prefill in place on a unified-memory state pool` | Stride-aware slot addressing plus `PoolStaging` gather/scatter lets prefill read/write pooled SSM/conv state in place instead of making packed copies. | **needs port** (13 of 14 touched files drift) |
| 4 | `5aee953e1a` | `[Fix] [NVIDIA] DSA: Fix BF16 TileLang shared memory overflow on SM120` | Adds `head_tile=16`/`block_I=32` for sm_120, `d_v=512`, `tail_dim=64` and bounds the cooperative loader to fit the 99 KiB SMEM limit. | **needs port** (kernel file rewritten: 223+/1385- vs donor base) |
| 5 | `e7c382c129` | `[Fix] [NVIDIA] moe: keep tuned autotune key for deferred-finalize FP4 TRT-LLM calls` | Passes a dummy symmetric BF16 output buffer so FlashInfer's autotune lookup keeps hitting the tuned kernel (workaround for flashinfer#6077). | **needs port** (2 files; both drifted) |
| 6 | `077acd49a1` | `[Perf] attention: Avoid host synchronization in GDN MIS state resets` | Replaces host-driven state resets with device-side `index_fill_` on conv/SSM pools using int64 indices. | **needs port** (2 files) |
| 7 | `304a6016b3` | `[Perf] gdn: Fuse MIS mapped gates into QKV preparation` | `fused_qkv_gating_gdn_prefill` produces `g`/`beta` inside QKV prep, eliminating a separate gating pass and its staging. | **needs port** (5 of 5 files) |
| 8 | `775db052b6` | `[Perf] [NVIDIA] runtime: Blackwell/Rubin optimizations for hybrid GDN + QSA + MoE` | Large bundle: multi-CTA verify sampling, GDN ReplaySSM spec decode, Mamba state scatter/transfer, cache residency and MoE-side runtime plumbing. | **drop** as an upstream/main pick (reverted); use local reapply `108c00266f` only after review — **needs port** (66 files, 53 drift) |
| 9 | `a8ac86595c` | `[Perf] beam: Use the tiny FP32 router GEMM for NVFP4 checkpoints` | 2-line gate change: allow `quant_scheme == "nvfp4"` alongside bf16 for the tiny FP32 router at hidden 5120 / 256 experts / compute capability 9–10. | **drop** for our Qwen3.8 stack (Beam-only; `beam.py` did not exist at `8edd1fa`); else needs port |
| 10 | `0f3159a683` | `[Feat] [NVIDIA] cuda-graph: Prefill CUDA graphs for FlashInfer CuTe DSL MoE and MegaMoE` | Captures EXTEND dummies for target-prefill/spec modes so CuTe DSL MoE/MegaMoE runs graphed instead of re-profiling. | **needs port** (11 of 11 files) |

Not shortlisted but worth a second pass if the corresponding path is in use: `a945a7216f` (pre-allocate trtllm-gen MXFP4 MoE workspace), `24feb92de0` (DeepGEMM weight localization SM10.0/10.3 — not SM120), `4929ba4024` (swap-AB MoE on Blackwell), `b2ce7f8811` (DSA MTP prefill KV without attention), `44ffcdead5` (RTX PRO 5000 channelwise FP8 GEMM tiles), `6a2f9b974a` (int64 GDN chunk-state offsets), `6eaca42f57` / `2b9eab7125` (SM12x dispatch guards).

## Explicit hash checks

| Hash | Verdict | Note |
|---|---|---|
| `775db052b6` (2026-11-15) | present in `upstream/main`, but **reverted** 4 commits later | 66 files; hybrid GDN/QSA/MoE optimization bundle. Do not cherry-pick from main as-is. |
| `3981de0825` (2026-11-15) | present in `upstream/main` | Exact revert of `775db052b6` (same 66-file stat, signs inverted). Commit message carries no reason. |
| `a8ac86595c` (2026-11-20) | present in `upstream/main` | 1 file / 2 lines; Beam model only, not Qwen3.8 Flash-Next. |
| `0f1509829b` (2026-11-17) | present in `upstream/main` | Fix for silent wrong outputs on misaligned NVFP4 TRT-LLM MoE hidden dims — highest value on this list. |
| `ff68baae6d` (2026-11-15) | present in `upstream/main` | 14 files; in-place unified-memory GDN prefill; overlaps `03d4f49f22` conceptually. |
| `03d4f49f22` (2026-11-15) | present in `upstream/main` | 10 files; GDN prefill host-stall removal + prewarm; pairs with `ff68baae6d`. |
| `5aee953e1a` (2026-11-19) | present in `upstream/main` | 2 files; SM120 TileLang BF16 shared-memory overflow fix (sparse/DSA path). |

**Why the revert matters.** `upstream/main` @ `563813c0d6` contains `775db052b6` *and* its revert `3981de0825`, so the net upstream state has none of those GDN+QSA+MoE optimizations. A cherry-pick of the original hash would reintroduce code maintainers deliberately backed out within hours, with an unrecorded reason (CI breakage or regression) and on top of a 66-file/53-file-drift delta against our donor base — high conflict and high regression risk. A reapply exists locally as `108c00266f` on branch `fp4local-port` (2026-11-17, upstream PR #50703 reapplied), but it is **not** in `upstream/main`; a path-level diff against the original shows no functional delta in the optimization files (only unrelated `engine.py`, `io_struct.py`, `scheduler.py`, `forward_batch_info.py` and testdata differences), so the reapply does not itself explain or fix the revert. Companion `0b28a39a2b` (megakernel-binding views, in main) may also be needed. Recommendation: treat the whole bundle as blocked pending human review of why #50703 was reverted; use the granular GDN commits (`03d4f49f22`, `ff68baae6d`, `077acd49a1`, `304a6016b3`) instead, and never assume the reverted patch is safe because its hash appears in main's history.

**Draft PRs #51051–#51068.** No commit with those PR numbers appears in `upstream/main`'s log through 2026-11-23; they remain open drafts with no merge hash to cherry-pick. Treat as candidates only once merged, and re-scan then.
