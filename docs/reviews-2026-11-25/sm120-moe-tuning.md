# SM120 NVFP4 MoE SwiGLU — read-only tuning analysis

Scope: donor `/tmp/opencode/sglang-donor-nvfp4kv` @ `dc333b6` (`donor-nvfp4-kv`), no edits, no GPU launches, no contact with port 8001. Evidence: source + dispatch + existing trace logs + upstream git refs.

## 1. Upstream cherry-pick verdict

- `/srv/community-repos/sglang-upstream`, `upstream/main` = `563813c0d6` **does not contain** `sm120_nvfp4_moe_swiglu.cuh` and its `sm120_nvfp4_moe.py` has only the MTP-MMA/weight-transpose helpers (`try_nvfp4_moe`, M<=4, H=1024, R=22), not the SwiGLU path.
- The newest ref touching the SwiGLU `.cuh` anywhere is `d5a4bc5a35` (2026-11-17, `rc/qwen38-rc5*` branches). Its `.cuh` is **byte-identical** to the donor copy (md5 `0a7ea19c…`). The donor `.py` is a trimmed copy of d5's `.py` (MTP part removed; uint8-view shape checks + trace added).
- **Verdict: nothing newer to cherry-pick.** Tuning must be local. The only related upstream kernel absent from the donor is `sm120_nvfp4_moe_mtp_mma.cuh` (draft-model small-M path, last touched 2026-11-11) — a separate lever for draft, not a replacement for this kernel.

## 2. What actually reaches the kernel

Dispatch gate (`flashinfer_cutlass.py:307-347`): `quant_type == fp4`, gated SiLU, TP=EP=1, no expert tier, no `x_sf`, no all-to-all, no OEA, no tactic policy, unfused finalize, no 4-over-6, fast-math enabled, SM120, plus `try_nvfp4_moe_swiglu` shape checks. All are satisfied by the current server env (`SGLANG_DONOR_SM120_SWIGLU=1`, `SGLANG_FLASHINFER_MOE_FUSED_FINALIZE=0`, etc.).

- Target MoE shape: **E=512, TopK=10, H=2560, I=640** (model `text_config`: 512 experts, 10 experts/token, hidden 2560, moe inter 640). `TopK` matters: it is a compile-time template arg.
- **Decode/verify** (target model): `M = BS × (steps+1)` with adaptive steps ∈ {3,7,15} and BS=1..8, plus `M = BS` single-token target steps. Existing trace (`hb-20261125-024251-nvidia.log`) shows captured fused calls at rows `1..8`, `12,16,20,24,28,32` (steps 3) and `40,48,56,64` (steps 7); M=16 is the common C4 verify shape.
- **Draft MoE** does *not* use this kernel: `SGLANG_OPT_DRAFT_MOE_GEMV=1` sends M=1 draft calls to the separate Triton GEMV; M>1 draft calls fail the `w2.shape == (E, H, w13.shape[1]//4)` gate (draft shape is H=1024, R=22) and go to CUTLASS.
- **Prefill** (>512 rows, chunked-prefill-size 8192) falls back: `SWIGLU_MAX_TOKENS = 512` is a hardcoded Python cutoff. Trace shows `cutlass:shape rows=612`; small extends <=512 still use the fused kernel (rows up to ~233 observed). During FlashInfer autotune the fused path is deliberately skipped (`cutlass:tuning`).

## 3. Knob inventory (E=512, TopK=10, H=2560, I=640)

Derived: `K1B=1280B`, `K2B=320B`, `NT1=I/64=10`, `NT2=H/128=20`, `NS2=5`, `NST1=10`, `NST2=5`, `kSlot=23040B`, `kMainSmem=46080B` (45 KiB), `tiles/unit=30`.

| Knob | Current value | Hardcoded / runtime | Effect at decode |
|---|---|---|---|
| `kUnit` tokens/unit | 32 | **hardcoded constant**, also duplicated as literal `32` in Python workspace sizing | Largest lever. With TopK=10 and M=16, only ~135–160 distinct experts get 1–2 tokens; `mcount≈1..2` of 32 → only 1–2 of 4 token groups do MMA (~29% utilization). Also loads/pads full 32-token A tiles. |
| `kRows` / `kSeg` weight tile | 128 / 64 (=64 up + 64 gate) | **hardcoded**; `kRows` also duplicated as `128` in Python `fcnt` sizing | Sets NT1=10, NT2=20; halving doubles tiles/unit (30→60) and atomics. |
| `kWarpRows` / `kThreads` | 32 / 128 (4 warps) | **hardcoded**, `__launch_bounds__` | Must satisfy `kRows = (kThreads/32) × kWarpRows`; acc regs/thread = `(kWarpRows/16)×(kUnit/8)×4` (currently 32 floats/thread). |
| `kStages` GEMM1 | 2 × 128 B | **hardcoded** | 10 K-chunks; 2-stage cp.async pipeline. |
| GEMM2 stages | `2*kStages` = 4 × 64 B | **hardcoded** (derived) | Only 5 K-chunks exist; 4 stages force an in-loop wait/barrier per chunk. |
| `kStage1` / `kStage2` | 128 B / 64 B | **hardcoded** | K-chunk = 256 / 128 FP4 values; tied to SF swizzle `CPS=KS/32`. |
| Persistent grid | `SM_count × blocks_per_sm`, `queue` atomicAdd | **runtime-derived** from occupancy + SM query | Hundreds of resident CTAs; at M=16 ~5k tiles → ~5k atomic queue pulls. |
| Split-K | none (full K, ascending k64; `g1done`/`fcnt` counters) | absent, by design | Do **not** add: bitwise parity with CUTLASS requires ascending-K accumulation and k-ordered finalize. |
| Cluster / TMA / multicast | none (`cp.async` + `mma.sync`) | absent | Adding clusters would require a TMA rewrite; out of scope for a small tuning experiment. |
| Dispatch cutoff / gating | `SWIGLU_MAX_TOKENS=512`, env gates | Python constant + env | 512 is fixed; no runtime tile/stage config knob exists. |

Worked upper-bound tile counts (`units = min(10M,512)+ceil(10M/32)`, `tiles = 30*units`; actual routing gives fewer): M=1 ≤330, M=8 ≤2490, **M=16 ≤4950**, M=32 ≤9900, M=64 ≤15960, M=128 ≤16560, M=512 ≤20160. At M=16 the persistent grid (~150 SMs × 4–5 CTAs) is saturated, so per-tile waste and queue/barrier overhead dominate — not raw parallelism.

## 4. Proposed experiment matrix (6 configs, one shape E512/TopK10/H2560/I640)

| ID | Change vs baseline | Mechanism / expected effect | Main risk |
|---|---|---|---|
| C0 | baseline | reference; ~22% GPU time at c1, ~37% at c4 (given) | — |
| C1 | `kUnit=16` (+ Python sizing) | Halves token padding; MMA utilization ~29%→~58% at M=16; ~2× units/tiles. Expect gain at M=8–32, neutral/worse at M>=64. | Python `_swiglu_workspace_sizes` hardcodes `32` twice; must change with C++ `Shape` or OOB. |
| C2 | `kUnit=8` (+ Python sizing) | DB=1 → ~100% MMA utilization at M<=8; maximal reduction of wasted A loads; more units/atomics. Expect best at M=1–16, watch M=64+ tail. | Same sizing risk; `kUnit>=8` required (`DB=kUnit/8`). |
| C3 | GEMM2 stages `4 → 5` (`NSTG2=5`, same `kSlot/2` buffers) | All 5 K-chunks of GEMM2 prefetched before consumption; removes in-loop global-load dependency and one barrier/chunk in the second GEMM. Low-risk, cheap. | 5 stages use exactly `kMainSmem` (offset 46080 = 46080); any 6th stage overflows. |
| C4 | `kThreads=256`, `kWarpRows=16` (8 warps) | Halves acc registers/thread (32→16 floats); doubles warps per CTA → more latency hiding for tiny tiles. | Warp-row partition, epilogue loops and `__launch_bounds__` must be updated consistently; occupancy may drop with 256 threads. |
| C5 | GEMM1 `kStage1=64,kStages=4`; GEMM2 `kStage2=32,NSTG=8` (same 45 KiB smem) | Deeper prefetch (20 GEMM1 / 10 GEMM2 K-chunks) without occupancy loss; hides weight-load latency. | Changes `CPS` (2→1) in the SF swizzle addressing — highest bitwise risk; extra barriers. Fallback if red: test `kStages=3` at 128 B instead (more smem, fewer CTAs). |

Persistent-grid A/B (cheap substitute for C5 if queue contention is suspected): cap grid at `SM_count × 1..2` instead of occupancy × 4–5; mechanism = fewer atomic queue pulls vs longer per-CTA tail. Static tile partition would remove the atomics entirely but is a larger change.

## 5. Correctness risks / validation

- Gold test is **bitwise** equality vs `cutlass_fused_moe(use_fused_finalize=False)` (`test/registered/kernels/ops/gemm/test_sm120_nvfp4_moe_swiglu.py`), including skipped experts and CUDA-graph replay. Any change to K order, SF layout, quantization rounding or SiLu fast-math breaks it.
- `kUnit`/`kRows` changes must be mirrored in Python `_swiglu_workspace_sizes` (literals 32 and `hidden//128`) and C++ `Shape::u8_bytes/i32_count`; mismatch = OOB writes, especially under graph capture.
- `kThreads=256` requires `kWarpRows=kRows/(kThreads/32)` everywhere (MMA A-fragment mapping `warp*kWarpRows+...`, epilogue row = `warp*kWarpRows+a*16+g+8*(i>>1)`), else rows are omitted/duplicated.
- GEMM2 5 stages sits exactly at the smem limit; validate `static_assert(stage_bytes<kStage2>() <= kSlot/2)` and `kUnit*kOutLd*2 <= kMainSmem` after each edit.
- The JIT cache is content-addressed over source contents, so `.cuh` edits rebuild automatically; but `_swiglu_module` is `@cache_once` — the server process needs a restart to load a new variant. Do not restart the port-8001 benchmark; use a separate port/worktree.
- Routing hooks (`before_launch` singleton prune, VT tail mask) and `TopK` as template arg stay unchanged; re-run the skipped-expert and empty-expert tests for every config.
- Measure with the same method for all configs (standalone bench rows {1,8,16,32,64,128,512}; then e2e cold-prefill/BB after the current benchmark finishes). Never mix warm-prefix and cold-prefill numbers.

## 6. Ten-line summary

1. No newer upstream kernel: `upstream/main` dropped the SwiGLU kernel; newest ref (d5a4bc5a35) has a byte-identical `.cuh` to the donor, so tuning is local work.
2. Decode route: target-model MoE only, E=512/TopK=10/H=2560/I=640, M=BS×(steps+1) with steps 3/7/15 plus M=BS steps; draft uses a separate GEMV, prefill >512 falls back.
3. Observed fused rows: 1–8 (single-token), 12–32 (steps 3), 40–64 (steps 7); M=16 is the key C4 verify shape.
4. Kernel is a 2-launch JIT: routing/quant prologue + persistent queue-driven CTAs with two GEMMs and counter-based unit/token dependencies.
5. All tile/stage/warp constants are compile-time hardcoded; only the grid is runtime-derived (SM count × occupancy), and the 512-row cutoff is a Python constant.
6. No split-K and no clusters/TMA — split-K would break the required bitwise ascending-K parity.
7. Biggest waste: `kUnit=32` vs `mcount≈1–2` at M=16 (≈29% MMA utilization) → test `kUnit=16` then `kUnit=8`.
8. Second lever: GEMM2 has only 5 K-chunks but uses 4 stages → prefetch all 5 at identical shared-memory cost.
9. Third/fourth levers: 8 warps with half accumulators; deeper 4×64B/32B pipeline (highest SF-swizzle risk), plus a cheap grid-cap A/B.
10. Every variant must pass the bitwise/CUDA-graph unit test and be measured with one consistent method; mirror any constant change in the Python workspace sizing and restart the process (never the live 8001 benchmark).
