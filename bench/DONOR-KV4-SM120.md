# Donor FIFO: NVFP4 KV and SM120 SwiGLU

## Test order

1. Restore the donor draft vocabulary: 49,152 static NVFP4 rows plus 256 FP8 FIFO rows, with full LRU off. Keep the existing eight-request configuration on `127.0.0.1:8001`, with decode BS1–8, 48 Mamba slots, pinned host PLE, and no HiCache/NIXL. Retain the completed LRU artifacts separately; a matched real coding replay is needed to evaluate that policy.
2. With the existing MoE path, test `nvfp4` KV instead of `fp8_e4m3`. Keep the donor engine, model, adaptive config, effective memory fraction, and launch flags. Record actual pool size, startup smoke, cold prefill, TTFT and c1–c8 aggregate throughput. Verify the effective KV dtype and that FIFO is still present. Restore the FP8 FIFO endpoint if the arm fails. Do not compare this as an isolated KV effect against a four-slot donor or an LRU arm.
3. Keep the KV dtype and map fixed and compare general CUTLASS against the ported SM120 NVFP4 SwiGLU path. Use actual MoE input rows of 256, 512, 513 and 1,024; check short decode and 4k/8k prefill separately. Distinguish CUDA Graph padding from logical prompt length. Capture actual kernel/fallback dispatch and check output agreement before timed, uninstrumented runs. Leave prefill chunks at 8,192.

## Integration boundary

The known limit in the SM120 specialized path is **512 MoE input rows per invocation**, not 120 or 512 nominal prompt tokens. At 512 the remaining eligibility checks still apply; at 513 it must fall back. The tested rc5-p1 implementation is upstream commit `d5a4bc5a35` on its line. It is not present on the HyperBurst donor and its patch does not apply directly: the runner, argument registration, and JIT layouts differ. Port only the needed kernel and dispatch. Preserve the donor's verification-tail masking, tuned/OEA path and ordinary fallback where the fused path cannot preserve their semantics. Do not silently substitute the rc5 model or service.

The upstream SwiGLU route requires `SGLANG_FLASHINFER_MOE_FUSED_FINALIZE=0`; the donor default is `1`. For the isolated kernel comparison, both the specialized and general CUTLASS arms must use `0`. If tested, the donor default is a separate reference, not the matched kernel control.

## First checks

- FIFO restore `donor-fifo8-20261124T164833Z` passed `HB-OK` and `/health` on port 8001. The log confirmed the fixed/FIFO row split, eight requests, 48 Mamba slots, and 509,760 KV token slots.
- Standalone SM120 port: the unmodified upstream CUDA source and bitwise/replay test fixture were imported into an isolated donor checkout. A CPU check with a mocked launch confirmed dispatch eligibility for valid shapes with 120, 256, and 512 rows and rejection at 513, 1,024, 4,096 and 8,192. This does not prove a CUDA build, kernel execution, or server dispatch.

## NVFP4 startup result

`donor-kv4-20261124T165057Z` failed before model load and before any benchmark: the KV4 MHA validator rejected the donor's `flashinfer` attention backend. The fallback restored FIFO/FP8 on port 8001 and passed the startup smoke. The donor also lacks the compressed-QSA NVFP4 cache path; changing the backend solely to pass validation is not sufficient. The rc5-p1 implementation `7ca3bcb15e` adds packed KV gathers and QSA scheduler/backend integration, and does not apply directly on this donor. No NVFP4 performance number was collected for this arm.
