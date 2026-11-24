# Donor FIFO: NVFP4 KV and SM120 SwiGLU

## Test order

1. Restore the donor draft vocabulary: 49,152 static NVFP4 rows plus 256 FP8 FIFO rows, with full LRU off. Keep the existing eight-request configuration on `127.0.0.1:8001`, with decode BS1–8, 48 Mamba slots, pinned host PLE, and no HiCache/NIXL. Retain the completed LRU artifacts separately; a matched real coding replay is needed to evaluate that policy.
2. With the existing MoE path, test `nvfp4` KV instead of `fp8_e4m3`. Keep the donor engine, model, adaptive config, effective memory fraction, and launch flags. Record actual pool size, startup smoke, cold prefill, TTFT and c1–c8 aggregate throughput. Verify the effective KV dtype and that FIFO is still present. Restore the FP8 FIFO endpoint if the arm fails. Do not compare this as an isolated KV effect against a four-slot donor or an LRU arm.
3. Keep the KV dtype and map fixed and compare general CUTLASS against the ported SM120 NVFP4 SwiGLU path. Use actual MoE input rows of 256, 512, 513 and 1,024; check short decode and 4k/8k prefill separately. Distinguish CUDA Graph padding from logical prompt length. Capture actual kernel/fallback dispatch and check output agreement before timed, uninstrumented runs. Leave prefill chunks at 8,192.

## Integration boundary

The known limit in the SM120 specialized path is **512 MoE input rows per invocation**, not 120 or 512 nominal prompt tokens. At 512 the remaining eligibility checks still apply; at 513 it must fall back. The tested rc5-p1 implementation is upstream commit `d5a4bc5a35` on its line. It is not present on the HyperBurst donor and its patch does not apply directly: the runner, argument registration, and JIT layouts differ. Port only the needed kernel and dispatch. Preserve the donor's verification-tail masking, tuned/OEA path and ordinary fallback where the fused path cannot preserve their semantics. Do not silently substitute the rc5 model or service.
