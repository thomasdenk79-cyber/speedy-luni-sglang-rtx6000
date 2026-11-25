# Speedy follow-up: SM120 dispatch boundary

Run after the current c1–c8 campaign. Do not mix these results with the donor baseline or change the current campaign. Keep the eight-request, 48-Mamba-slot, decode BS1–8 and [49,152-row LRU profile](TOKENMAP-LRU-48K.md) fixed. Keep PLE on pinned host RAM, not on SSD, in every arm. Use the [canonical benchmark driver](CANONICAL-DRIVER.md) for comparable runs.

## Test matrix

- Keep HiCache off and retain the qualified memory headroom. Compare `flashinfer_cutlass_sm120_swiglu` against `flashinfer_cutlass` with the same model, NVFP4 KV, and token shapes.
- Probe **256, 512, 513, 1,024** tokens plus **4,096 and 8,192** as long-prefill references. Count actual rows passed to MoE, not requested prompt length; avoid cached prefixes and record chunking or padding. Check actual dispatch on small decode batches as well.
- Use a separate diagnostic pass to record selected kernels and peak GPU memory (including free VRAM before prefill). At 512 the specialization is eligible, not guaranteed; at 513 it must fall back on this build. Compare warmed, uninstrumented prefill throughput, TTFT and peak memory at matched shapes, with output checks. Do not change the default chunk size to 512 based on dispatch alone.

HiCache and NIXL remain off; their evaluation is deferred.
