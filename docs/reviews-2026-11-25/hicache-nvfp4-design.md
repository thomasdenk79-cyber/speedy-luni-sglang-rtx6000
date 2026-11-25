# HiCache `file` backend + NVFP4 KV: donor-native port design (READ-ONLY ANALYSIS)

Status: design doc only. No repo/worktree edits, no GPU use, no benchmarking, and the
service on port 8001 was not touched. Everything below is derived by reading
`/tmp/opencode/sglang-donor-nvfp4kv` on 2026-11-25. Where a number depends on the
running model/launcher, the formula is given and the value is marked **unknown**.

---

## Contents
1. [Scope and evidence](#1-scope-and-evidence)
2. [What the donor actually persists today](#2-what-the-donor-actually-persists-today)
3. [NVFP4 KV representation: bytes that must survive a page write](#3-nvfp4-kv-representation-bytes-that-must-survive-a-page-write)
4. [Per-page inventory and packed row/page size](#4-per-page-inventory-and-packed-rowpage-size)
5. [Sidecar file layout proposal](#5-sidecar-file-layout-proposal)
6. [Write-complete design](#6-write-complete-design)
7. [Pinned host tier + OS/ZFS page-cache experiment](#7-pinned-host-tier--oszfs-page-cache-experiment)
8. [Smallest first milestone](#8-smallest-first-milestone)
9. [Explicit unknowns](#9-explicit-unknowns)

---

## 1. Scope and evidence

Files actually read (all under `/tmp/opencode/sglang-donor-nvfp4kv/`):

| Area | File | Key lines / facts |
|---|---|---|
| HiCache storage API + file backend | `python/sglang/srt/mem_cache/hicache_storage.py` | `HiCacheStorage` (153-329), `HiCacheFile` (364-728); `set()` uses tmp + `os.replace`, **no `fsync`** (503-550); `get()` reads into a host tensor (466-488); v2 component keys (441-451, 575-653); v2 page IO (658-713); `PoolName` incl. `DRAFT`, `MAMBA`, `QSA_COMPRESSED` (58-79); hit policies (84-92) |
| File evictor | `python/sglang/srt/mem_cache/storage/file/lru_file_evictor.py` | inert when `max_size` and `min_free` are unset (21, 93-102); config from `extra_config` or `SGLANG_HICACHE_FILE_BACKEND_*` |
| Backend factory | `python/sglang/srt/mem_cache/storage/backend_factory.py` | `"file"` → `HiCacheFile` (198) |
| Storage controller | `python/sglang/srt/managers/cache_controller.py` | `file` uses `_generic_page_set`/`_generic_page_get` (591-602, 1235-1240, 1248-1268); batches of `STORAGE_BATCH_SIZE=128`; `ack_backup_queue` drains on write completion (1270-1282) |
| Hybrid controller | `python/sglang/srt/mem_cache/hybrid_cache/hybrid_cache_controller.py` | sidecar `batch_set_v2` / `batch_get_v2` (560-737), trailing-key fold (788-828), per-rank backup thread (739-752) |
| Write-through ack | `python/sglang/srt/mem_cache/hiradix_cache.py` | `_finish_write_through_ack` marks DMA-to-host done and calls `write_backup_storage` (927-935) |
| Host pools | `python/sglang/srt/mem_cache/pool_host/base.py`, `mha.py`, `mamba.py` | host dtype = `device_pool.store_dtype` (base 167); MHA flat page = `[2, layer, page, H, D]` bytes (mha 504-524); MTP draft layers packed as extra KV layers (mha 71, 87, 129-149); Mamba page = temporal + conv + PLE siblings (mamba 300-332, 371-379, 636-665) |
| Device KV pool | `python/sglang/srt/mem_cache/memory_pool.py` | quantized buffers: `k/v_buffer` `[m,H,D/2]`, `k/v_scale_buffer` `[m,H,D/16]` uint8 (2130-2152); FP4 slot moves update both (2188-2198) |
| Quant method | `python/sglang/srt/layers/quantization/fp4_kv_cache_quant_method.py`, `kvfp4_tensor.py` | NVFP4 two-level scales, block 16, global FP32 per layer (333-556); buffers are `uint8`; scales are E4M3 bit patterns; QSA subclass has no dequant workspace (559-612) |
| QSA NVFP4 reader | `python/sglang/srt/layers/attention/qsa/nvfp4_kv.py` | packed `[slots,heads,dim/2]`, scales `[slots,heads,dim/16]` (1-15, 121-134); `_dequant_rows` (37-60) |
| QSA guard | `python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py` | **HiCache + QSA NVFP4 is hard-failed** (108-121) with reason "host mirror sizes rows from logical head_dim and has no copy of token-linear block scales" |
| Pinned host | `python/sglang/srt/mem_cache/pool_host/common.py`, `base.py` | `SGLANG_HICACHE_PREPIN_BYTES` arena, `cudaHostRegister` (common 32-78, 187-197, 231-295); 10 GiB reserve (base 29-31) |
| Buffer-mode pieces | `python/sglang/srt/mem_cache/buffer_mode/storage_existence_cache.py`, `buffer_page_cache.py` | existence cache wired; `BufferPageCache` docstring says **NOT WIRED YET** |
| PLE state | `python/sglang/srt/mem_cache/ple_state_pool.py` | siblings `ple_short_conv` (135-137), `ple_ngram` (234-236) |
| Launcher | `configs/pennyroyal/serve-flash-next.sh`, `RUN.md`, `MEMORY-AND-PERSISTENCE.md` | current page size 64, KV dtype `fp8_e4m3`, 24 Mamba slots, NIXL backend; 32 GB host tier; namespace identity concept (MEMORY 95-117) |

### Hard statements established by reading code
- `HiCacheFile` is a **raw byte store**: one `.bin` file per page per pool component, keyed by
  chained SHA-256 page hashes plus a rank/model suffix. It does **not** know tensor layout,
  dtypes, scales, or global scales.
- There is **no write-complete** anywhere in the file backend, nor any `fsync`/`fdatasync`
  in it (global grep finds fsync only in the loader and the 3FS client). `os.replace`
  gives atomic visibility, not durability.
- `file` uses the generic path: `host_pool.get_data_page(slot, flat=True)` → `batch_set`
  → `HiCacheFile.set`. For v2 sidecars it uses `batch_set_v2` with the same flat-page
  convention.
- Existing "write-through ack" in `hiradix_cache.py` is the **D2H DMA ack**; storage
  writes are separately enqueued to `backup_queue` and acked after `batch_set` returns.
- Donor deliberately refuses `HiCache + QSA NVFP4` (assembler lines 114-121).
- Donor's host KV mirror is sized from **logical** `head_dim` and has **no scale buffers**
  (`mha.py` 153-157, 162-198, 504-524; `base.py` 167). This is the root of the
  "dropped or truncated packed NVFP4 pages" note in the assembler comment.

---

## 2. What the donor currently persists today

### 2.1 Key and filename construction (`hicache_storage.py`)
```
config_suffix = f"_{model_name}"
             + f"_{tp_rank}_{tp_size}"                 # unless MLA
             + f"_{pp_size}_{pp_rank}"                 # if pp_size > 1
             + f"_cp{attn_cp_rank}_{attn_cp_size}"     # if attn_cp_size > 1

KV page key:      <page_hash><config_suffix>.bin
component page:   <page_hash>.<component><config_suffix>.bin
```
`PoolName` components observed in code: `kv`, `mamba`, `swa`, `indexer`,
`qsa_compressed`, `draft`, `draft_indexer`, `draft_swa`, plus DeepSeek-V4 variants.
`_log_key()` (655-656) and `_get_component_key()` (441-444) produce exactly these names.
Page hashes are chained SHA-256 values from `mem_cache/utils.py::get_hash_str`, so a
prefix is a contiguous run of hashes.

### 2.2 Page payload = host flat page, verbatim
- `HiCacheFile._write_page` (667-673): `host_pool.get_data_page(page_offset, flat=True)`
  then `set(key, value)`; `value.contiguous().view(torch.uint8).numpy().tofile(tmp)`.
- `HiCacheFile._read_page` (658-665): allocate `host_pool.get_dummy_flat_data_page()`,
  `f.readinto(memoryview(...))`, then `set_from_flat_data_page(offset, page)`.
- KV `page_first` host layout (`mha.py` 504-524) flattens
  `(2, page_size, layer_num, head_num, head_dim)`; `layer_num` includes packed MTP draft
  layers when `mtp_draft_device_pools` is set (mha 129-149, base 202-214).
- Mamba (page size 1) flattens `[temporal | conv_0 .. conv_n | ple_short_conv | ple_ngram]`
  byte slices (`mamba.py` 300-332, 636-665). Keys use `TRAILING_PAGES` with the **last**
  KV page hash (`mamba_component.py` 752-763).
- LRU in `lru_file_evictor.py` is inert unless `max_size`/`min_free` are configured →
  satisfies "LRU stays off" if neither is set.

### 2.3 Persistence/ack gap
`set()` writes `<path>.tmp.<pid>.<tid>.<uuid>`, `os.replace()`, updates metadata cache,
returns `True`; there is no `fsync` of the file or directory, no background writer, and
no durability callback. `batch_set_v2` returns per-page `bool` = "bytes accepted".
The controller increments `operation.completed_tokens` and the tree later marks the
node `backuped`. So an acknowledged page can still be lost on crash/power loss.

---

## 3. NVFP4 KV representation: bytes that must survive a page write

NVFP4 (recipe `nvfp4`, block size 16) stores a value as
`e2m1_code * block_scale(E4M3) * global_scale(FP32, per layer)` — see
`kvfp4_tensor.py` 150-229 and `qsa/nvfp4_kv.py` 1-15.

Per token, per **full-attention layer**, with `H_k = kv heads`, `D_k = key dim`,
`H_v = value heads`, `D_v = value dim`:

| Component | Shape per token/layer | Bytes | dtype on disk |
|---|---|---:|---|
| K packed E2M1 (2 per byte, even code in low nibble) | `H_k * D_k/2` | `H_k*D_k/2` | raw uint8 |
| V packed E2M1 | `H_v * D_v/2` | `H_v*D_v/2` | raw uint8 |
| K block scales | `H_k * ceil(D_k/16)` | `H_k*D_k/16` | raw E4M3 bit pattern (uint8 view) |
| V block scales | `H_v * ceil(D_v/16)` | `H_v*D_v/16` | raw E4M3 bit pattern (uint8 view) |
| K/V global scales | 1 scalar each per layer | 4 + 4 | **not per page**; model-derived |

For `H_k=H_v=H`, `D_k=D_v=D`, `16 | D`:
```
bytes/token/layer = H*D/2 + H*D/2 + H*D/16 + H*D/16 = (9/8) * H*D
```
Compare FP8 E4M3: `2*H*D` bytes/token/layer → NVFP4 = **9/16 = 56.25 %** of the FP8
write volume (excluding workspace, which is never persisted).

Two corrections are required relative to the donor host mirror:
1. **Packed row size**: device row is `H*D/2` bytes (`memory_pool.py` 2138-2151),
   but `MHATokenToKVPoolHost.get_size_per_token()` uses `H*D` bytes per layer
   (`mha.py` 153-157, 187) and `transfer_kv_*` uses `item_size = token_stride_size =
   H*D*itemsize` (`mha.py` 187, 260-502). A donor-native NVFP4 host pool must use the
   packed row and carry the scale buffers, otherwise reads/writes truncate or smear rows.
2. **Scale sidecars**: `k_scale_buffer`/`v_scale_buffer` exist on the device
   (`memory_pool.py` 2147-2148) but are never transferred to/from host by `mha.py`
   (260-502, 387-502) and never written by `HiCacheFile`. Without them a restored page
   is uninterpretable (denominator/gather scale is missing).

**Global scales** are loaded from the checkpoint by
`NVFP4KVCacheMethod.load_scales_from_model()` (`fp4_kv_cache_quant_method.py` 357-428;
SM100 multiplies by `E2M1_MAX=6` at 419-421). They are small, model-owned, and must be
reconstructed identically after restart. Store them in the representation manifest, not
in each page. This matches the donor's NIXL namespace rule
(`MEMORY-AND-PERSISTENCE.md` 95-117): cache content is disposable; incompatible
representations get a different root and become a clean miss.

**MTP draft sidecar** has two donor paths, both relevant:
- *Packed native MTP*: `mtp_draft_device_pools` are appended to the host KV layer axis
  (`mha.py` 129-149; `hybrid_pool_assembler.py` 194-235, 270-304). With NVFP4 draft
  pools, their packed bytes **and** their per-block scales are needed, i.e. target and
  draft layers must share the new scale-aware host layout.
- *Separate `PoolName.DRAFT`* sidecar (`hybrid_pool_assembler.py` 1092-1165): its own host
  pool, own `<hash>.draft_<suffix>.bin`, indices from KV (`SidecarPoolSpec`,
  `ALL_PAGES`). If its device pool is NVFP4, add `<hash>.draftscale_*` sidecars.
Which of the two applies is launcher/model-dependent → **unknown until the actual
draft runner is read/confirmed** (see §9).

**Mamba/PLE state** is already persisted by the donor, with no NVFP4-specific change:
per slot `get_size_per_token()` = `num_mamba_layers * (Σ conv_elem*itemsize +
temporal_elem*itemsize) + Σ_siblings(num_layers_sib * prod(item_shape) * itemsize)`
(`mamba.py` 371-379). Siblings are `ple_short_conv` (tensor
`(layers, size+1, state...)`, slot axis 1) and `ple_ngram` (`(size+1, context_len)`
int64, slot axis 0), from `ple_state_pool.py` 135-137, 234-236. With 48 slots, each
host Mamba page is one slot; keys are the trailing KV page hash.

---

## 4. Per-page inventory and packed row/page size

Define:
```
P   = KV page size in tokens            (launcher currently 64)
Lt  = target full-attention layers
Ld  = packed native-MTP draft layers (0 if separate DRAFT sidecar)
L   = Lt + Ld
H   = KV heads per TP rank, D = head dim (16 | D assumed by recipe)
Hs  = draft head dim (if different), Ls = separate draft layers
B   = 16 (NVFP4 block size)
```

Per KV page (one `<hash>.bin`), with the proposed packed host page order:
```
data bytes/page = P * Σ_layer ( H_k*D_k/2 + H_v*D_v/2 )
scale bytes/page = P * Σ_layer ( H_k*D_k/16 + H_v*D_v/16 )
total NVFP4 = P * L * (9/8) * H * D                (symmetric case)
FP8 baseline = P * L * 2 * H * D
```
If the donor's *current* (wrong) host mirror were used, the `.bin` would instead be
`P * L * 2 * H * D` bytes of `uint8` — i.e. the FP8-sized blob that cannot be unpacked
into NVFP4 rows; this is the defect to fix, not a format to preserve.

Per separate draft page, if applicable: same formula with `Ls, Hs` and draft keys.

Per Mamba page (1 slot), from `MambaPoolHost`:
```
bytes/slot = num_mamba_layers * (temporal_numel*temporal_itemsize
                                 + Σ_conv conv_numel_i*conv_itemsize)
           + ple_short_conv_layers * short_conv_state_numel * dtype_size
           + ple_ngram_context_len * 8
```
Absolute numbers need the device shapes (`MambaPool.mamba_cache.temporal/conv`,
`ShortConvPool.state_shape`, `NGramPool.context_len`) → **unknown until runtime config
is read**; the formula is exact and the code path for flattening is already present.

Write-volume planning identity: NVFP4 KV file bytes ≈ 0.5625 × FP8 KV file bytes;
add scale sidecars at 1/9 of the packed data volume. This is a useful sanity check on
the measured write volume (§7).

---

## 5. Sidecar file layout proposal

Donor-native pattern to preserve: one raw `.bin` per page per component, suffix by
rank/model, content-addressed names, opaque bytes, `batch_exists_v2` prefix fold.
Add a **representation root** and explicit scale components; keep the existing KV
`.bin` name stable for non-NVFP4 recipes.

```
$SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR/
  <representation-id>/
    representation.json                # immutable manifest (see below)
    <H>_<S>.bin                        # KV data page (existing name)
    <H>.kvscale_k_<S>.bin              # NEW: K E4M3 block scales
    <H>.kvscale_v_<S>.bin              # NEW: V E4M3 block scales
    <H>.draft_<S>.bin                  # only for separate DRAFT pool
    <H>.draftscale_k_<S>.bin           # NEW, only if draft KV is NVFP4
    <H>.draftscale_v_<S>.bin
    <H>.mamba_<S>.bin                  # existing: temporal+conv+PLE siblings
    <H>.qsa_compressed_<S>.bin         # existing
```
`<H>` = page hash, `<S>` = `config_suffix` from §2.1.

Byte order inside each file is the **host flat page order** in `page_first`:
`[K/V, layer, token, head, dim]` for data; `[layer, token, head, dim/16]` for scales.
`representation.json` must pin: layout (`page_first`), `P`, `Lt`, `Ld`, `H_k`,
`D_k`, `H_v`, `D_v`, `B=16`, recipe (`nvfp4`), PLE sibling names/order, Mamba shapes,
per-layer global K/V scales (or a checksum + checkpoint revision), SGLang HEAD, PyTorch
version, and GPU arch. On mismatch: select a new root / treat as miss (donor NIXL
precedent). Do **not** put global scales into per-page files.

Registration changes (design-level, no code here):
- Add `PoolName` values `KV_SCALE_K`, `KV_SCALE_V`, and optionally
  `DRAFT_SCALE_K`, `DRAFT_SCALE_V`; expose each via a small host-pool view whose
  `get_data_page()` returns the scale-buffer bytes for that page, and register them
  with `register_mem_host_pool_v2()` alongside `kv`.
- `HiCacheFile._get_component_key()` then naturally produces
  `<H>.kvscale_k_<S>.bin`; no change to the raw writer/readers.
- Hit logic: a KV page is usable only if data **and both** scale sidecars exist.
  Extend `batch_exists_v2` for `ALL_PAGES` co-presence and fold usable prefix to the
  minimum across these components; a missing scale = clean miss, never a garbage read.
- Write ordering: scales are small; write data first, then scales, then publish.
  Existence must be gated on all three (see write-complete below).

Alternative considered: pack data+scales into one `.bin`. Rejected as the first
milestone because it changes the existing KV filename payload for all recipes;
separate components match the donor's NIXL/Mamba component convention and allow a
scale-free FP8 path to remain untouched.

---

## 6. Write-complete design

Define **write-complete** = durable acknowledgement of a storage write batch: data bytes
have reached stable media (file `fsync` + directory `fsync`) and the controller may
publish the node as `backuped`. Donor currently acks only "rename succeeded".

Minimal donor-native implementation:
1. `HiCacheStorage` gains a `write_complete(ops, timeout)` / completion callback
   interface (default: no-op → current behavior for non-file backends).
2. `HiCacheFile.set()` keeps tmp + `os.replace` for atomic visibility, but enqueues
   `(op_id, file_path, dir_path)` to a single writer thread.
3. Writer thread group-commits: consume up to `STORAGE_BATCH_SIZE` (128) ops,
   `os.open(path, O_RDONLY)`, `os.fsync(fd)`, then one `os.fsync(dir_fd)` per directory;
   publish op ids into a completed set guarded by a `Condition`. Count and time each
   fsync for metrics.
4. `batch_set_v2` / `batch_set` return "issued" as today. Controller tracks issued
   op ids; a dedicated completion drain polls `write_complete` before putting
   `StorageOperation` onto `ack_backup_queue`. Only then does the tree mark `backuped`
   and release host locks. On timeout/error: do not mark backuped, remove positive
   metadata-cache entries for those keys so a later insert rewrites them.
5. `detach`/`shutdown`/`clear` must drain the writer queue and fsync the directory
   after any unlink; `StorageAttachment.detach()` already provides the lifecycle hook.
6. Keep existence semantics explicit: after `os.replace` the file is *visible* (so a
   same-process read can hit it) but *durable* only after write-complete. The
   completeness gate is in the controller, not in `exists()`.
7. TP/PP determinism: do **not** add a new collective in the completion path unless a
   cross-rank gate is required; per-rank durable acks keep the existing
   `ongoing_backup` / `_all_reduce(MIN)` structure for prefetch intact. If a collective
   is introduced later, it must be reached in identical order on all ranks.

This is deliberately a *file-backend* change. It does not require NIXL/GDS and is
independent of the FP4 format work, though both should land together for a clean A/B.

---

## 7. Pinned host tier + OS/ZFS page-cache experiment

Goal: decide whether the "small pinned host tier + OS/ZFS file cache as dynamic RAM
pool" is sufficient, i.e. whether reload from warmed page cache is fast and whether
write volume/temperature stay acceptable. **NIXL/GDS only if cold reload is the
bottleneck.**

### 7.1 Configuration under test
- Keep 8 requests, decode BS 1-8, `--max-mamba-cache-size 48`, pinned host PLE:
  `SGLANG_HICACHE_PREPIN_BYTES` sized to cover the KV + Mamba + PLE host mirrors
  (`common.py` 32-78) and `--hicache-mem-layout page_first`, `--hicache-io-backend kernel`.
- `file` backend via `--hicache-storage-backend=file` and
  `SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR` on the ZFS-backed mount.
- **LRU off**: do not set `SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE` /
  `..._MIN_FREE_SPACE` and do not put `max_size`/`min_free` in extra config
  (`lru_file_evictor.py` 93-102). Keep `BufferPageCache` unwired as in the donor.
- Hold the metadata-cache setting constant across arms (`..._ENABLE_METADATA_CACHE`
  and `..._METADATA_TTL`); `set()` skips rewrites of existing keys, which strongly
  affects write volume. Report it as a factor, not a hidden variable.

### 7.2 A/B reload measurement
Use identical serialized prefixes (content-addressed hashes) and a service restart
(or `/flush_cache` as in `test_hicache_storage_file_backend.py`) so GPU radix and host
state cannot answer the request; only the file backend can.

| Arm | Cache state | What it isolates |
|---|---|---|
| W (warm) | identical prefix already written; file pages resident in OS page cache / ZFS ARC | reload from RAM cache, near-zero SSD I/O |
| C (cold) | same files, but caches evicted before the reload | reload from SSD + decompression/format overhead |
| W' (write-complete on) | same as W with fsync/dir fsync enabled | durability cost on TTFT/TPOT and write latency |

Measure per arm: restored/cached tokens vs recomputed tokens (cache report / `meta_info`),
TTFT and decode TPOT for BS1-8, prefetch hit pages and prefetch wall time, bytes read
(`iostat -x`, `/proc/<pid>/io`), host-pool occupancy, and `write_complete` latency.
Use ≥3 repeats; record the prefix length and hash chain so arms are comparable.

**Important correction to the naive "drop_caches" plan:** `echo 3 > /proc/sys/vm/drop_caches`
drops the Linux page cache, **not** ZFS ARC. On a ZFS mount the W/C contrast will be
fake unless ARC is also bounded/evicted (e.g. temporarily lower `zfs_arc_max`, observe
`arcstat.py`/`arc_summary.py`, or use an ext4/tmpfs root for the controlled cold arm).
Do not drop caches or tune ARC while the benchmark on port 8001 is running — dropping
caches is host-wide; coordinate the cold arm for after that job or use an isolated
mount/namespace.

### 7.3 Write volume and SSD temperature
- Before/after each write phase: `du -sb` and file count under the representation root;
  separate KV data vs scale sidecars vs Mamba vs draft. Cross-check against the
  analytic identity in §4 (NVFP4 ≈ 0.5625 × FP8 bytes; scales = 1/9 of data).
- Per-process I/O: `pidstat -d`, `/proc/<pid>/io`, plus counters added at
  `HiCacheFile.set()` (bytes/page, pages/s) and at the fsync group commit (§6).
- SSD temperature/endurance: `smartctl -A -d nvme /dev/<nvme>` (or the appropriate
  `-d` for SATA/SAS) before, during, and after sustained writes; record
  `Temperature: Current`, `Percentage Used`, `Data Units Written`, plus
  `iostat -x` util/latency. Temperature lags write bursts; sample a time series, not
  just endpoints. ZFS compression and ARC mean on-disk bytes ≠ write amplification;
  report both logical page bytes and device-level units written.

Interpretation: if W reload is fast and C reload is rare, the page cache is already the
dynamic RAM pool and no NIXL/GDS is needed. If C dominates TTFT, escalate to NIXL/GDS
as a separate milestone, reusing the same representation manifest.

---

## 8. Smallest first milestone

**M1 (minimal, donor-native):** *KV-only NVFP4 round-trip through the `file` backend
with write-complete, no draft/Mamba format changes, no NIXL/GDS, LRU off.*

Concrete scope:
1. New NVFP4-aware host mirror for the **KV pool** only: packed `H*D/2` rows plus
   `k_scale_buffer`/`v_scale_buffer` in the flat page; keep donor `page_first`
   ordering and the existing `<H>_<S>.bin` data filename.
2. Two new registered components `kvscale_k`, `kvscale_v` → `<H>.kvscale_*_<S>.bin`;
   `batch_exists_v2` folds the usable prefix to the min of data + both scales.
3. `representation.json` with geometry and per-layer global scales.
4. `HiCacheFile` write-complete (group fsync + dir fsync) and controller ack gating (§6).
5. Fix/remove the QSA guard only if the target is actually QSA; otherwise leave it.
6. Verify with the 8-request / BS1-8 workload: identical prefix after restart,
   `cached_tokens` ≈ prefix, exact output match, and both scale sidecars present.
   Record write bytes and fsync latency; this is the baseline for §7.

Deliberately **out of M1**: separate `DRAFT` scale sidecars, Mamba/PLE layout changes
(already persisted), buffer-mode page cache wiring, NIXL, GDS, eviction/LRU.

Follow-ons: M2 = draft sidecars and QSA path; M3 = full A/B page-cache + SSD study;
M4 = NIXL/GDS only if M3 shows cold reload is the bottleneck.

---

## 9. Explicit unknowns (do not guess; resolve before coding)

1. **Model geometry**: `H_k/H_v`, `D_k/D_v`, `Lt`, whether `16 | D`, and TP/CP ranks —
   read `$TARGET_MODEL/config.json` and the launcher's `--json-model-override-args`.
   Launcher currently sets `PAGE_SIZE=64`, `KV_DTYPE=fp8_e4m3`, 24 Mamba slots; the
   requested port changes the dtype and 48 slots, not necessarily `P`.
2. **Which draft path** is active: packed native MTP layers in the KV host page, or
   separate `PoolName.DRAFT` sidecar; draft layer count/dims and whether draft KV is
   NVFP4. Confirm from the spec worker / `build_hicache_draft_sidecars` caller.
3. **QSA vs MHA target**: donor hard-fails HiCache + QSA NVFP4; if the target is QSA,
   M1 must also replace the QSA host mirror and lift that guard. If MHA, the donor
   path is not guarded but the row-size/scale mismatch remains (code-level finding;
   runtime behavior cannot be confirmed without a GPU).
4. **Mamba/PLE shapes**: `MambaPool.mamba_cache.temporal/conv` shapes and dtypes,
   `ShortConvPool.state_shape`, `NGramPool.context_len`, and whether both siblings are
   enabled in this checkpoint. Needed for absolute bytes/slot.
5. **Global scale values / SM100 ×6 adjustment**: must be taken from the actual
   checkpoint and `load_scales_from_model()` path; never inferred from file bytes.
6. **Pinned arena sizing**: whether `SGLANG_HICACHE_PREPIN_BYTES` is set by the target
   launcher and can cover KV + Mamba + PLE mirrors plus the 10 GiB host reserve.
7. **Write policy**: `write_through` vs `write_back` in the target launcher; this
   changes when pages reach storage and therefore the write-volume/temperature result.
8. **Filesystem details**: exact ZFS dataset/mount for the file root, ARC settings,
   NVMe device name/smartctl capability, and whether the port-8001 benchmark shares
   that dataset. All cold-arm actions require owner coordination.
9. **Cross-rank ack semantics** for the new write-complete (whether any collective is
   required) must be validated in a multi-TP/PP run; M1 should avoid adding one.

---

## 10-line summary

1. Donor `HiCacheFile` persists one raw `.bin` per page/component, keyed by chained page hash + rank/model suffix, using `os.replace` with no fsync and no write-complete.
2. Write policy acks D2H DMA first (`hiradix_cache`), then storage after `batch_set` returns; crash-durable completion does not exist.
3. `file` uses the generic path: `host_pool.get_data_page(flat=True)` → `batch_set` → raw bytes; sidecar pools use `batch_set_v2` with the same flat-page convention.
4. NVFP4 KV per token/layer needs packed K+V (`H*D/2` each) **plus** K/V E4M3 block scales (`H*D/16` each); global FP32 scales are per-layer model data, not per page.
5. Symmetric total = `9/8*H*D` bytes/token/layer = 56.25 % of FP8 KV write volume; scales add another 1/9 of packed data.
6. Donor host mirror uses logical `H*D` rows and has no scale buffers — the code-level root of dropped/truncated NVFP4 pages; HiCache + QSA NVFP4 is explicitly hard-failed.
7. Proposed layout: keep `<H>_<S>.bin` for data, add `<H>.kvscale_k_<S>.bin` and `<H>.kvscale_v_<S>.bin`, plus a `representation.json` manifest pinning geometry and global scales; Mamba/PLE and draft components keep donor names.
8. Write-complete = group `fsync` + directory `fsync` in a `HiCacheFile` writer thread, controller gates `backuped` on durability, and failures invalidate metadata-cache positives.
9. Experiment: 8 requests / BS1-8 / 48 Mamba slots / pinned PLE, LRU off, A/B warm page-cache vs cold reload — `drop_caches` does not evict ZFS ARC, so bound ARC or use an isolated mount; log bytes written and `smartctl` temperature/endurance.
10. Smallest milestone: KV-only NVFP4 packed bytes + both scale sidecars + write-complete through the `file` backend, verified by restarted-prefix restore; NIXL/GDS only if cold reload is measured to be the bottleneck.
