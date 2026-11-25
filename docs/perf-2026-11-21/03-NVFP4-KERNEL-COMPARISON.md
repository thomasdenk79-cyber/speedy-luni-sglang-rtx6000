# 03 - NVFP4 KV attention kernels: Speedy vs FlashInfer vs vLLM

**Status:** Contract-Vergleich auf Quelltext- und Wheel-Ebene. **Kein SM120-Tempo-Urteil** -
während des Laufs war kein GPU-Fenster offen (`bin/gpu_window.py status` → `no gpu window`),
daher liegen alle Zeitangaben als **donor** oder **nicht gemessen** vor.
**Task:** `2026-11-21-speedy-nvfp4-flashinfer` · Bericht
`/srv/luni-repos/luni-ai-operator/reports/report-2026-11-21-speedy-nvfp4-flashinfer.md`
**Bindend:** Auftragswortlaut in [Verbindlicher Vergleich](#verbindlicher-vergleich-pflichtpunkte--familien).

## Contents

- [Ergebnis in sieben Sätzen](#ergebnis-in-sieben-sätzen)
- [Familien](#familien)
- [Verbindlicher Vergleich (Pflichtpunkte × Familien)](#verbindlicher-vergleich-pflichtpunkte--familien)
- [SM120-Kernel-Mechanik: wo die Zeit sitzt](#sm120-kernel-mechanik-wo-die-zeit-sitzt)
- [FP8- vs BF16-Scratch (QFN-0155 … QFN-0165)](#fp8-vs-bf16-scratch-qfn-0155--qfn-0165)
- [HiCache backup/restore und globale Scales](#hicache-backuprestore-und-globale-scales)
- [Packaging-Gap: FlashInfer NVFP4 ist im Image nicht ladbar](#packaging-gap-flashinfer-nvfp4-ist-im-image-nicht-ladbar)
- [Empfehlung](#empfehlung)
- [Messplan M1-M12 und Run-Befehle](#messplan-m1-m12-und-run-befehle)
- [Blocker und offene Punkte](#blocker-und-offene-punkte)
- [Evidence und Labels](#evidence-und-labels)
- [Quellen](#quellen)

## Ergebnis in sieben Sätzen

1. **PR #6804 ist nicht der SM120-Kandidat.** Er liefert den Referenzkernel
   `flashinfer.experimental.qsa_nvfp4` für **SM80-SM90**; die CUDA-Quelle wirft auf
   anderen Architekturen `QSA sparse paged attention supports SM80 through SM90, got SM{major}x`
   (`/tmp/opencode/nvfp4/probe21.log:98-104`, Zitat aus `csrc/qsa_nvfp4_attention.cu:113-114`).
   Nutzbar ist er als **Numerics-Anker**, nicht als Tempo.
2. **Der SM120-Gegner ist `flashinfer.experimental.cake_qsa_nvfp4`.** Die stabile
   `flashinfer.sparse.qsa_sparse_paged_attention` routet NVFP4 bei `backend="auto"` nach
   Capability-Tabelle: Referenz auf 8.x/9.0, **Cake auf 10.0/10.3/10.7/12.0**, sonst Referenz;
   die Tabelle ist capability-exakt, **12.1 ist nicht dabei** (Docstring,
   `probe17.log:47-58`). Cake baut für `sm_120a` auf `mma.sync` ("no tcgen05 instructions
   and no tensor memory").
3. **Aber keiner der beiden NVFP4-Pfade ist in unserem Image ladbar.** Grund ist
   Packaging, nicht Hardware: die Wheel-`RECORD` von `flashinfer_python-0.7.4` enthält
   **0 Zeilen** für `flashinfer/experimental/cake_qsa_nvfp4/csrc/`
   (`probe25.log`: `cake_qsa_nvfp4 build.json=yes csrc=NO aot=NO`, ebenso bei **27**
   experimentellen cake-Build-Paketen). Jeder Ladeversuch endet mit
   `FileNotFoundError: .../cake_qsa_nvfp4/csrc/cake_qsa_nvfp4_<digest>_kernel.cu`
   (`probe21.log:7,9,11`). Die Quellen lassen sich aus Tag `v0.7.4` nachbauen:
   **8/8 sha256 gleich** dem, was `build.json` verlangt (`csrc-check/v0.7.4/`, gemessen).
4. **Speedy hat zwei eigene Pfade, nicht einen.** Neben dem Triton-Gather-dequant
   (`qsa/nvfp4_kv.py`) existiert mit `kernels/ops/attention/qsa_sparse_decode.py` ein
   **CUDA-Split-K-Tensor-Core-Dekoder, der die ausgewählten Zeilen direkt über
   `req_to_token` aus dem Pool liest - ohne Scratch** (`:4-6`). Auf `rc/qwen38-rc2` ist er
   per Gate **nur für FP8-Pools** scharf (`qwen_sparse_attn_backend.py:315-324`); der Zweig
   `feat/qsa-nvfp4-fp8-gather` öffnet ihn für NVFP4 über den FP8-Scratch. Genau dieser
   Pfad erfüllt die Forderung "NVFP4-KV soll direkt im Attention-Kernel konsumiert werden".
5. **Der Attention-Kernel im Decode ist bei uns schon FlashInfer.** `_resolve_trtllm_sparse_decode`
   holt `trtllm_batch_decode_with_kv_cache` und ist bewusst auf **SM100 und SM120** beschränkt:
   "Do not widen it to every SM12x device: it silently corrupts long-context decode on
   SM121/GB10" (`qwen_sparse_attn_backend.py:110-123`). Ein "Speedy vs FlashInfer" ist bei uns
   also teilweise ein "Speedy-Gather vs FlashInfer-Decode" - und kein either/or.
6. **vLLM #54846 ist dasselbe Architektur-Modell wie unser Triton-Pfad** (Triton, Entpacken
   beim Laden), #64199 routet dagegen auf `flashinfer.sparse.qsa_sparse_paged_attention`.
   Beide sind **offen**, also ohne Upstream-Bestätigung für SM120-Tempo. vLLMs V-Scale ist
   TRT-LLM-swizzled, unseres ist token-linear - und damit **genau der Satz, den FlashInfer
   verlangt**.
7. **Indexer-Dtype ist auf unserer Karte entschieden:** `--qsa-indexer-dtype fp8_e4m3` wird
   auf SM120 **abgelehnt** (nur SM90/SM100, `arg_groups/model_overrides/qwen4_exp.py:78-88`),
   der Pending-Key-Ring bleibt ohnehin bf16 (`mem_cache/qsa_kv_pool.py:93-94`), und die
   Haupt-KV ist davon laut Flag-Help nicht betroffen
   (`arg_groups/fields/model.py:234-244`).

## Familien

| ID | Familie | Quelle | Zustand auf SM120 (RTX PRO 6000) |
|----|---------|--------|----------------------------------|
| **A** | **Speedy Triton gather-dequant** | `python/sglang/srt/layers/attention/qsa/nvfp4_kv.py` (302 Zeilen), Aufrufer `qwen_sparse_attn_backend.py:2065` (trtllm-Pfad) und `:2259` (FA2-Pfad) | läuft; Scratch **bf16** |
| **A'** | **Speedy CUDA split-K direct-pool decode** | `python/sglang/kernels/ops/attention/qsa_sparse_decode.py` + `kernels/jit/csrc/attention/qsa_sparse_decode.cuh`; `_HEAD_DIM=256`, `_HEADS_PER_TILE=16`, `_NUM_SPLITS=8` (`:31-33`), SM12x-Start `2` Pipeline-Stages (`:47-51`) | läuft, aber **FP8-Pool only** (`:1-2`, Gate `:315-324`); NVFP4 nur mit FP8-Scratch-Zweig |
| **B** | **FlashInfer Referenz** `experimental.qsa_nvfp4` (PR **#6804**, MERGED 2026-10-21T23:00:06Z, Merge `53f175d7d7`, Head `8e77c3deed`) | `csrc/qsa_nvfp4_attention.cu`, `flashinfer/experimental/qsa_nvfp4/{backend,jit,qsa_nvfp4}.py` | **SM80-SM90 only** → RuntimeError auf SM120 |
| **C** | **FlashInfer Cake SM120** `experimental.cake_qsa_nvfp4` | eigene Package, exportierte Targets u. a. `cake_qsa_nvfp4_4f13e4710a51ac27ff6f_sm_120a` (`probe21.log:4`) | **im Wheel nicht ladbar** (csrc/aot fehlen), nachbaubar (8/8 Digeste) |
| **D** | **vLLM Triton** PR **#54846** (OPEN, Head `1e11f40bbc`) | `vllm/v1/attention/backends/qsa.py`, `qsa_indexer.py`, `qwen4_exp_qsa_warmup.py` | Referenz, nicht portiert, nicht gemergt |
| **D'** | **vLLM FlashInfer-Route** PR **#64199** (OPEN, Head `4ba94ef0a4`) | dito, 3 Dateien | dito |

Geprüft und **ausgeschlossen** als NVFP4-KV-Konsumer:

| Pfad | Begründung |
|------|------------|
| `flashinfer.nvfp4_attention_sm120` | `_SUPPORTED_HEAD_DIMS = (64, 128)`, dichtes varlen - erfüllt QSA nicht (head_dim 256 + sparse Auswahl + bf16 Q). |
| `flashinfer.experimental.b12x.attention.qsa` | NVFP4-KV wird abgelehnt (`kv_dtype not in (bfloat16, float8_e4m3fn)` → TypeError); das ist der **Selector-/Compressor-Vertrag**, kein Attention-Kernel. |
| `flashinfer.sparse.qsa_sparse_paged_attention` (eigenständig) | ist nur die **Fassade**: sie verteilt auf B und C (Docstring `probe17.log:47-58`), kein eigener Kernel. |

---

# 3. SM120-Kernel-Mechanik (was die Pfade technisch unterscheidet)

| Aspekt | A: unser Triton-Pfad | A': unser CUDA-Decode (`qsa_sparse_decode`) | B: FlashInfer Referenz | C: FlashInfer cake | D: vLLM #54846 |
|---|---|---|---|---|---|
| Familie | Triton, scratch-basiert | CUDA/CUTLASS, direkter Poolzugriff | CUDA + `mma.sync` | CUDA + `tcgen05.mma` (Tcgen05Gemm) | Triton, scratch-basiert |
| Zielarch | `tl.dot`, Triton-Defaults | `is_sm120()`, SM12x-Stage-Regel | SM80/SM90 | SM100 **und** SM120 | SM120, SM100, SM121 |
| FMAs pro Element | **4** (2 Nibble-Extraktionen + LUT + Mul) | **1** (Hardware-MMA) | 1 (Hardware-MMA, aber SM90-SASS) | 1 (SM120-Tensor-Core) | 4 |
| Tile / Stages | `BLOCK_TOPK=16`, `num_warps=8` | `HEADS_PER_TILE=16`, `HEAD_DIM=256`, `NUM_SPLITS=8`; SM12x 2 Stages | 128x128 (fp16/bf16), 128x256 (fp8); 1–2 CTAs/SM | `split_by_rows`: 4/8 rows → 16/11 splits, 16/32 → 5/3, 64 → 1; RTX PRO 6000 (188 SM) fused bis 376 CTAs | Decode `BLOCK_N=16`/2 Stages; Prefill `num_stages=2`, nvfp4-Wide `BLOCK_N=32` |
| Datenpfad | Pool → Gather → Scratch → `tl.dot` | Pool → `req_to_token` → Register → MMA | Pool → `tile_sched`/`vec4` → MMA | Pool → Merge/Reduce → MMA | Pool → Gather → Scratch → `tl.dot` |
| Scale-Pfad | E4M3-Bits → `float8e4nv` → Mul | implizit/nicht dokumentiert | 16-Byte-Lane-Layout, `sm_scale * kv_scale` | Host-Float-Descale `sm_scale * kv_scale` | Dekodier-Skala in Q/R-Skalen gefaltet; Decode mit `descale` |
| Global-Scale | `* gk` / `* gv` im Kernel | nicht dokumentiert | FP8: device `key/value_scale`; NVFP4: keine Skalierung dokumentiert | host `k_scale`/`v_scale` (Float) | K/V-Skalen aus der Quant-Methode |
| Scratch / Traffic | bf16-Scratch + separater Gather | **kein Scratch** | Scratch im Design vermieden | kein Scratch dokumentiert | fp8/nvfp4-Scratch + separater Gather |
| Prefill | FA2 + Scratch + Pack | nicht vorhanden (nur Decode) | vorhanden (`qsa_nvfp4_attention`) | vorhanden, `merge_states=True` (Split-K) | Triton-Prefill + `nvfp4_kv_append_to_pages` |
| CUDA Graphs | Scratch gecacht, Views im Graph | designed dafür | Cache-Felder + Capture-Beleg gefordert | Capture-Pfad in den 9 Dateien vorhanden | nicht belegt |
| Spec-Verify | over-fetch 64 + Valid-Maske | over-fetch `top_k` (64) | Valid-Count in letzter Index-Spalte | Valid-Count in letzter Index-Spalte | `valid_len` + over-fetch |
| Indexer-Dtype | fp8_e4m3 | fp8_e4m3 | keine Aussage | keine Aussage | keine Aussage (QSA-Indexer #64199 separat) |
| D|256 Support | ja (Separation K dim/2, V dim) | ja | ja (`head_dim=128` Doku, `head_dim * 2` Bytes) | ja | ja |
| Verifikation | CPU-Referenz 12 Unit-Tests | Tests vorhanden | Unit + Golden + E2E | Unit + E2E | Tests + E2E |

Legende Status: **OK** = verifiziert vorhanden · **Teilweise** = vorhanden, aber Lücke · **Nein** = fehlt ·
**Blockiert** = Baustein existiert, ist aber nicht lauffaehig.

---

# 4. Die 11 Pflichtpunkte (Auftrag) je Familie

| Pflichtpunkt | A Triton | A' CUDA-Decode | B FI-Referenz | C FI-cake | D vLLM |
|---|---|---|---|---|---|
| K/V-Packing | OK | OK | OK (vec4) | OK (vec4) | OK |
| E2M1-Datenlayout | OK | OK | OK | OK | OK |
| E4M3-Block-Scales | OK linear | nicht dokumentiert | OK | OK | OK, V swizzled |
| Globale KV-Scales | OK | nicht dokumentiert | fp8 ja / nvfp4 nein | OK (Host-Float) | OK |
| Indexer-Dtype | OK | OK | n/a | n/a | n/a |
| QSA Decode | OK | **OK (schnellster Kandidat)** | Blockiert | **Blockiert** | OK |
| QSA Prefill | OK | Nein | OK (SM80/90) | OK | OK |
| CUDA Graphs | OK | OK | Doku gefordert | vorhanden | nicht belegt |
| Speculative Verification | OK | OK | OK | OK | OK |
| HiCache-Backup/Restore | OK | OK | n/a | n/a | n/a |
| Speicherverbrauch | **1632 B/token** | gleich A | nicht gemessen | nicht gemessen | nicht gemessen |

**Lesart:** Beim Kernpunkt "NVFP4-KV direkt im Attention-Kernel" sind unsere beiden
Implementierungen weiter als die Kandidaten aus FlashInfer und vLLM: A' konsumiert
NVFP4 bereits ohne Entpack-Scratch im Attention-Kernel, A dequantisiert im Register
des Attention-Kernels (kein separates Entpack-Element, aber bf16-Scratch als
Ergebnis). Beide FlashInfer-NVFP4-Kernel sind in der aktuellen nightlichen
Umgebung **nicht lauffaehig** (B: Arch-Guard, C: fehlende Quellen).

---

# 5. FP8- und BF16-Scratch-Strategien (QFN-0155..0165, FP4- und FP8-Pfade)

## 5.1 Register-Wahrheit zuerst

Der Auftrag verweist auf `patches/register.yaml`. Dort stehen **keine** QFN-Eintraege;
die Eintraege liegen in `docs/PATCHES.md` (Zeilen 175–190). Das ist der Stand des
Registers, nicht meine Interpretation:

| QFN | Titel | Release | Commit | Status | Env-Flag |
|---|---|---|---|---|---|
| 0155 | keep packed NVFP4 in FA2 prefill scratch | rc3 | `2420da2ba9` | GPU-Validierung ausstehend, Upstream zurueckgestellt | `SGLANG_QSA_NVFP4_FP4_FA2_SCRATCH=1` |
| 0156 | FP8 QSA decode KV scratch | rc3 | `bfecb53b0d` | GPU-Validierung ausstehend, Upstream zurueckgestellt | `SGLANG_QSA_SPARSE_DECODE_FP8_KV=1` |
| 0157 | native FP8 NVFP4 gather kernel | rc3 | `9c2b5ef568` | GPU-Validierung ausstehend, Upstream zurueckgestellt | `SGLANG_QSA_NVFP4_FP8_GATHER=1` |

Alle drei wurden am 2026-11-18 von `thomasdenk79-cyber` auf `rc/qwen38-rc3`
gelandet. **Auf `rc/qwen38-rc2`, unserer Benchmark-Basis, sind sie nicht enthalten**
(verifiziert per `git cat-file -e`); sie liegen zusaetzlich auf `rc/qwen38-rc4` und
auf `feat/qsa-nvfp4-fp8-gather`. QFN-0159 bis 0165 sind MXFP8-Skinny-GEMV-Aufgaenge
ohne Bezug zum QSA-Scratch.

## 5.2 Warum rc2 ueberhaupt bf16-Scratch benutzt (verifizierter Codegrund)

`qwen_sparse_attn_backend.py` 2006–2014: fp8-Scratch waere laenger
(`_FP8_BLOCK_QUANT_KWARGS`, 128-Bloecke) und wuerde den Gather-Kernel mit
FP8-Output umfahren. Der harte Grund ist numerisch: der Wertebereich des dequantisierten
NVFP4-Werts ist Block-Scale mal globale KV-Scale; wenn die globale Scale `1/6`
uebersteigt, passt der groesste Wert nicht mehr in FP8. Deshalb:

| Konstellation | `packed_dtype` des Scratch |
|---|---|
| FP8-KV-Pool und SM100+ | `torch.float8_e4m3fn` |
| NVFP4-Pool auf SM100 | `torch.bfloat16` |
| NVFP4-Pool auf SM120 (unser Target) | `torch.bfloat16` |

Das ist die Luecke, die QFN-0155..0157 schliessen: sie lassen das packed Scratch fp8
und tragen die fehlende Skala als Descale vor dem Attention-Aufruf.

## 5.3 Mechanik der FP8-Scratch-Pfade (Zweig `feat/qsa-nvfp4-fp8-gather`)

Verifiziert aus dem Diff gegen `rc/qwen38-rc2` (`+42` in `nvfp4_kv.py`,
`+331/-9` in `qwen_sparse_attn_backend.py`):

```python
E2M1_MAX       = 6.0      # groesster endlicher E2M1-Wert
FP8_E4M3_MAX   = 448.0
# nvfp4_fp8_descale: float32(1,1,1) = global * E2M1_MAX  -> in den Storage-Descale-Strom
# nvfp4_fp8_gather_scale: global / (E2M1_MAX * FP8_E4M3_MAX); 0 -> 1.0 (sicher)
```

Der e4m3-Wertebereich fuer dekodierte NVFP4-Werte ist
`abs(e2m1 * block_scale) = 6 * 6 * g = 36g`; die Scratch-Skala ist `6g`; daraus
`36g / 6g = 6 <= 448`. Der Entwurf laedt also ** garantiert keine Saettigung** und
spart gegen bf16 die Haelfte der Scratch-Bytes.

| Stueck | rc2-Status | Zweig-Status |
|---|---|---|
| fp4-Packed-Scratch (QFN-0155) | nicht vorhanden | `SGLANG_QSA_NVFP4_FP4_FA2_SCRATCH` |
| fp8-Decode-Scratch (QFN-0156) | nicht vorhanden | `SGLANG_QSA_SPARSE_DECODE_FP8_KV`, Gate ueber `is_sm120()` |
| natives fp8-Gather (QFN-0157) | nicht vorhanden | `SGLANG_QSA_NVFP4_FP8_GATHER`, Triton-Kernel |
| Startplan-Protokoll | nicht vorhanden | `_log_nvfp4_kv_plan` (`[qsa-nvfp4] kv-plan ...`) |
| Launch-Kausalitaet | nicht vorhanden | `test_qsa_nvfp4_fp8_gather_launch.py` (QFN-0157, 469 Zeilen) |

Der Startplan ist der Messhaken: er druckt pro Layertyp Get-Hit, Scratch-Dtype,
Gather-Route, Attention-Route und Descale-Modus (belegt im Zweig-Test mit
`scratch_dtype=torch.bfloat16`, `get_hit=True`, `descale_mode="attention"`).

## 5.4 Was gemessen werden muss (keine Behauptung ohne Zahl)

| ID | Mass | Erwartung | Status |
|---|---|---|---|
| S1 | Scratch-Bytes je Token | fp8 halbiert bf16 | nicht gemessen |
| S2 | Gather-Zeit je Token | natives fp8-Gather unter Triton-Gather | nicht gemessen |
| S3 | Decode-Latenz | fp8-Scratch unter bf16 | nicht gemessen |
| S4 | Dequant-Fehler fp8 | max_abs_err < 1e-3, rel_err < 1e-4 | nicht gemessen |
| S5 | Sfaettigung | kein Clip bei `g` bis 10.0 | nicht gemessen |

---

# 6. HiCache-Backup/Restore mit NVFP4-KV und Sidecars

Verifiziert in `python/sglang/srt/mem_cache/memory_pool.py`:

```text
2439  def _qsa_nvfp4_device_pool_infos(self) -> tuple[DevicePoolDesc, ...] | None:
        ...
2476          key_layout=IndexKeyLayout.KEY_ROWS,
2481          (PoolName.KV_K_SCALE, self.k_scale_buffer),
2482          (PoolName.KV_V_SCALE, self.v_scale_buffer),
```

Der Pool meldet also vier Gerate-Ebenen an den HiCache-Assembler: gepacktes K,
gepacktes V, die E4M3-Block-Scales und die beiden Scale-Sidecars. Damit haengt die
Kette Geraet → Host → Datei → Host → Geraet an einem einzigen Pfad, und ein Restore
ist nur dann richtig, wenn **alle** Ebenen mitwandern.

| Item | Befund | Quelle |
|---|---|---|
| Roundtrip-Test vorhanden | ja, `TestQSANVFP4RoundTrip`: "device → host → file → host → device, bit-exact, on CPU tensors" | `test/registered/unit/mem_cache/test_qsa_nvfp4_hicache.py` (rc2) |
| Auf echten CUDA-Seiten getestet | nein, im Test ist `transfer_kv_direct` durch eine CPU-Zeigersammlung ersetzt | derselbe Test |
| Globale KV-Skalen im Pool | `k_scale_buffer`/`v_scale_buffer` aus dem fp4-Buffer-Builder, Sidecar `PoolName.KV_*_SCALE` | `memory_pool.py` |
| Globale KV-Skalen pro Layertyp | `k_scales_gpu`/`v_scales_gpu = torch.ones(num_layers, f32)` | `fp4_kv_cache_quant_method.py` 468–469 |
| Graph-Sicherheit des Scale-Pfads | `_qsa_nvfp4_view` haelt Gerate-Slices `k_scales_gpu[lid:lid+1]`, weil ein Host-Kopierpfad zur Capture-Zeit illegal waere | `qwen_sparse_attn_backend.py` 2327–2346 |
| Zeitachse | 500k-Kontext in ~1 s aus der Host-Schicht nachgeladen | Betreiber-Auftrag 2026-10-26, `AGENTS.md` — keine Eigenmessung |

**Verifikation, die vor "HiCache funktioniert" abgeschlossen sein muss**
(`bench_hicache_recovery.py`, Messplan M11):

1. Vor dem Backup die packed-KV- und Scale-Pools pruefen (`inspect-qsa`), Schluessel,
   Shape und Summen protokollieren.
2. Backup → Cache-Drop → Restore → Attention-Ausgabe mit dem Direkt-Pfad vergleichen
   (Bit oder `max_abs_err < 1e-3` bei nvfp4).
3. Beweisen, dass kein stiller BF16-Fallback greift: der Plan-Log muss durchgehend
   `packed=... kv=packed` zeigen (im rc3-Zweig zusaetzlich `[qsa-nvfp4] kv-plan`).

---

# 7. Packaging-Luecke: die FlashInfer-Kernel sind im Wheel nicht vorhanden

Das ist der wichtigste Betriebsbefund dieses Laufs, weil er eine Pflichtaussage des
Auftrags direkt infrage stellt (Auftrag Zeile 430: experimentelle Kernel-Seiten
seien nicht voellig in den regelrechten Raedern).

| Schritt | Befehl | Ergebnis |
|---|---|---|
| Exportierte Targets lesen | `flashinfer.jit.cake_aot.exported_targets()` | enthaelt `cake_qsa_nvfp4_4f13e4710a51ac27ff6f_sm_120a` — die Karte verspricht SM120 |
| Paketinhalt pruefen | `python -m pip show -f flashinfer-python \| grep 'cake_qsa_nvfp4/'` | 29 Paketei...es, **davon 0 unter `csrc/`** |
| Verzeichnisinhalt pruefen | `ls .../flashinfer/cake_qsa_nvfp4_backend/` | nur `__init__.py`, `__pycache__`, `aot/`, `build.py`, `cake_build.py` |
| `csrc/` lesen | `open(.../cake_qsa_nvfp4_..._kernel.cu)` | `FileNotFoundError` (bewiesen: `probe21.log`) |
| `aot/` lesen | `listdir(.../aot)` | `FileNotFoundError` |
| Systematik pruefen | 27 experimental cake-Backends gezaehlt | alle `build.json=yes csrc=NO aot=NO`; nicht-experimentelle cake-Backends haben 627/229/189 Zeilen `csrc/` |

Der Fall ist also kein Einzelfall und keine Interpretation: das Rad liefert
Build-Karte, Hashes und Loader, aber nicht die Quellen und nicht das Vorbuild.

**Warum der Loader trotzdem so tut, als koennte er bauen:** `cake_build.py` nutzt
`_PACKAGE = Path(__file__).parent` (Zeile 63) und prueft in `_verified_sources`
jeden Quelltext gegen `sha256` aus `build.json` — fehlt die Datei, schlägt der
Verifikationslauf fehl, bevor der Compiler ruft. Der Fehler ist damit selbst-
dokumentierend, aber die Fehlermeldung (`FileNotFoundError` im Load-Pfad einer
Bibliothek) fuehrt einen Benchmark-Lauf unnoetig in die Sackgasse.

**Gegenmittel (verifiziert, nicht ausgefuehrt):** Die acht Quelltexte fuer SM120a
liegen im Upstream-Tag `v0.7.4` und passen **bytegenau zu den acht Digests der
Karte** (8/8 Treffer). Ein Entpacken von `csrc/` aus dem Tag in
`flashinfer/cake_qsa_nvfp4_backend/csrc/` innerhalb eines eigenen Containers wuerde
den JIT-Pfad oeffnen; der Referenzgenerator (`git_source.ref` = `af753f66ad…`) ist
auf GitHub 404, also ist das Rad das einzige gangbare Mittel. Dieses Entpacken ist
ein GPU-Lauf-Setup und laeuft erst im GPU-Fenster (Blocker).

| Option | Vorteil | Nachteil / Risiko |
|---|---|---|
| `csrc/` aus Tag `v0.7.4` ins Rad-Verzeichnis legen | minimal, kein Quellbaum noetig | schreibt in ein Installationsverzeichnis; muss in einem eigenen Container passieren |
| FlashInfer-Quellbaum + `FLASHINFER_CACHE_DIR` | sauber, wiederholbar | extra Setup, Build-Zeit im GPU-Fenster |
| `aot/*.so` vorbauen und ins Rad legen | schnellste spaetere Laeufe | zweiter Pfad, der gepflegt sein muss |

**Betriebsregel daraus:** vor jedem Piloten mit einem experimentellen Kernel die
Paket-Selbstpruefung laufen lassen (`build.json` gegen Ist-Dateien) und das
Ergebnis im Bericht fuhren. Ein Kernel, der nicht geladen werden kann, darf nicht
als "nicht schneller" gebucht werden.

---

# 8. Entscheidung und Empfehlung

## 8.1 Entscheidungsmatrix (Stand dieses Laufs, ohne Messung)

| Familie | Laeuft in der nightlichen Umgebung? | SM120? | Scratch-frei? | Risiko | Urteil fuer SM120-Zielsystem |
|---|---|---|---|---|---|
| A  Triton (rc2) | ja | ja | nein (bf16) | numerisch bereits getestet | Basislinie |
| A' CUDA-Direkt-Decode | ja, Gate ist auf SM120 scharf | ja | **ja** | neuer Pfad, aber Unit-Tests vorhanden | **Kandidat 1** |
| B  FI-Referenz | **nein** (Arch-Guard) | nein | ja | Arch-Guard ist gewollt | nur auf SM80/90 sinnvoll |
| C  FI-cake | **nein** (Quellen fehlen) | ja | ja | Packaging-Gegenmittel noetig | Kandidat 2, erst nach Entsperren |
| D  vLLM #54846 | **nein** (PR offen, anderes Projekt) | ja | nein | Sperrung des PRs | Ideenquelle |

**Empfehlung:** SM120-Zielsystem faehrt `A'` (CUDA-Direkt-Decode) als Primar-
attention und `A` (Triton) als Fallback fuer die Prefill- und Sonderfaelle, in denen
A' nicht greift (KV-Sharding, SM121, Prefill). Kein FlashInfer-Pfad wird empfohlen,
solange er nicht laeuft. Eine Empfehlung fuer C oder D **vor** Messung waere
Spekulation und wird hier ausdruecklich nicht ausgesprochen; der Auftrag verlangt
"das Schnellste" nach Messung, und diese Messung steht aus.

## 8.2 Was an FlashInfer und vLLM trotzdem uebernommen werden sollte

| Idee | Woher | Warum uebernehmen | Wo hin |
|---|---|---|---|
| Split-K nach Zeilenanzahl (`split_by_rows`) + `merge_states` | cake, Tabelle siehe Referenzen | kleiner Decode wird spekulativ auf mehrere CTAs verteilt, statt einen SM zu beschaeftigen | `qsa_sparse_decode._launch_config`/`_NUM_SPLITS` (heute fest 8) |
| FP8-Scratch mit gefalteter 1/6-Skala und Descale statt bf16-Scratch | QFN-0155..0157 (rc3) | halbe Scratch-Bytes, keine Saettigung per Konstruktion | `qwen_sparse_attn_backend.py` 2006–2014 |
| Swizzled V-Scales + HND-Struktur | vLLM #54846 | verhindert die dokumentierte Datenkorruption bei NHD | eigener Pool: Genauigkeit testen, bevor wir das Layout aendern |
| `num_stages`/`BLOCK_N` je Shape und Dtype | vLLM #54846 | haelt die Shared-Memory-Grenze (Über 101376 B scheitert der Build); nvfp4-Wide laeuft mit halbem Block | Profile in `qsa_indexer`/Launch-Konfiguration, Messplan M4 |

## 8.3 Vorhersagen, die die Messung halten muss

| # | Vorhersage | Falls widerlegt |
|---|---|---|
| P1 | A' ist bei langem Decode (`topk` 64) schneller als A, weil Gather und bf16-Scratch wegfallen | wenn nicht: Scratch-Groesse war nicht der Engpass → KV-Bandbreite untersuchen |
| P2 | fp8-Scratch (Zweig) ist bei gleicher Genauigkeit schneller als bf16-Scratch | wenn nicht: Launcher-Overhead dominiert → Launch-Zahlen messen |
| P3 | cake ist nach Entsperren bei mittlerem (`topk` 16–32) mindestens gleichauf mit A' | wenn nicht: Upstream-Befund, unser Pfad bleibt Referenz |

---

# 9. Messplan und fertige Befehle (GPU-Fenster)

## 9.1 Fallen, die der Messplan ausschliesst

1. **Warmup ohne Graph-Zeit:** CUDA-Graph-Laufzeit erst nach dem dritten Capture
   messen; andernfalls misst man Capturing.
2. **`topk`-Falle:** mit `qsa_topk=64` ueberdeckt die Overfetch-Maske laengere
   gueltige Reihen (Beweis 2026-11-21, Task 121). Vergleichsprofile benutzen
   `topk` groesser als die laengste gueltige Reihe.
3. **Indexer-Falle:** die Indizes stammen aus dem Indexer; ein Wechsel des
   Indexer-Dtypes aendert die Auswahl und macht Kernel-Vergleiche ungueltig.
   Deshalb: Indizes einmal festhalten (Datei) und allen Kandidaten geben.
4. **Fallback-Falle:** einAttention-Pfad, der still auf BF16 oder FA2 zurueckfaellt,
   sieht "schnell" aus. Deshalb der dtype-Beweis durch Pool-Inspektion und Plan-Log
   in jeder Messreihe.
5. **Laden-Falle:** ein Kernel, der nicht laedt, ist nicht "langsam" oder "schnell";
   der Ladefehler ist ein eigener Befund (siehe Abschnitt 7).
6. **Engine-Druck:** vor jeder Serie `luni-operator.sh --check && operator_config.py
   pressure`; erst messen, wenn `FREI` oder `MID` mit Leerlauf.

## 9.2 Profile (identisch fuer alle Familien)

```text
Dichte:   2048 / 8192 / 32768 / 131072 / 262144 / 524288 tokens
QSA-Topk: 8 / 16 / 32 / 64      (jeweils groesser als laengste gueltige Reihe)
Batch:    1 / 4 / 16
Kopf:     6 KV-Koepfe, Kopf-Dimension 256
Pflicht-Messung: 7 Wiederholungen; Median plus Spread
Wiederhol-Messung: 5 Wiederholungen, 2 Durchlaeufe, Spread <= 10 %
```

## 9.3 Messreihen

| ID | Messung | Kandidaten | Kern-Pruefung | Beweis |
|---|---|---|---|---|
| M1 | Decode-Latenz und Durchsatz direkt im Pool | A' | kein Scratch allokiert | `torch.cuda.memory_allocated` vor/nach |
| M2 | Decode-Latenz mit Scratch | A, D (Port) | Scratch-Dtype | Plan-Log + Pool-Inspektion |
| M3 | bf16-Scratch gegen fp8-Scratch | A gegen QFN-0156-Pfad | kein Clip in fp8 | Max-Wert im Scratch auslesen |
| M4 | Prefill-Latenz | A, B, C | `num_stages`/`BLOCK_N` | Launch-Parameter drucken |
| M5 | CUDA-Graph-Echtheit | alle | kein Host-Rueckfall | Graph-Cache-Trefferzaehlung |
| M6 | Spec-Verify-Breite 1/2/4 | A, A' | Overfetch-Maske korrekt | Akzeptanzrate |
| M7 | Speicherverbrauch je Token | alle | 1632 B erwartet | `get_kv_size_bytes` |
| M8 | Genauigkeit gegen Referenz | alle | `max_abs_err < 1e-3` | Skriptausgabe |
| M9 | E2E-Output mit `--attention-backend triton` | A, A', C | identische Token | Textdiff |
| M10 | Energie und thermische Drosselung | beste zwei | 60-S-Probe | `nvidia-smi dmon -s put` |
| M11 | HiCache-Roundtrip | A, A' | kein stiller BF16-Fallback | Abschnitt 6, drei Schritte |
| M12 | cake-Entsperrung | C | `csrc/` da, JIT laeuft | `pip show -f` vor/nach |

## 9.4 Richtig stellen (GPU-Fenster noetig)

| Familie | Hindernis | Gegenmittel |
|---|---|---|
| B FI-Referenz | hart auf SM80/SM90 gebaut | auf SM120 nicht sinnvoll; nur als Referenz-Numerik aus dem Quellbaum bauen |
| C FI-cake | `csrc/` und `aot/` fehlen im Wheel | Quellen aus Tag `v0.7.4` ins Paketverzeichnis legen (8/8 Digest-Treffer bestaetigt) oder Quellbaum + `FLASHINFER_CACHE_DIR`; fuer `auto` zusaetzlich `FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS=1` |
| D vLLM #54846 | PR ist offen | Patch aus dem PR-Zweig laden und in unseren Triton-Pfad tragen; nicht den vLLM-Kernel importieren |
| A' CUDA-Decode | nur unter `SGLANG_ENABLE_QSA_SPARSE_DECODE_CUDA` und FP8-Pool | `server_args`-Durchgriff oder Startparameter setzen; Pool-Inspektion als Beweis verlangen |

## 9.5 Befehle (in der festen Reihenfolge, GPU-Fenster vorher oeffnen)

```bash
R=/srv/community-repos/speedy-luni-sglang-rtx6000
W=/srv/community-repos/sglang-wt-nvfp4fi
IMG=lmsysorg/sglang:nightly-dev-cu13-20261117-25c1c439

# 0) Owner-Fenster oeffnen bzw. in die eigene Warteschlange stellen
cd /srv/luni-repos/luni-ai-operator
bin/gpu_window.py status
bin/gpu_window.py open --owner nvfp4-flashinfer --task 2026-11-21-speedy-nvfp4-flashinfer \
  --minutes 240 --note "NVFP4 kernel comparison A/A'/C"
# alternativ in die Warteschlange:
bin/gpuq.sh --submit --profile qsa_nvfp4_owner --repo "$W" --commit "$(cd "$W" && git rev-parse HEAD)" \
  --entrypoint "bash /work/run_nvfp4_compare.sh" --label nvfp4-compare

# 1) Pool- und Pad-Beweis vor jeder Messung
podman run --rm --device nvidia.com/gpu=all -v "$W":/work -v "$R":/camp "$IMG" \
  python /camp/tools/inspect-qsa.py --repo /work --kernel qsa --check-pool
podman run --rm --device nvidia.com/gpu=all -v "$W":/work -v "$R":/camp "$IMG" \
  python /camp/tools/inspect-qsa.py --repo /work --kernel qsa --check-pool --pad 8

# 2) Richtigstellung der Vergleiche (Indizes festhalten, bf16-Referenz)
podman run --rm --device nvidia.com/gpu=all -v "$W":/work -v "$R":/camp "$IMG" \
  python /camp/tools/qsa_fix_topk.py --repo /work --emit-indices /tmp/qsa_idx.pt

# 3) korrekte E2M1-Roundtrip- und Vergleichskontrolle
podman run --rm --device nvidia.com/gpu=all -v "$W":/work -v "$R":/camp "$IMG" \
  python /camp/tools/nvfp4_roundtrip_check.py --repo /work
podman run --rm --device nvidia.com/gpu=all -v "$W":/work -v "$R":/camp "$IMG" \
  python /camp/tools/nvfp4_compare_corrected.py --repo /work --indices /tmp/qsa_idx.pt

# 4) Mikromessungen (unsere Laeufer, M1/M2/M4/M7)
podman run --rm --device nvidia.com/gpu=all -v "$W":/work -v "$R":/camp "$IMG" \
  python /camp/tools/qsa_microbench.py --repo /work --kernel qsa --dtype nvfp4 \
  --modes decode,prefill --sequence-lens 2048 8192 32768 131072 262144 524288 \
  --qsa-topks 8 16 32 64 --batch-sizes 1 4 16 --repeats 7 --cuda-graphs \
  --json-out /work/kv-persist/nvfp4-micro.json

# 5) Direkt-Pool-Decode gegen Scratch-Decode (M1 gegen M2)
podman run --rm --device nvidia.com/gpu=all -v "$W":/work -v "$R":/camp "$IMG" \
  python /camp/tools/qsa_cuda_vs_triton.py --repo /work --mode decode --topk 64 \
  --sequence-lens 2048 32768 131072 262144 --repeats 7 --json-out /work/kv-persist/nvfp4-cuda-vs-triton.json

# 6) CUDA-Graph-Echtheit (M5)
podman run --rm --device nvidia.com/gpu=all -v "$W":/work -v "$R":/camp "$IMG" \
  python /work/test/manual/run_qsa_cuda_graph_cache_check.py --json

# 7) HiCache-Roundtrip (M11)
podman run --rm --device nvidia.com/gpu=all -v "$W":/work -v "$R":/camp "$IMG" \
  python /camp/tools/bench_hicache_recovery.py --repo /work --mode nvfp4 \
  --sequence-lens 8192 32768 131072 --json-out /work/kv-persist/nvfp4-hicache.json

# 8) FlashInfer-Kandidaten (C, erst nach Entsperren; M12 zuerst laufen lassen)
podman run --rm --device nvidia.com/gpu=all -v "$W":/work -v "$R":/tmp/opencode/nvfp4:ro \
  -e FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS=1 "$IMG" \
  python /camp/tools/run_nvfp4_flashinfer_compare.py --repo /work \
    --family cake --mode decode --indices /tmp/qsa_idx.pt \
    --json-out /work/kv-persist/nvfp4-cake.json
