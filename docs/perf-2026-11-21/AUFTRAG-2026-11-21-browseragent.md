# OPUS 5.5 – KONKRETER IMPLEMENTIERUNGSAUFTRAG FÜR SPEEDY
## SGLang + RTX PRO 6000 Blackwell | Performance, NVFP4-KV, HiCache/NIXL, Agenten-Scheduling

**Recherchebasis: 21.11.2026**

Du bist Lead Architect, CUDA/Triton-Kernel-Engineer, SGLang-Maintainer und Performance-Reviewer.

Dein Auftrag ist **nicht**, eine neue Architektur zu entwerfen. Du sollst vorhandene, konkret benannte Patches aus anderen Implementierungen mit unserem aktuellen Fork abgleichen, die besten Kandidaten portieren, testen und unsere Agenten-Orchestrierung anpassen.

Arbeite über OMO mit getrennten Explorer-, Implementer-, Reviewer- und Benchmark-Agenten.

**Wichtig: Wir haben bereits viel implementiert. Nicht alles neu schreiben!**

---

# 1. Unsere Repositories

## Hauptprojekt

https://github.com/thomasdenk79-cyber/speedy-luni-sglang-rtx6000

Engine-Source:

https://github.com/thomasdenk79-cyber/sglang

Pflichtdateien:

- `README.md`
- `docs/PATCHES.md`
- `patches/register.yaml`
- `docs/releases/speedy-rc2.md`
- `examples/run-rc2-production.sh`

Alle RC3- und Pending-Branches prüfen, insbesondere:

- `perf/hb-lowrow-kernels`
- `perf/hb-prune-pdl`
- `feat/draft-vocab-head-buckets`

Branch-Namen, tatsächliche Commits und Integrationsstatus zuerst lokal durch `git branch -a`, `git log` und `git merge-base` bestätigen.

## Zusätzlicher zwingender Auftrag: Luni AI Operator

https://github.com/thomasdenk79-cyber/luni-ai-operator

Dort müssen die bisherigen TASKS überprüft, aktualisiert und neue Aufgaben für diese Performance-Kampagne erstellt werden.

Der Operator soll anschließend die Arbeiten in sinnvollen, kleinen Einheiten orchestrieren können.

---

# 2. Verbindliche Zielhierarchie

## Priorität 1: Geschwindigkeit

**HyperBurst ist unsere wichtigste Speed-Referenz.**

Insbesondere C1, C2, C3 und C4.

Die veröffentlichten HyperBurst-Referenzwerte lauten unter bestimmten BetterBench-0.6.0-Bedingungen:

- C1: 369,9 tok/s
- C4: 710,2 tok/s aggregiert

Diese Werte sind keine garantierten Zielwerte für unseren anderen Checkpoint und unsere andere Konfiguration.

Reproduziere deshalb die Vergleichsbedingungen, bevor du Regressionen oder Gewinne behauptest.

Ziel:

**HyperBurst-Geschwindigkeit erreichen oder übertreffen, dabei unsere besseren Speicher- und Cache-Funktionen behalten.**

## Priorität 2: NVFP4-KV vollständig und performant

NVFP4-KV bleibt Pflicht.

Es muss mit QSA, Prefill, Decode, MTP, CUDA Graphs, Radix Cache, HiCache und NIXL funktionieren.

NVFP4-KV darf keine wesentlichen Decode-Regressionen gegenüber FP8 verursachen, die durch schlechte Dequantisierung oder ungeeignete Kernel entstehen.

## Priorität 3: Mehr GPU-KV-Kapazität

Speichereinsparung ist ausdrücklich erwünscht, wenn sie nicht unverhältnismäßig Geschwindigkeit kostet.

Reihenfolge:

1. Mehr freier VRAM und mehr Hot-KV.
2. Geringerer System-RAM-Verbrauch.
3. Geringerer SSD-Speicherbedarf.

Performance bleibt das entscheidende Kriterium.

## Priorität 4: HiCache ohne Decode-Blockierung

Storage-I/O und Backup/Restore dürfen nicht synchron den laufenden Decode-Pfad aufhalten.

Wenn Restore oder Backup laufen, müssen andere bereits laufende Requests weiterdecodieren können.

Physische Ressourcen-Konkurrenz durch USB4, GPU-Copy-Engines und CPU ist zu messen und zu begrenzen.

## Priorität 5: GPU-resident-first Scheduling

**Keine reine FIFO-Warteschlange mehr.**

Bei mehreren wartenden Requests werden diejenigen bevorzugt, deren benötigter KV-Präfix noch im VRAM liegt.

Danach kommen vollständig wiederverwendbare Host-RAM-Hits, danach SSD-HiCache-Hits.

Die Priorisierung gilt für gleichzeitig bereits wartende Requests. Es wird nicht künstlich auf möglicherweise später eintreffende Requests gewartet.

Fairness muss verhindern, dass eine SSD-residente Session unbegrenzt verhungert.

## Priorität 6: PLE bleibt gepinnt

NVIDIA FP4-Local-Checkpoint.

Gepacktes NVFP4-PLE im Host-RAM, derzeit laut Projekt rund 26,8 GiB.

**Keine weiteren Experimente mit PLE-SSD-Offloading, demand-paging oder PLE-Streaming.**

Diese Richtung haben wir bereits untersucht. Sie ist nicht mehr Bestandteil des aktiven Optimierungsplans.

Falls eine neue Komprimierung oder ein Lookup-Kernel ohne Decode-Nachteil zusätzlichen RAM spart, ist das interessant. Aber keine erneute SSD-Offload-Entwicklung.

---

# 3. Workload: 1 bis 10 Agenten und Subagenten

Der häufigste Betrieb besteht aus **1–6 Agenten**.

8–10 Agenten einschließlich Subagenten sind Erweiterungs- und Stressprofile, nicht der wichtigste Standardfall.

Agenten führen kurze Generierungen aus, starten Tools, warten auf deren Ergebnisse und senden neue Requests.

Viele Requests dauern nur 0,5–2 Sekunden.

Die effektive Decode-Parallelität schwankt deshalb zwischen C0 und C6, gelegentlich darüber.

Wir benötigen:

- Sehr schnelle C1/C2-Decodes.
- Hohe C3/C4-Leistung.
- Möglichst wenig GPU-Leerlauf.
- Schnelle Wiederaufnahme langer Sessions.
- Viele warme, wiederverwendbare Präfixe im VRAM.
- Tiefe wartende Queues ohne unnötige KV-Verdrängung.
- Möglichst wenig SSD-Restore-Traffic.

**Wichtig:** Anzahl Agenten, Anzahl wartender Requests und tatsächliche Decode-Concurrency sind getrennte Größen.

Es kann zehn Agenten und zehn wartende Requests geben, während nur vier Requests gleichzeitig decodieren.

Benchmarke `max_running_requests` 3, 4, 6, 8 und optional 10. Wähle den Standard nach realer Agentenleistung, nicht nach der höchsten künstlichen Concurrency.

---

# 4. TASK P0-001: Vorhandene HyperBurst-Speed-Patches fertigstellen

## Quelle

https://github.com/aiueo52/hyperburst-rtxpro6000

https://github.com/aiueo52/sglang-rtxpro6000

Dokumentation:

https://github.com/aiueo52/hyperburst-rtxpro6000/blob/main/docs/optimizations.md

Unser Patchregister enthält die entsprechenden Portierungsarbeiten bereits.

## P0-001A: Low-Row-Kernel

**Vorhandene QFN-0193 bis QFN-0202 übernehmen und GPU-validieren.**

| Patch | Konkrete Funktion |
|---|---|
| QFN-0193 | HyperBurst Draft-MoE W4A16 GEMV |
| QFN-0194 | Triton-MoE-Router-GEMV für 1–16 Rows |
| QFN-0195 | GDN Gated-RMSNorm mit Out-Projection-GEMV fusionieren |
| QFN-0196 | MTP Entry-FC auf SM120 Skinny-GEMV umstellen |
| QFN-0197–0202 | Shape-, Alignment-, Device- und Sicherheitsprüfungen |

Die zugehörigen Flags sind unter anderem:

- `SGLANG_OPT_DRAFT_MOE_GEMV`
- `SGLANG_ROUTER_GEMV`
- `SGLANG_NORM_INTO_GEMV`
- `SGLANG_MTP_FC_GEMV`

**Anweisung:**

1. Branch und Commit-Status feststellen.
2. Bereits implementierte Kernel nicht neu schreiben.
3. Fehlende GPU-Validierung nachholen.
4. Je Kernel M=1,2,3,4,6,8,10,16 benchmarken.
5. CUDA-Graph- und Eager-Pfad getrennt messen.
6. Auf unnötige temporäre Allokationen und Host-Synchronisation prüfen.
7. End-to-End C1/C2/C4/C6 messen.
8. Nur profitable Varianten standardmäßig aktivieren.

Keine GPU-Kernel allein aufgrund von CPU-Unit-Tests freigeben.

## P0-001B: HyperBurst PDL und MoE-Pruning

**QFN-0203 bis QFN-0214 prüfen und abschließen.**

Enthalten:

- Triton Programmatic Dependent Launch.
- MoE Singleton-Route-Pruning.
- Architektur- und Launch-Gates.
- PDL-Kompatibilität mit fusionierten Norm-GEMVs.

Flags unter anderem:

- `SGLANG_TRITON_PDL`
- `SGLANG_MOE_PRUNE_SINGLETON_TAU`

PDL auf SM120 einzeln und kombiniert messen.

**Pruning ist eine approximative Modelloptimierung.**

Deshalb:

- `tau=0` als qualitätswahrender Referenzzustand.
- Andere Werte nur mit Qualitäts- und Geschwindigkeitstest.
- Kein standardmäßig aktives approximatives Pruning ohne nachgewiesenen akzeptablen Qualitätsverlust.

Akzeptanzkriterium: Kein unbemerkter Verlust bei Codegenerierung, Tool-Calls oder langem Kontext.

---

# 5. TASK P0-002: Pennyroyal-Patches gezielt übernehmen

Quelle:

https://github.com/jpezzulli/sglang-rtxpro6000

Referenzversion: Pennyroyal 3.1.7 vom 19.11.2026.

**Nicht den gesamten Fork importieren.**

Jeden Patch gegen unseren Source-Tree und aktuellen SGLang-Upstream vergleichen.

## P0-002A: QSA Long-Prefix-Prefill

**Pennyroyal PR #162**

https://github.com/jpezzulli/sglang-rtxpro6000/pull/162

Status zum Recherchezeitpunkt: merged.

Dateien:

- `python/sglang/srt/layers/attention/qsa/mqa.py`
- `python/sglang/srt/layers/attention/qsa/qsa_indexer.py`
- `test/registered/kernels/test_qsa.py`

Dieser Patch verwendet request-relative Score-Workspaces, reduziert unnötige Initialisierung und verbessert Key-Split-Parallelität.

Pennyroyal 3.1.6 meldet ungefähr 26 % schnelleren Fresh-Prefill bei C1 und 33 % bei C4 nach 200K bereits gecachten Tokens plus 4K neuen Tokens.

**Konkrete Arbeit:**

1. Mit SGLang-Upstream PR #45791 vergleichen:
   https://github.com/sgl-project/sglang/pull/45791
2. Prüfen, ob der Optimierungspfad bereits in unserem Upstream-Commit vorhanden ist.
3. Fehlende Teile selektiv portieren.
4. Für NVFP4-KV und QSA validieren.
5. Benchmark-Matrix:
   - 50K cached + 512 neu.
   - 150K cached + 1K neu.
   - 200K cached + 4K neu.
   - 500K cached + 1K neu.
   - C1, C2, C4, C6.

**Dieser Task hat hohe Priorität**, weil kurze neue Suffixe nach langen Agentenpräfixen unser Normalfall sind.

## P0-002B: QSA-Prefill-KV-Gather fusionieren

**Pennyroyal PR #38**

https://github.com/jpezzulli/sglang-rtxpro6000/pull/38

Dateien:

- `python/sglang/srt/layers/attention/qsa/sparse_attn.py`
- `python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py`

Der Patch ersetzt mehrere Gather-Operationen durch einen fusionierten Triton-Kernel.

Flag:

`SGLANG_OPT_FUSED_QSA_PREFILL_KV`

**Achtung:** Der Originalpatch behandelt insbesondere BF16/FP8-Staging. Unser NVFP4-KV-Layout besitzt gepackte Daten und Block-Scales.

Nicht blind kopieren.

Implementiere nur eine Variante, die das NVFP4-Layout einschließlich Scales korrekt behandelt, oder nutze einen schnellen, kontrollierten Konvertierungspfad.

Messe Kernel-Launches, Workspace-Verbrauch, Prefill-Leistung und Decode-Interferenz.

## P0-002C: Draft-Step-Längen wiederverwenden

**Pennyroyal PR #43**

https://github.com/jpezzulli/sglang-rtxpro6000/pull/43

Datei:

`python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py`

Die Sequenzlängen mehrerer QSA-Draft-Schritte werden vorbereitet und wiederverwendet, statt sie pro Schritt unnötig neu zu berechnen.

Auf unserem aktuellen Code prüfen, portieren und besonders bei C1/C2 testen.

Testreferenz:

`test/registered/unit/layers/attention/test_qsa_draft_step_seq_lens.py`

## P0-002D: Adaptive Speculation ohne synchrones Feedback

**Pennyroyal PR #55**

https://github.com/jpezzulli/sglang-rtxpro6000/pull/55

Relevante Dateien:

- `speculative/eagle_worker_v2.py`
- `speculative/adaptive_runtime_state.py`
- `speculative/adaptive_confidence.py`
- `managers/scheduler_components/batch_result_processor.py`

Der Patch ordnet Acceptance-Feedback dem richtigen Forward zu und vermeidet falsches Feedback bei überlappender Ausführung.

Prüfen, ob unser adaptiver Draft-Controller dasselbe bereits korrekt tut.

Wenn nicht, portieren.

**Keine zusätzliche GPU→CPU-Synchronisation im Draft-Hot-Path einführen.**

## P1-002E: SM120 FP8 Low-Row Output Head

**Pennyroyal PR #33**

https://github.com/jpezzulli/sglang-rtxpro6000/pull/33

Dateien:

- `python/sglang/kernels/ops/gemm/sm120_online_fp8.py`
- `python/sglang/kernels/ops/gemm/sm120_w8a16_gemv.py`

Prüfen, ob diese Optimierung bereits durch QFN-0159 bis QFN-0169 oder die HyperBurst-Ports abgedeckt ist.

Nicht doppelte GEMV-Implementierungen dauerhaft pflegen.

Einen Gewinner pro tatsächlicher Shape- und Dtype-Klasse auswählen.

## P1-002F: HiCache-Korrektheit

**Pennyroyal PR #75**

https://github.com/jpezzulli/sglang-rtxpro6000/pull/75

Datei:

`python/sglang/srt/mem_cache/qsa_kv_pool.py`

QSA compressed keys dürfen erst gelesen werden, wenn der jeweilige Layer-Restore abgeschlossen ist.

**Diesen Korrektheitstest in unseren NVFP4-HiCache-Pfad übernehmen**, falls er noch nicht äquivalent vorhanden ist.

Nicht den gesamten Scheduler blockieren. Nur der konkrete Datenkonsument muss auf seine erforderlichen Daten warten.

## P1-002G: NIXL O_DIRECT für Mamba-State

**Pennyroyal PR #73**

https://github.com/jpezzulli/sglang-rtxpro6000/pull/73

Datei:

`python/sglang/srt/mem_cache/pool_host/mamba.py`

Dieser Patch richtet Host-Slot-Strides auf 4096-Byte-Grenzen aus, damit O_DIRECT für hybride Mamba-Daten nicht wegen Fehlalignment auf zusätzliche Bounce-Buffer zurückfällt.

Mit aktuellem Upstream vergleichen.

Nur übernehmen, wenn die Alignments im aktuellen Layout fehlen.

---

# 6. TASK P0-003: NVFP4-KV-Attention-Kernel verbessern

Unser Fork enthält bereits Lesilva-basierten NVFP4-KV-Support.

Referenz:

https://github.com/Lesilva/qwen38-flashnext-nvfp4-kv

Prüfe besonders die aktuelle Overlay-Implementierung:

`overlay/pennyroyal-3.1.5/nvfp4-kv/python/sglang/srt/layers/attention/qsa/nvfp4_kv.py`

## Zusätzliche Quellen

### FlashInfer PR #6804

https://github.com/flashinfer-ai/flashinfer/pull/6804

Status: merged.

Implementiert QSA Sparse Paged Attention für NVFP4.

### vLLM PR #64199

https://github.com/vllm-project/vllm/pull/64199

Status bei Recherche: offen.

Betroffene Dateien:

- `vllm/models/qwen4_exp/nvidia/qsa.py`
- `vllm/models/qwen4_exp/nvidia/qsa_indexer.py`
- `vllm/model_executor/warmup/qwen4_exp_qsa_warmup.py`

Benutzt die FlashInfer-NVFP4-QSA-API.

**Wichtiger Build-Hinweis:** Die experimentellen Kernelquellen sind laut PR nicht vollständig in den regulären FlashInfer-PyPI-Wheels enthalten. Prüfe Nightly- oder Source-Build, bevor du den Kernel als nicht verfügbar bewertest.

### vLLM PR #54846

https://github.com/vllm-project/vllm/pull/54846

Alternative Triton-Implementierung für FP8/NVFP4-QSA. Explizit auf SM120 entwickelt.

## Verbindlicher Vergleich

Teste drei Implementierungsfamilien:

1. Unser aktueller Lesilva-/Speedy-Kernel.
2. FlashInfer QSA NVFP4 nach #6804/#64199.
3. Triton QSA NVFP4 nach #54846.

Dabei jeweils:

- K/V-Packing.
- E2M1-Datenlayout.
- E4M3-Block-Scales.
- Globale KV-Scales.
- Indexer-Dtype.
- QSA Decode.
- Prefill.
- CUDA Graphs.
- Speculative Verification.
- HiCache-Backup/Restore.
- Speicherverbrauch.

**Entscheidend:** NVFP4-KV soll direkt im Attention-Kernel konsumiert werden, wenn dies technisch effizient möglich ist.

Vermeide vollständige BF16-Expansion des KV-Caches.

Vergleiche auch vorhandene FP8- und BF16-Scratch-Strategien aus QFN-0155 bis QFN-0165.

Wähle den schnellsten korrekten SM120-Pfad anhand echter Messungen.

---

# 7. TASK P0-004: vLLM-Fusionen nach SGLang portieren

## P0-004A: QSA Prepare Fusion

**vLLM PR #57097**

https://github.com/vllm-project/vllm/pull/57097

Status: merged.

Relevante Dateien:

- `vllm/models/qwen4_exp/nvidia/ops/qsa_prepare.py`
- `vllm/models/qwen4_exp/nvidia/qsa.py`
- `vllm/models/qwen4_exp/nvidia/indexer_qsa.py`

Fusioniert:

- Q/K-Norm.
- RoPE.
- QSA-Indexer-Vorbereitung.
- Gate.
- KV-Cache-Write.

Dadurch können zwei Kernel-Launches und Zwischenmaterialisierung entfallen.

**Aufgabe:** Architektur auf unseren SGLang-QSA-Pfad übertragen, sofern Upstream das nicht bereits äquivalent implementiert.

NVFP4-KV inklusive Scales muss unterstützt werden.

Microbenchmark bei C1/C2/C4/C6 und vollständige Modelltests.

## P0-004B: Projektionen zusammenführen

**vLLM PR #59533**

https://github.com/vllm-project/vllm/pull/59533

Status: merged.

Relevante Dateien:

- `vllm/models/qwen4_exp/nvidia/model.py`
- `vllm/models/qwen4_exp/nvidia/qsa.py`
- `vllm/models/qwen4_exp/nvidia/low_latency_gemm.py`

Führt QSA-QKVG und Indexer-QK-Projektionen in einen GEMM zusammen.

Veröffentlichte Kernel-Gewinne bei kleinen Batches auf getesteten anderen Architekturen: ungefähr 1,45–1,68x.

**SM120 wurde damit nicht automatisch validiert.**

Prüfe:
- ModelOpt-NVFP4-/MXFP8-Kompatibilität.
- Gewichtslayout.
- Quantisierungsscales.
- LoRA-/Fallback-Pfade.
- GEMM-Routing für kleine M.
- CUDA-Graph-Capture.

Wenn die Fusion auf SM120 nicht schneller ist, nicht übernehmen.

---

# 8. TASK P0-005: GPU-RESIDENT-FIRST SCHEDULER

**Das ist eine neue Kernfunktion unseres Forks.**

## Ausgangspunkt im SGLang-Upstream

`python/sglang/srt/managers/schedule_policy.py`

Relevante Funktionen:

- `SchedulePolicy.calc_priority`
- `_compute_prefix_matches`
- `_sort_by_longest_prefix`
- `poll_storage_prefetches`

Weitere Dateien:

- `python/sglang/srt/mem_cache/common.py`
- `python/sglang/srt/managers/schedule_batch.py`
- `python/sglang/srt/mem_cache/base_prefix_cache.py`
- `python/sglang/srt/mem_cache/unified_radix_cache.py`
- `python/sglang/srt/mem_cache/buffer_mode/pipeline.py`

SGLang kennt bereits:

- `MatchResult.device_prefix_len`
- `req.prefix_len`
- `req.num_matched_prefix_tokens`
- Host-/Storage-Hit-Metadaten
- `staged_storage_prefix_end`

Upstream `lpm` sortiert nach `num_matched_prefix_tokens`.

**Das ist für uns nicht ausreichend**, weil ein langer SSD-Hit teurer sein kann als ein kürzerer bereits GPU-residenter Hit.

## Neue Policy

Implementiere beispielsweise:

`--schedule-policy gpu-resident-first`

Der endgültige Name ist frei, muss aber dokumentiert und getestet werden.

### Auswahlalgorithmus

Bei freien Compute-Ressourcen:

1. Aktualisiere den gültigen Cache-Match aller relevanten wartenden Requests.
2. Berücksichtige QSA-, Mamba-, SWA- und Draft-Zustand, nicht nur nackte Token-IDs.
3. Prüfe die tatsächliche Residency.
4. Prüfe, ob der Request mit ausreichendem KV-, Prefill- und Speculation-Headroom aufgenommen werden kann.
5. Sortiere die zulassungsfähigen Requests nach den folgenden Klassen.

### Klasse 1: VRAM-resident

Ein Request kann seinen benötigten wiederverwendbaren Präfix ohne L2-/L3-Restore nutzen.

- Bevorzuge große verwertbare GPU-Präfixe.
- Berücksichtige neue Prefill-Tokens.
- Berücksichtige benötigten zusätzlichen KV-Speicher.
- Berücksichtige Mamba- und Speculation-Headroom.

**Nicht allein anhand der Session-ID entscheiden.**

### Klasse 2: Host-RAM-resident

Ein gültiger, vollständig wiederherstellbarer L2-Hit ist bereits im Host-RAM vorhanden.

Nutze diese Klasse nur, wenn wirklich ein persistenter L2-Cache vorhanden ist.

**Wichtig:** In `buffer_only` ist der Host-Speicher nur Staging. Ein noch nicht geladener SSD-Hit darf nicht fälschlich als RAM-Hit bewertet werden.

Ein bereits abgeschlossener und gültiger Storage-Prefetch im Staging kann separat als „restore-ready“ klassifiziert werden.

### Klasse 3: SSD/NIXL

KV-Präfix liegt nur in L3.

Bevorzuge hier grundsätzlich den am längsten wartenden zulassungsfähigen Request.

Innerhalb der Klasse darf eine einfache Kostenschätzung weitere Optimierungen erlauben, aber Fairness hat Vorrang.

### Wenn eine Klasse nicht zulassungsfähig ist

Falls ein GPU-residenter Request nicht mit sicherem KV-/Scratch-Headroom ausgeführt werden kann, blockiere nicht die gesamte Queue.

Prüfe den nächsten zulassungsfähigen Kandidaten.

Keinen Request starten, dessen KV- oder Hybrid-State-Reservierung nicht sicher möglich ist.

### Fairness

Implementiere Aging beziehungsweise eine konfigurierbare maximale Überholzeit.

Vorschlag für erste Tests:

- Aging-Schwelle 2 Sekunden.
- Aging-Schwelle 5 Sekunden.
- Aging-Schwelle 10 Sekunden.

Die Schwelle ist experimentell.

Ein SSD-residenter Request darf nicht unbegrenzt von schnellen VRAM-Hits überholt werden.

### Ganz wichtig: Keine künstlichen Sammelpausen

Wenn nur ein Request wartet und Compute frei ist, starten.

Nicht 100–200 ms auf einen hypothetischen besseren Request warten.

Die neue Policy wirkt auf die tatsächlich vorhandene Queue.

### Bestehende SGLang-Policies zum Vergleich

- `fcfs`
- `lpm`
- `uniboost`
- `shortest-prefill-first`
- `gpu-resident-first`

Prüfe die tatsächlichen Upstream-Implementierungen.

Die neue Policy muss mit dem UnifiedRadixCache und nach Möglichkeit dem Rust-Tree-Core funktionieren.

### Benchmarks

Teste 10 wartende Sessions:

- 3 GPU-resident.
- 2 Host-resident, sofern L2 aktiviert.
- 5 SSD-resident.

Zusätzlich:

- Alle GPU-resident.
- Alle SSD-resident.
- Gemischte partielle Präfixe.
- Mehrere Agenten mit gemeinsamem Präfix.
- 20 wartende Requests als Stressprofil.
- Alternde SSD-Requests während ständig neue VRAM-Hits eintreffen.

Messe:

- Tatsächliche GPU-Idle-Zeit.
- Bytes SSD→RAM und RAM→GPU.
- Anzahl vermiedener Restores.
- Agenten-Turn-Latenz.
- P95/P99 Queue-Wait.
- Starvation.
- Aggregierte tok/s.

**Abnahme:** Die neue Policy muss gegenüber FCFS/LPM bei wiederverwendeten warmen Sessions einen nachvollziehbaren Vorteil nachweisen, ohne bei Cold-Only-Last relevante Regressionen zu erzeugen.

---

# 9. TASK P0-006: HiCache darf aktives Decoding nicht blockieren

Upstream-Komponenten:

- `python/sglang/srt/managers/cache_controller.py`
- `python/sglang/srt/mem_cache/buffer_mode/pipeline.py`
- `python/sglang/srt/mem_cache/unified_radix_cache.py`
- `python/sglang/srt/mem_cache/storage/nixl/`

## Konkrete externe Patches

### SGLang PR #47621

https://github.com/sgl-project/sglang/pull/47621

Zum Recherchezeitpunkt offen.

Dateien:

- `mem_cache/buffer_mode/pipeline.py`
- `managers/cache_controller.py`
- `mem_cache/hybrid_cache/hybrid_cache_controller.py`
- `mem_cache/unified_radix_cache.py`

Behandelt zu große SSD-Hits, die nicht in einen einzigen `buffer_only`-Staging-Transfer passen.

**Diesen Patch untersuchen und bei Bedarf portieren.**

Besonders wichtig für unsere 200K–500K-Agentenpräfixe und begrenzten Host-Staging-Buffer.

Prüfe korrekte Mamba-/SWA-Recovery-Grenzen.

### SGLang PR #50236

https://github.com/sgl-project/sglang/pull/50236

Zum Recherchezeitpunkt offen.

Korrigiert die Begrenzung für durch Storage-Prefetch geschützte GPU-Anker.

Der Fehler kann zu langem Stillstand führen, wenn die Scheduler-Kapazität durch gesperrte Blöcke unbenutzbar wird.

Prüfe den Fix und übernehme ihn, falls nicht vorhanden.

### SGLang PR #48421

https://github.com/sgl-project/sglang/pull/48421

Merged.

Verhindert unzulässige partielle L3-Restores bei Hybrid-State.

Verifiziere seine Präsenz in unserem Fork.

### Relevante Fehlerberichte

- https://github.com/sgl-project/sglang/issues/45989
- https://github.com/sgl-project/sglang/issues/50790

Auch PR #49158 prüfen:

https://github.com/sgl-project/sglang/pull/49158

Er adressiert `write_back`. Für unseren aktuellen `buffer_only + write_through`-Standard ist dieser spezielle Fix möglicherweise nicht erforderlich.

Nicht unreflektiert in unseren Standardpfad portieren.

## Invarianten

**A. Laufende Decode-Requests warten niemals synchron auf SSD-I/O.**

**B. Ein Restore-Request wird erst admitted, wenn seine benötigten Daten bereit und seine GPU-Ressourcen sicher reservierbar sind.**

**C. Keine KV-Blöcke freigeben, solange die letzte gültige Kopie noch von einem unvollständigen Transfer abhängt.**

**D. Kein globales `torch.cuda.synchronize()` als bequeme Transferbarriere im kritischen Decode-Pfad.**

**E. Hintergrundtransfers dürfen nicht unbegrenzt GPU-Copy-Engines und USB4 beanspruchen.**

**F. Kein Scheduler-Spin, OOM-Crash oder Deadlock bei vollem KV-Pool.**

## Backups abgeschlossener Requests

Prüfe, ob ein abgeschlossener Request seinen GPU-residenten Präfix zeitnah asynchron auf L3 sichern kann, ohne ihn sofort zu verdrängen.

Gewünschte Semantik:

- Request beendet Generation.
- KV bleibt im VRAM.
- Bei geeigneter Gelegenheit beginnt asynchrones Backup.
- Währenddessen können andere Requests decodieren.
- Nach erfolgreichem Backup bleibt der KV weiterhin warm.
- Erst unter VRAM-Druck erfolgt Eviction.
- Existiert bereits eine gültige L3-Kopie, keine redundante Speicherung.

**Achtung:** Zusätzliche Backup-Locks können selbst VRAM-Verdrängung blockieren. Prüfe deshalb die Auswirkungen auf `full_evictable_size`, Anchor-Locks und Admission.

Falls Upstream `write_through` diese Funktion bereits bietet, nicht doppelt implementieren.

## Restore

Ein Request mit SSD-Hit soll:

- Asynchron Storage lesen.
- Begrenzte Host-Staging-Puffer nutzen.
- Nach Möglichkeit Transfer- und Lesephasen überlappen.
- Erst nach gültigem vollständigem erforderlichem Restore admitted werden.
- Während des Wartens keinen Compute-Slot unnötig belegen.

Die bereits laufenden C1–C4-Decodes müssen weiterlaufen.

Benchmarke 1, 2 und 5 GiB Restore-Payloads.

---

# 10. TASK P0-007: NIXL gegen File-HiCache testen

Unsere RC2-Startkonfiguration verwendet:

`--hicache-storage-backend file`

Das ist **nicht NIXL**.

Deshalb zwei separate Profile erstellen.

## Profil A: File

Aktueller optimierter File-Backend-Pfad einschließlich vorhandener Zero-Copy-Patches.

## Profil B: NIXL POSIX

Upstream-Dokumentation:

https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/mem_cache/storage/nixl

Prüfe insbesondere:

- `--hicache-storage-backend nixl`
- POSIX-Plugin
- `use_uring`
- `SGLANG_HICACHE_NIXL_USE_DIRECT_IO`
- O_DIRECT-Alignment
- `page_first` / kompatible Layouts
- Mamba-State-Transfer
- NVFP4-Block-Scale-Sidecars
- Transferregistrierung
- Pinned-Host-Puffer

O_DIRECT ist in aktuellem Upstream für NIXL standardmäßig aktiviert, kann aber bei inkompatiblen Pfaden auf Buffered-I/O zurückfallen.

Das muss im Log überprüft werden.

**Kein GPUDirect-Storage-Speedup annehmen.** Das ist eine TB4-eGPU.

### Vergleich

- Storage-Lesegeschwindigkeit.
- SSD→GPU-Restore-Zeit.
- D2H-Backup-Zeit.
- CPU-Overhead.
- RAM-Verbrauch.
- Staging-Peak.
- P95/P99 ITL aktiver Decodes während Transfers.
- Vollständigkeit von QSA- und Mamba-Sidecars.
- Restore nach Container-Neustart.

**Nicht NIXL nur deshalb wählen, weil es moderner klingt.**

Das schnellere und stabilere Backend gewinnt.

---

# 11. TASK P0-008: Speculative Decoding für C1–C6 optimieren

Unsere Ausgangsbasis:

- NEXTN/MTP.
- FR-Spec.
- Dynamische Tokenmap.
- Adaptive Draft-Konfiguration.
- Bestehende Speedy-Patches QFN-0148 bis QFN-0150.
- Pending Draft-Vocabulary-Buckets QFN-0170 bis QFN-0192.

## Pennyroyal-Referenz

https://github.com/jpezzulli/sglang-rtxpro6000/pull/118

Maintained-Pennyroyal-Profil:

- C1: Draft-/Verify-Block 8.
- C2+: Block 4.

Zusätzlich das ältere adaptive Profil testen.

Prüfe insbesondere, ob unser selbstlernendes Token-Vokabular mit den neuen Draft-Head-Buckets tatsächlich besser arbeitet als eine feste 49K-Tokenmap.

## Verbindliche Tests

- C1: W4, W6, W8, W12, optional W16.
- C2: W4, W6, W8.
- C3/C4: W4, W6, W8.
- C6: W4, W6.
- C8/C10: nur ergänzend.

Metriken:

- Accepted Tokens pro Verify.
- Verify-Zeit.
- Draft-Zeit.
- Draft-Head-Zeit.
- MTP-KV-VRAM.
- CUDA-Graph-Speicher.
- Tok/s.
- Output-Qualität.

## DFlash2

Pennyroyal Flash-Next DFlash2 ist noch ein Alpha-Profil.

Teste es nur, wenn ausreichend Host-RAM vorhanden ist und eine konkrete Leistungsverbesserung plausibel erscheint.

Ein zusätzlicher Draft-Cache mit mehreren zehn GiB ist auf unserem 64-GB-Inference-Host problematisch.

Kein großes Draft-Modell dauerhaft aktivieren, wenn NEXTN bei ähnlichem Durchsatz wesentlich weniger RAM benötigt.

## Tencent Hy4 DFly2

https://huggingface.co/tencent/Hy4-flash

DFly2 ist kein direkt kompatibler Qwen-Drafter.

Analysiere das Prinzip des KV-/Index-Sharings als P2-Forschung.

Keine sofortige Qwen-Portierung ohne passende trainierte Draft-Gewichte.

---

# 12. TASK P0-009: PLE vereinfachen

Unser bevorzugter Standard:

**NVIDIA FP4-Local + gepacktes NVFP4-PLE + pinned Host-RAM.**

Prüfe:

- QFN-0093 bis QFN-0096.
- Pinned-Memory-Allokation.
- Host-RAM-Fragmentierung.
- PLE-Lookup-Kernel.
- Per-Token-Transferkosten.
- Eventuelle doppelte Tabellenkopien.
- Speicherlecks bei Reload und Neustart.

**Keine PLE-SSD-Offloading-Experimente mehr.**

Stop-Liste:

- PLE-Streaming von SSD.
- PLE-Seitenwechsel über SSD.
- Große neue PLE-Offload-Cache-Systeme.
- Komplexe SSD-Prefetch-Heuristiken für PLE.

Zulässig:

- Besseres NVFP4-Packing.
- Schnellere Lookup-Kernel.
- Zero-Copy- beziehungsweise bessere DMA-Übergabe, soweit tatsächlich unterstützt.
- RAM-Einsparungen ohne nennenswerten Durchsatzverlust.

---

# 13. TASK P0-010: Benchmark-Harness für echte Agentenlast

Baue einen reproduzierbaren Benchmark, der die tatsächliche Nutzung simuliert.

Nicht nur lange, gleichzeitige Generierungen.

## Agentenprofile

| Profil | Agenten inklusive Subagenten | Hauptziel |
|---|---:|---|
| A1 | 1 | Maximale Single-Stream-Geschwindigkeit |
| A2 | 2 | Schnelle parallele Turns |
| A3 | 3 | Übergang zu C3 |
| A4 | 4 | C4-Durchsatz |
| A6 | 6 | **Wichtigstes produktives Mehragentenprofil** |
| A8 | 8 | Erweiterte Last |
| A10 | 10 | Stress und Stabilität |

Unter jedem Agentenprofil variiere die GPU-Grenze zwischen C2, C3, C4, C6 und bei Bedarf C8.

## Workload

- 50K-, 150K-, 250K- und 500K-Kontexte.
- Häufig kurze neue User-/Tool-Suffixe.
- Generationen von 32, 128, 512 und 2K Tokens.
- Tool-Pausen von 0,1 s, 0,5 s, 2 s, 5 s und 20 s.
- Wiederverwendete Präfixe.
- Gemeinsame Präfixe mehrerer Subagenten.
- Queue-Tiefen 0, 2, 4, 8 und 16.
- Cold-, Warm- und Mixed-Cache.

## Metriken

- Aggregierte Decode tok/s.
- Decode tok/s pro Request.
- Prefill tok/s.
- TTFT p50/p95/p99.
- ITL p50/p95/p99.
- GPU-Leerlaufzeit.
- Agenten-Turns pro Minute.
- Queue-Wartezeit.
- GPU-residente Cache-Hits.
- Host-Cache-Hits.
- SSD-Restores pro Minute.
- SSD-Restore-Bytes pro Minute.
- KV-Verdrängungen.
- VRAM-Hot-KV-Kapazität.
- Mamba-State-Reclaims.
- RAM-Belegung.
- CPU-Auslastung.
- GPU-Performance während paralleler HiCache-Transfers.

## Messmethodik

- Identischer Checkpoint.
- Identische Sampling-Einstellungen.
- Identische Prompts.
- Identische Kontextlängen.
- Identische Power-Limits.
- Identische Tool-Delay-Traces.
- Warmup vor Messung.
- Alternierende A/B-Durchläufe.
- Mehrere Wiederholungen.
- CUDA-Graph- und Eager-Mikrobenchmarks getrennt.

Ein gemessener Vorteil innerhalb der normalen Streuung ist kein bestätigter Speedup.

Die End-to-End-Messung hat Vorrang gegenüber isolierten Kernel-Erfolgen.

## Referenzdaten aus unserem Projekt

RC2 dokumentiert:

- 30 Stunden echten Agentenbetrieb.
- 35.720 Requests.
- 2,31 Milliarden Prompt-Tokens.
- Ungefähr 94,9 % Prefix-Cache-Nutzung.
- P95-TTFT ungefähr 9,4 s.
- P99-TTFT ungefähr 30 s.

Verifiziere die Rohdaten.

**Die hohen P95/P99-Werte müssen in der neuen Optimierungskampagne ausdrücklich adressiert werden.**

---

# 14. TASK P0-011: `luni-ai-operator` aktualisieren

Repository:

https://github.com/thomasdenk79-cyber/luni-ai-operator

Standardbranch laut GitHub: `master`.

## Zuerst den Operator reparieren

Datei:

`bin/luni-operator.sh`

### Fehler 1: Task-Pfad nach Claim

Der Operator macht derzeit sinngemäß:

`mv "$task" TASKS/work/`

Danach lesen die Agent-Runner weiterhin:

`cat "$task"`

Das ist der alte Pfad.

**Fix:** Nach dem Move den neuen `work/`-Pfad als verbindlichen Task-Pfad verwenden.

Unit-/Integrationstest: Der Agent erhält exakt den Inhalt der geclaimten Task-Datei.

### Fehler 2: Task-Metadaten

Das Task-Format definiert:

- `role`
- `lane`
- `budget`
- `report`
- `gate`

Überprüfe, ob alle Felder tatsächlich validiert und verwendet werden.

Der aktuelle Runner verwendet das `report:`-Feld nicht konsequent als Zielpfad; `budget:` wird nicht als echte Runtime-Begrenzung durchgesetzt.

Beides korrigieren.

### Fehler 3: Busy-Check

Prüfe die aktuelle Logik, die beim Vorhandensein einer aktiven OMO-Sitzung weitere Tasks blockiert.

Wir wollen standardmäßig 1–6 Agenten einschließlich Subagenten unterstützen.

Der Operator muss unterscheiden zwischen:

- Bereits aktiven Agenten.
- Aktiven GPU-Decodes.
- Wartenden Inferenzrequests.
- Freier KV-Kapazität.
- Tatsächlich verfügbarem Task-Budget.

Nicht pauschal jede neue Aufgabe blockieren, nur weil irgendeine OMO-Sitzung existiert.

Trotzdem dürfen keine unkontrollierten GPU-heavy Benchmarks parallel starten.

### Fehler 4: Queue-Head-of-Line-Blocking

Der Operator betrachtet derzeit im Wesentlichen den ältesten Task.

Wenn dessen `gate: sglang-quiet` nicht erfüllt ist, bleiben auch andere unabhängige Tasks liegen.

Implementiere eine sichere Auswahl des ältesten **zulassungsfähigen** Tasks.

### Fehler 5: GPU-Metriken

Prüfe die tatsächlichen Metriknamen und Endpunkte des aktuellen Speedy-Servers.

Fehlende Metriken dürfen nicht stillschweigend als 0 % KV-Auslastung interpretiert werden.

Eine nicht verfügbare Engine bedeutet nicht „unbegrenzt freie Kapazität“.

### Fehler 6: Sichere Task-Ausführung

- Task-Content nach Claim unveränderlich beziehungsweise konsistent einlesen.
- Kein unbeabsichtigter leerer Prompt.
- Exit-Code korrekt auswerten.
- Zeitüberschreitung korrekt melden.
- Fehlgeschlagene Tasks nicht als erledigt markieren.
- Keine fremden Sessions beenden.
- Keine produktive Engine bei aktiver Arbeit unaufgefordert neu starten.

Für alle Fixes ShellCheck und Integrationstests hinzufügen.

## Vorhandene TASKS überprüfen

In `TASKS/todo/` liegen unter anderem:

### `2026-10-20-sglang-mixed-chunk-ab-test.md`

Der Task stammt aus einem älteren Pennyroyal-Setup.

Überarbeite ihn auf:

- Aktuellen Speedy-Source.
- Aktuellen SGLang-Upstream.
- Unseren echten NVFP4-KV-Pfad.
- Neue C1–C10-Agentenmatrix.
- GPU-resident-first Scheduling.
- Aktuelles HiCache-Verhalten.

Prüfe, ob `--enable-mixed-chunk` noch existiert und mit unserem Speculation-Pfad kompatibel ist.

Keine alten Flags ungeprüft weiterreichen.

### `2026-10-20-mamba-pool-groesser.md`

Bisher basiert der Task auf älteren Mamba-Pool-Werten und einer Pennyroyal-Konfiguration.

Neu bewerten:

- Aktuelle `extra_buffer_lazy`-Strategie.
- `--max-mamba-cache-size`.
- Mamba-State-Reclaim.
- NVFP4-KV-Pool.
- Speculative Headroom.
- Warm-KV-Retention.
- C1–C6 als Hauptlast.

Nicht einfach einen alten Wert von 160 übernehmen.

VRAM-Kosten messen und auf tatsächliche Reclaims optimieren.

### Übrige Tasks

Die Aufgaben zu OMO-Boundary, OMO-Härtung, Bun, Kibitzer, senpi-grep und xmig gehören nicht automatisch zu dieser GPU-Performancekampagne.

Prüfe ihren aktuellen Status, ihre Abhängigkeiten und veraltete Laufzeitannahmen.

**Nicht löschen oder als erledigt markieren, ohne den tatsächlichen Befund zu prüfen.**

## Neue Tasks anlegen

Erstelle unter `TASKS/todo/` kleine, einzeln ausführbare Task-Dateien:

- `2026-11-21-speedy-baseline.md`
- `2026-11-21-speedy-hyperburst-gemv.md`
- `2026-11-21-speedy-hyperburst-pdl.md`
- `2026-11-21-speedy-pennyroyal-qsa162.md`
- `2026-11-21-speedy-pennyroyal-draft43-55.md`
- `2026-11-21-speedy-nvfp4-flashinfer.md`
- `2026-11-21-speedy-vllm-qsa-fusions.md`
- `2026-11-21-speedy-gpu-resident-scheduler.md`
- `2026-11-21-speedy-hicache-async.md`
- `2026-11-21-speedy-hicache-nixl-ab.md`
- `2026-11-21-speedy-agentload-c1-c10.md`
- `2026-11-21-speedy-quality-regression.md`
- `2026-11-21-speedy-release-integration.md`

Die Dateinamen sind eine Vorgabe für die neue Kampagne; vor Erstellung auf Kollisionen prüfen.

Jede Datei benötigt:

- Korrekte Rolle.
- Korrekte Lane.
- Realistisches Budget.
- Report-Pfad.
- Abhängigkeiten.
- Konkrete Quelldateien und PRs.
- Arbeitsschritte.
- Tests.
- Abnahmekriterien.
- Rollback.
- Klare Aussage, ob GPU-Exklusivität nötig ist.

Nicht mehrere GPU-heavy Benchmark-Tasks parallel ausführen.

Der Operator soll Codeanalysen und Reviews auf dem leistungsfähigeren Entwicklungsrechner ausführen können, während der T15 weiter inferiert.

Wenn die vorhandene Infrastruktur diese Trennung nicht unterstützt, erst die Ausführungsumgebung prüfen und dokumentieren.

---

# 15. Integration, Qualität und Sicherheitsregeln

Für jede Portierung:

1. Source-PR und Commit-SHA dokumentieren.
2. Merge-Status prüfen.
3. Upstream-Duplikat ausschließen.
4. Lizenz und Attribution erhalten.
5. Isolierten Branch erstellen.
6. Kleine Tests ausführen.
7. GPU-Kernel-Tests ausführen.
8. Modellkorrektheit prüfen.
9. A/B-Benchmark durchführen.
10. Erst danach in den Integrationsbranch übernehmen.

Keine gleichzeitigen unkontrollierten Änderungen an Scheduler, HiCache und Attention-Kernel.

## Qualitätsprüfungen

- Token- und Logit-Vergleich gegen Referenzpfade.
- NVFP4-Quantisierungsfehler.
- Needle Retrieval bei langen Kontexten.
- Codegenerierung.
- Tool-Call-JSON.
- Structured Outputs.
- Reasoning-Stabilität.
- Restore vor/nach Eviction.
- Restore nach Neustart.
- Gemeinsam genutzte Prefixes.
- Mamba-State-Korrektheit.
- Draft-Acceptance und Target-Verifikation.

Ein schneller Kernel mit falschem KV-Layout ist kein Optimierungserfolg.

## Performance-Freigabe

Unterscheide:

- Bestätigter Gewinn.
- Messrauschen.
- Qualitätsregression.
- VRAM-Gewinn.
- RAM-Gewinn.
- Startup-Gewinn.
- Noch unvalidiert.

Eine neue Funktion darf nicht allein aufgrund eines einzelnen positiven Benchmarklaufs standardmäßig aktiviert werden.

---

# 16. Erwartete Artefakte

Erstelle im Speedy-Repository:

`docs/perf-2026-11-21/`

Darin:

- `00-SOURCE-INVENTORY.md`
- `01-UPSTREAM-AND-FORK-DIFF.md`
- `02-PATCH-PORT-MATRIX.md`
- `03-NVFP4-KERNEL-COMPARISON.md`
- `04-HICACHE-NIXL-AUDIT.md`
- `05-GPU-RESIDENT-SCHEDULER-DESIGN.md`
- `06-BENCHMARK-PLAN.md`
- `07-QUALITY-GATES.md`
- `08-INTEGRATION-ROADMAP.md`
- `09-OPERATOR-TASK-MIGRATION.md`
- `RESULTS.md`

Die Patch-Matrix muss für jeden konkret genannten PR enthalten:

- Source-Repository.
- PR-Nummer.
- Commit-SHA.
- Geänderte Dateien.
- Merge-Status.
- Bereits bei uns vorhanden: ja/nein/teilweise.
- Konkreter Integrationsbranch.
- Erwarteter Vorteil.
- Tatsächlicher Benchmark.
- Qualitätsstatus.
- Entscheidung: PORT, KEEP, REPLACE, DROP oder HOLD.

**Keine erfundenen Benchmarks oder unbestätigten Merge-Zustände.**

---

# 17. Implementierungsreihenfolge

## Phase A: Grundlage

1. `luni-ai-operator`-Runner reparieren.
2. Bestehende Tasks aktualisieren.
3. Source- und Patch-Inventar erstellen.
4. Reproduzierbare Speedy- und HyperBurst-Baselines sichern.

## Phase B: Geschwindigkeit

5. HyperBurst QFN-0193 bis QFN-0214 GPU-validieren.
6. Pennyroyal #162 auf Vollständigkeit prüfen und gegebenenfalls portieren.
7. Pennyroyal #43 und #55 prüfen.
8. QSA-Fusionen aus vLLM #57097 und #59533 evaluieren.
9. C1–C6 vergleichen.

## Phase C: NVFP4-KV und HiCache

10. Lesilva-Kernel gegen FlashInfer #6804 und vLLM #64199/#54846 testen.
11. NVFP4-Sidecars und Mamba-Restore vollständig validieren.
12. HiCache-Blockierungsfehler einschließlich #47621/#50236 behandeln.
13. File-Backend gegen NIXL POSIX testen.

## Phase D: Scheduling

14. GPU-resident-first implementieren.
15. Mit FCFS, LPM und UniBoost vergleichen.
16. Deep Queue plus niedrige GPU-Concurrency optimieren.
17. Fairness und Starvation testen.

## Phase E: Produktion

18. Agentenprofile A1–A10 testen.
19. Standardkonfiguration für A1–A6 bestimmen.
20. Langzeittest mit echten OMO-Subagenten ausführen.
21. P95/P99-Latenzen auswerten.
22. Dokumentation, Task-Berichte und Patchregister aktualisieren.
23. Nur getestete Änderungen in den Produktionsbranch übernehmen.

---

# 18. Was ausdrücklich NICHT getan werden soll

- Kein neues PLE-SSD-Offloading.
- Keine weitere PLE-Streaming-Architektur.
- Kein kompletter Fork-Wechsel zurück zu HyperBurst.
- Keine pauschale Übernahme sämtlicher Pennyroyal-Patches.
- Keine doppelten Kernels für dieselbe Shape-Klasse.
- Keine neuen Cache-Ebenen ohne messbaren Nutzen.
- Keine blockierenden SSD-Wartevorgänge in laufenden Decode-Requests.
- Kein FIFO als einzige Queue-Policy.
- Keine dauerhaft erhöhte Decode-Concurrency nur für größere Benchmarkzahlen.
- Keine unkontrollierte GPU-Überlast durch parallele Benchmarks.
- Keine approximative Optimierung ohne Qualitätstest.
- Keine erneute Erstellung bereits vorhandener QFN-Patches.

---

# 19. Abschließende Entscheidungsregel

**Optimierungsziel:**

> Maximale praktische Agentenleistung für 1–6 Agenten, skalierbar auf 10 Agenten und Subagenten, mit HyperBurst-ähnlicher oder höherer Decode-Geschwindigkeit, großem NVFP4-Hot-KV-Pool und nicht blockierendem HiCache.

Es gewinnt nicht automatisch:

- der geringste RAM-Verbrauch,
- die höchste aktive Concurrency,
- der größte SSD-Cache,
- die höchste einzelne Kernel-Beschleunigung.

Es gewinnt die Konfiguration, mit der unsere Agenten ihre Arbeit am schnellsten, stabilsten und korrektesten abschließen.

**Starte mit den konkreten Repositories, PRs und vorhandenen QFN-Patches. Überarbeite `luni-ai-operator` und dessen Tasks, erstelle die Aufgaben und arbeite die priorisierte Implementierungs- und Benchmarkkampagne systematisch ab.**