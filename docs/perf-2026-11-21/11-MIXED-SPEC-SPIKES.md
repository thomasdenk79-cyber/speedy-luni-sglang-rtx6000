# 11 — Mixed+Spec Spikes (22.11., zwei Agenten, read-only)

Contents:
- [Root-Cause SGLang](#root-cause-sglang-warum-kein-spec-im-mischschritt)
- [Fremde Engines](#fremde-engines-gemeinsames-design)
- [Beschlüsse](#beschluss-und-naechste-schritte)

## Root-Cause SGLang: warum kein Spec im Mischschritt

Kausalkette (alle Belege Container qwen38-rc2):

1. **Dispatch ist batchweit.** `schedule_batch.py:2759-2780` setzt die
   gesamte Batch-Mode auf `MIXED`; `eagle_worker_v2.py:1561-1580` waehlt
   die Worker-Route aber pro BATCH (`is_extend()` ->
   `_forward_prefill_batch`, Verify nur im `else`-Zweig `:1692-1718`);
   der Eager-Runner rewritten MIXED->EXTEND (`eager_runner.py:195-209`).
   Es gibt keine per-Row-Moduswahl.
2. **Was per-Row-Verify braeuchte:**
   - Attention-Metadaten sind extend-geformt
     (`forward_batch_info.py:941-944,1196-1222`); Schwanz-Rows haben
     1-Token-Extends statt 1+8-Query-Geometrie.
   - Verify-Baum wird nur fuer alle Zeilen oder keine gebaut
     (`eagle_worker_v2.py:1810-1874` trivial, `:1597-1626` Draft);
     Mischpfad deponiert Bonus-Token nur in `future_map`
     (`schedule_batch.py:2778-2780`).
   - Logit-Slicing folgt Extend-Semantik
     (`forward_batch_info.py:661-681`) vs. Verify-Geometrie
     (`step_span_utils.py:80-116`).
   - CUDA-Graphen: Capture nur fuer reine Decode-/VERIFY-Formen
     (`decode_cuda_graph_runner.py:386-410`, `base_runner.py:523-545`),
     keine Erfassung fuer Mischform.
   - **Mamba:** Mischpfad deaktiviert Tracking fuer Decode-Schwaenze
     explizit (`schedule_batch.py:2813-2844`); Verify braucht
     Checkpoint/Rollback des State — fuer GDN unbelegt.
3. "Verified resume path" (validation_hook.py:360) = der 1-Token-
   Commited-Resume, **kein** Hinweis auf existierendes per-Row-Verify.

## Fremde Engines: gemeinsames Design

- **vLLM** (4388bec): Scheduler reserviert `1+K+lookahead` pro Decode-
  Row im Budget (`v1/core/sched/scheduler.py:374-380,988-1003`);
  Runner baut `cu_num_logits` pro Row (`v1/worker/gpu/model_runner.py:
  1492-1530`), Prefill-Row = 1 Logit, Decode-Row = 1+k; Rejection-
  Sampler ueber Row-Grenzen (Triton). Zeilentyp ist eine explizite
  Invariante (Padding-Guard `:1380-1396`). CUDA-Graph: gemischte
  Batches laufen piecewise/eager — volle Graphen nur bei einheitlicher
  Decode-Form (`config/compilation.py:708-723,1608-1625`).
- **TensorRT-LLM**: Issue #11302 (AutoDeploy) zeigt dieselbe
  Zeilentyp-Diskriminierung als Knackpunkt
  (context-vs-spec-Row Verwechslung `input_pos > 0`).
- **TokenSpeed** existiert: github.com/lightseekorg/tokenspeed
  (MIT, ~2,9k Stars) — Spec-/Chunked-Verhalten ungeprueft.
- Gemeinsamer Kern: flacher Token-Tensor + `query_start_loc`-Offsets,
  Zeilentyp (prefill|decode|decode+K), Logit-Bereiche pro Zeile,
  Verify nach dem Forward pro Row. Preis: CUDA-Graph-Verlust auf
  Mischschritten.

## Beschluss und naechste Schritte

1. **Option A zuerst** (`mixed_above_bs`): Mischscheduler nur ab
   Schwelle, dort ohne Spekulation (die bei hohem c ohnehin saettigt —
   Betreiberdaten c5+ flach). 10–30 LOC, Scheduler-only, default-off.
2. A messen im gpuq-Fenster (c1 vs c5–8, TTFT/ITL/accept-len), bevor
   Option B (per-Row-Verify-Port, >1000 LOC, P1-Risiken GDN-State +
   Graph-Abdeckung) überhaupt in Erwägung geht.
3. Beweispflicht gegen Erfundenheit: LOC-Schaetzung erst validieren
   (Anzahl `is_mixed()`-Aufrufstellen zaehlen), ITL-Kosten erst messen.

Offene Unverifizierte Punkte: vLLM-Piecewise-Claim pro Step loggen;
TRT-LLM-Hauptruntime-Unterstuetzung (PR zu #11302 lesen); SGLang
Adaptive-Breite fuer MIXED (Batch-Size = Gesamtrows? — vor A-Test
klären, `eagle_worker_v2.py:478`).
