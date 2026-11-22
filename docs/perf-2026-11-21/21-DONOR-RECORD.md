# 21 — Donor record: HyperBurst reference numbers (campaign 2026-11-21)

What the donor numbers are, where they came from, what they may be compared against and what
they may never be used for. Every number that enters a campaign row as `donor` must be
traceable to this file.

## Contents

- The numbers
- Provenance and what is published
- Field definition (settled, with a recomputation)
- Power cap: the unresolved part
- Conditions to reproduce
- Verdict

## The numbers

From `hyperburst-rtxpro6000/docs/updates/2026-10-27.md:128-142` (header line 128 verbatim:
"BetterBench (BetterBench 0.6.0, the machine used by aiueo52 and contributors, screen off,
one start per build)"):

| Donor build | c=1 decode | 4 concurrent total | TTFT p50 c=1 | prefill @32k |
|---|---|---|---|---|
| `flash-next-fast` normal | **369.9 tok/s** | **710.2 tok/s** | 101 ms | 12,796 tok/s |
| `flash-next-fast` hyper pro | 342.3 tok/s | 693.7 tok/s | 103 ms | 12,266 tok/s |

Same document, line 142: runs with a **private token map and fine-tuned MTP head** (not
shipped) reached 420.9 (normal) and 423.7 (hyper pro) tok/s at c=1; the public assets were
about 12 % and 19 % slower in those runs.

Line 116 adds that the *comparison ratios* in that update used the private head and token map,
while the absolute speeds are the BetterBench table above — so a ratio and an absolute from the
same document are not from the same asset set. Do not mix them.

Donor hardware is one RTX PRO 6000 Blackwell **Max-Q** 96 GB, which also drives a 4K desktop;
screen off during measurement (line 101, 128). Donor source: `aiueo52/hyperburst-rtxpro6000`
@ `8ff73f4169` (2026-11-12) driving `aiueo52/sglang-rtxpro6000` @ `8edd1fab4b`. Other sm_120
cards are explicitly untested by the donor.

## Provenance and what is published

The donor states its own evidence status in the same document, line 105:

> "The per-genre runs are published in `results/runs-1027/genre/`. The other figures in this
> update are author-reported; their run files are not published in this repository. Besides
> the per-genre runs, `results/` contains the earlier September and 2026-10-02 runs, not the
> run files for the October 27 comparisons, quality checks or BetterBench results."

So: **the raw BetterBench artifact behind 369.9 / 710.2 is not published by the donor and does
not exist on this host.** Verified by searching for any file carrying `betterbench_version`
under `/srv/community-repos`, `/srv/luni-repos` and the user's tool directories: exactly one
real BetterBench result exists locally —
`hyperburst-rtxpro6000/results/betterbench-1007/results.json`, a **2026-10-07** run of an older
build (`sglang-prod-1002`, profile `wa`, adaptive `3/7/15`) with `mtp_head=private-finetuned`,
`token_map=private-49k`, `tuned_for=c1-only` recorded in its launch script, plus the queue's
own JSON fixtures. Everything else in `RESULTS.md` terms is `historical`.

Line 175 also warns: "One start per build: treat differences of a few percent as run-to-run
noise. A first hyper pro attempt was stopped when the screen turned on and was repeated from a
fresh server start; its numbers are not used."

## Field definition (settled, with a recomputation)

Donor line 139, verbatim: **"c=1 decode is BetterBench's weighted median over its
single-stream categories."** That is `decode_weighted_tps` in the queue's parser
(`gpuq_parsers.py`, weight-averaged per-category `decode_med` with weights code 0.30,
reasoning 0.20, prose 0.15, json 0.15, file_edit 0.10, summarization 0.10; chat and math carry
weight 0). It is **not** `concurrency[level=1].per_req_tps_med` and not an aggregate.

The definition was checked by running the campaign's own parser over the donor's only raw
artifact:

```
$ python3 -c "import sys; sys.path.insert(0,'/srv/community-repos/qwen38-flashnext-nvfp4-kv/scripts/gpu-queue');
    import gpuq_parsers as gp; print(gp.parse_betterbench_json(
    '/srv/community-repos/hyperburst-rtxpro6000/results/betterbench-1007/results.json'))"
betterbench_version 0.6.0  schema 2  timestamp 2026-10-07T05:01:28+0900
decode_weighted_tps 367.80   ttft_p50_ms 111.83   update_p99_ms 12.33
concurrency[1] aggregate 313.72  per_req_med 355.93  per_req_p1 236.29  ttft_p50 112.24  tok_per_update 3.78  n 48/48
concurrency[4] aggregate 310.22  per_req_med 180.38  ttft_p50 2411.52  n 48/48
```

Readings:

- A **weighted median at c=1 of 367.8** from an independent build and a different owner is the
  same order as the donor's published 369.9 (−0.6 %) and as the campaign's local Nov-18
  reference of 367.2 (−0.2 %). Three numbers from two machines and three builds agreeing to
  well under 1 % is *not* proof they are comparable — it is proof the field is stable enough
  that a 5 % claim needs repeats, and proof that the two local tables and the donor table are
  plausibly the same measurement. That is precisely why the field path must be written down.
- Its c=4 **aggregate is 310.2, not ~700**, and its level-4 TTFT median is **2.4 s**: that build
  queued at c>1 (`tuned_for=c1-only`). Donor line 140 says the same for older builds — only
  `flash-next-fast` serves 4 at once (`CONC=4`). So the local artifact cannot corroborate
  710.2; it corroborates why 710.2 is a build-level property, not a BetterBench property.
- The same artifact's published conditions say `greedy`, and the raw `config` block says
  `greedy false`, `temperature 0.7`, `seed null`. For our arms, greedy + fixed seed is
  mandatory (20-BENCH-PLAN) — an unseeded arm is not repeatable and the interval arithmetic
  in BetterBench's A/B loses its meaning.

## Power cap: the unresolved part

- The donor's BetterBench conditions name BetterBench 0.6.0, screen off, one start per build —
  and **state no power cap**.
- The donor's own lab notes elsewhere record caps of 300 W after reboot and 325 W before reboot
  (`docs/lab-notes/FC_fc-glue_2026-10-01.md:143,197`, `MOE_SMALLM_SPEC.md:372,566`,
  `RS1_REJECTION_SAMPLING.md:107`, `bench/prof/OVERHEAD_REPORT.md:277`), and
  `bench/bench/megakernel/harness.py:15` says the harness runs without `-pl` because it needs
  sudo. The pennyroyal repo documents no cap either.
- Consequence: the cap behind 369.9 / 710.2 / 12,796 is **unknown**. The candidate values the
  donor's own notes support are 300 W or 325 W (or unlimited if the harness truth applies to
  that run), never 350 W.
- `docs/releases/speedy-rc2.md:214` claims a "350 W cap" for the **local** Nov-18 reference.
  This host has never enforced 350 W: the Prometheus series
  `nvidia_smi_enforced_power_limit_watts{job="gpu",gpu="rtx-pro-6000"}` shows 600 W until
  2026-11-18 14:51 CET and 300 W from then on, at 60 s resolution on that day (1441 samples,
  no gap), and all 47 queue rows cap out at ≤ 319.7 W. So the local table's stated condition
  is unsupported too — its real condition was 600 W or 300 W depending on the hour.

## Conditions to reproduce

To produce a row that may sit next to 369.9 / 710.2 at all:

1. BetterBench **0.6.0**, shipped corpus `v1`, hash `b31e9141…a6beed`, pinned config
   `bench/campaign-20261121.json` (20-BENCH-PLAN).
2. **Public** token maps and the **stock** MTP head — the donor's headline is the public-assets
   number; the private-assets family (420.9 / 423.7) is a different row.
3. One server start per arm, screen off / no desktop redraw, `max-running-requests` and
   `chunked-prefill-size` as the production card in 00, thinking state recorded.
4. `concurrency_levels` including 4 for the "4 concurrent total" = `aggregate_tps` at level 4.
5. Power: pin and read back 300 W, keep `power_max_w` / `power_mean_w`; state the cap in the
   row. Comparing a 300 W row to the donor's unknown cap is a **labelled** comparison
   (`donor, cap unknown`), never a claimed parity.
6. ≥ 3 repeats with the field path, `n`, and min/median/max printed.
7. To settle the donor side properly: ask for (or fetch, if published later) the raw
   `results.json` behind the 2026-10-27 BetterBench table and its `power.limit` at run time.
   Until then the numbers are `donor / author-reported`.

## Verdict

| Statement | Status |
|---|---|
| HyperBurst c=1 369.9 tok/s, 4-concurrent 710.2 tok/s | **donor, author-reported**, no raw artifact in this repo or on this host; condition "cap unknown"; usable as a target, never as a baseline |
| c=1 field = BetterBench weighted median over single-stream categories | **settled** by donor line 139 and recomputed from the donor's own raw artifact (`decode_weighted_tps` 367.80) |
| local artifact corroborates 710.2 | **no** — the only local artifact is an older c1-only build that queues at c=4 (aggregate 310.2, TTFT p50 2.4 s) |
| donor ran at 350 W | **unsupported**; donor's own notes point to 300/325 W, and no artifact records the cap |
| local Nov-18 "350 W cap" claim | **unsupported** by Prometheus and by every queue row on this host → follow-up task, see `RESULTS.md` |
