# 20 — Pinned benchmark plan (campaign 2026-11-21)

One card that makes a number re-obtainable: same tool, same prompts, same sampling, same
power ceiling, same warmup, alternating arms, three repeats, raw JSON kept, spread stated.
Anything that varies must be a field in the row, not a memory. Pinned by task
`2026-11-21-speedy-baseline`, 2026-11-22; every value below was read off the installed tool
or the queue, commands in `00-SOURCE-INVENTORY.md`.

## Contents

- Pins
- What each number means (field contract)
- Power contract
- Arms and validity gates
- Warmup, order, repeats
- Noise gates
- Feasibility (measured budget)
- Acceptance for a row in `RESULTS.md`

## Pins

| Item | Pinned value |
|---|---|
| tool | BetterBench **0.6.0** (`__version__ = "0.6.0"`, `RESULTS_SCHEMA = 2`, `CORPUS_VERSION = "1.0"`); campaign runs `/srv/community-repos/BetterBench-wt-v060/.venv/bin/betterbench`. Corroborated by the uv tool at `/home/z000g9hu/.local/bin/betterbench`: identical 13-file `*.py` tree (`b68a6d3d3395b4e2`) |
| tool source | `/srv/community-repos/BetterBench` @ `v0.6.0` = `d00ad5ec8098c06584a88ec3468bacd37d5ed098`, built into the campaign worktree `/srv/community-repos/BetterBench-wt-v060` (clean checkout of that commit; the source repo's dirty files touch neither `betterbench/`, `corpus/` nor `config/`) |
| prompts | shipped corpus `v1`: chat 3, code 4, file_edit 3, json 4, math 4, prose 4, reasoning 4, summarization 3 = **29 prompts** in 8 categories, per-prompt `max_tokens` 120–900; the tool's own `content_hash` over `{category: [ids]}` = **`e332dceff176d033`** (== `env.corpus_hash` of the donor artifact → same prompts as 369.9/710.2; measured 2026-11-22 with `betterbench.corpus.load_corpus` + `betterbench.env.content_hash`), file-tree `sha256` over the 8 `*.jsonl` bytes = `b31e9141ad5cf1ea29f62612afc528c96649cbb0dd1504926461cfcd75a6beed` (reproduce: `find corpus/v1 -name '*.jsonl' | sort | xargs cat | sha256sum`). Passed explicitly as `--corpus /srv/community-repos/BetterBench-wt-v060/corpus/v1` |
| config | `bench/campaign-20261121.json`, `sha256` = `ef946cf2cd674d08bbc9c2ca2219e282f75873c86b8b6c5d371a874c3dd9a3a9`; loaded through `Config.load()` it differs from the bundled `config/default.json` in exactly three fields: `greedy true`, `seed 20261121`, `concurrency_levels [1,2,4,6]`. `Config.load()` ignores unknown keys silently, so the validator compares every card key against the echoed `config` block |
| sampling | greedy (`temperature=0`), `seed 20261121`, `top_p 0.95`, `top_k 20`, thinking/reasoning channel left at server default and **recorded per arm** (donor ran thinking on) |
| passes | `runs_per_category 20`, `warmup 3` per category, `unique_nonce true` |
| concurrency | `concurrency_levels [1,2,4,6]` — campaign C1/C2/C4/C6 — with `concurrency_requests 48` (48 divisible by every level: 48/1, 48/2, 48/4, 48/6 waves) |
| prefill | `prefill_depths [2000, 8000, 16000, 32000, 64000]`, `prefill_runs 8`, `prefill_warmup 2`, `prefill_max_tokens 16`, `prefill_ctx_margin 256` |
| engine | one server per arm, launched by `gpuq` from the arm's pinned SHA in a dedicated worktree; production args of `qwen38-rc2` (see 00) unchanged unless the arm's whole point is one flag, which then becomes a named second experiment, never a silent edit |
| power | 300 W enforced limit, read back and stored — next section |
| host | `t15`, RTX PRO 6000 Blackwell 96 GB, single GPU, `tp=ep=dp=1` |

Do not confuse BetterBench versions: **0.5.0 prefill numbers are not comparable to 0.6.0.**
The 0.6.0 changelog is explicit — before 0.6.0 the prefill filler was one paragraph repeated,
whose 16-token blocks had only 25 distinct values, so a content-keyed per-layer LRU served the
sweep from cache and reported cache lookups as prompt processing. That cache is *in our
production build* (QFN-0080 per-layer LRU). Any prefill number without a tool version is
worthless; a 0.5.0 prefill number is not a regression baseline, it was the wrong measurement.

## What each number means (field contract)

A campaign row must say which field it is. The queue's parser
(`qwen38-flashnext-nvfp4-kv/scripts/gpu-queue/gpuq_parsers.py`) defines the canonical set;
the names below are what goes into `results.jsonl` and into `RESULTS.md`.

| Campaign name | JSON path | Definition |
|---|---|---|
| **C1 decode** | `concurrency[level=1].per_req_tps_med` | **median** of per-request `decode_tps` at level 1 |
| **C2 total** | `concurrency[level=2].aggregate_tps` | completion tokens / wall seconds for the level |
| **C4 total** | `concurrency[level=4].aggregate_tps` | idem |
| **C6 total** | `concurrency[level=6].aggregate_tps` | idem |
| TTFT p50 c=1 | `concurrency[level=1].ttft_p50_ms` | median of level-1 `ttft_ms` |
| tail | `concurrency[level].per_req_tps_p1`, `ttft_p99_ms` | 1st percentile rate, 99th percentile TTFT |
| prefill @D | `prefill[depth].pp_tps_med` (+ `pp_tps_low1`, `pp_tps_p99`) | median of `pp_tps`, cold by construction |
| single-stream quality | `single[category].decode_med`, `decode_weighted_tps`, `tok_per_update_med` | per-category medians and the weight-averaged combined score |

**Median, not mean.** The queue reports medians. At c=1 the donor's only raw artifact
(`results/betterbench-1007/results.json`) has median **355.9** and mean **373.7** — 5 % apart,
larger than most effects the campaign is chasing. A number quoted as "c=1 decode" without the
field name is ambiguous by ±5 %, so quote the path.

**`aggregate_tps` is not `per_req_tps_med × level`.** In that same artifact, level 4 has
aggregate 310.2 while per-request median is 180.4 — the aggregate is the box's throughput and
is what "C4 = 710.2 tok/s" refers to.

## Power contract

`power.limit` is 300 W now; nobody sets it from Git (00 §GPU and power state), so an arm must
pin it, prove it, and archive the proof.

1. Before launch: `nvidia-smi -pl 300` — or, since `-pl` needs root and the queue runs
   unprivileged, assert the current value and **abort the arm if it is not 300 W**:
   `nvidia-smi --query-gpu=power.limit,power.default_limit,power.draw,temperature.gpu,clocks.sm,utilization.gpu --format=csv,noheader,nounits`
   → `300.00, 600.00, 299.99, …` (measured 2026-11-22 01:29 CET). `power.limit` is the
   software-enforced cap, i.e. "Enforced Power Limit"; **`power.enforced.power.limit` is not a
   valid query field on driver 620.70.06 and kills the whole query with rc=2** (measured), which
   is why the campaign step reads `power.limit` and reads `power.default_limit` alongside.
2. During: the queue sampler already writes `metrics.power_max_w` / `power_mean_w` per job
   (observed across 47 rows: max 319.7 W, no row above 350 W). Keep those two numbers in the row.
3. After: read `power.limit` again; if it moved, the row is void.
4. Cross-check against Prometheus series `nvidia_smi_enforced_power_limit_watts{job="gpu",
   instance="llm-gpu-exporter:9835",gpu="rtx-pro-6000"}` for the run's wall-clock bounds —
   the queue writes `replay-start.epoch` / `replay-end.epoch` style bounds for exactly this.
5. Compare arms only at the same cap. A 600 W row against a 300 W row is not an A/B.

The campaign's Nov-18 reference table claims a 350 W cap that this host has never shown
(600 W until 2026-11-18 14:51 CET, 300 W since; two transient 450 W samples on 2026-11-17).
Until that row's condition is re-measured it is **historical, condition unknown** — see
`RESULTS.md` and the follow-up task.

## Arms and validity gates

Four arms, each with its own server start, config and label. `sglang_head` is read
**per arm** from the container's `Label: ai.hyperburst.sglang_head`; the production
head (`a090492091`, `/get_server_info`) and the git-ref head of the pinned worktree
`/srv/community-repos/sglang-wt-tagrc2` (`1ad48f14e1`, five commits behind) differ, so
"rc2" has to be pinned by SHA in every row, not by name.

| Arm | What changes vs production | HiCache env (`name=value`) | Status |
|---|---|---|---|
| **A** reference | nothing — production args as they run | `SGLANG_HICACHE_STORAGE_BACKEND=`, `SGLANG_HICACHE_STORAGE_MODE=`, `SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE=` (all empty) | planned |
| **B** candidate | the fork branch under test | empty | planned |
| **C** HiCache | HiCache on the file backend | `SGLANG_HICACHE_STORAGE_BACKEND=file`, `SGLANG_HICACHE_STORAGE_MODE=per_page`, `SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE=40G` | planned |
| **D** prefill | chunk 16384 **and** `max_prefill_tokens` 16384 | empty | planned |

The values in the C row are the campaign's choice, not a fact about the box: the
**running** engine has none of them set (`podman inspect qwen38-rc2` → the three
`HiCache` keys are empty, `ChunkedPrefillSize=8192`, `MaxPrefillTokens=8192`,
`EnableMixedChunk=False`). A launch env saying otherwise is not the truth of a run —
`bench/bb-campaign.sh` therefore reads `/get_server_info` before and after the pass and
stores it as `server-info.json`; if it cannot, the run is not publishable.

**The A/B/C run of 2026-11-21 17:46:17Z is invalid as a comparison** (`INVALID ARMS:
A/B/C cross-arm mixing` in its own ledger row: one server, one start, three arms, one
pass each; `INVALID: no prefill depth isolation` — the tool ran its fixed depth list and
the step asked for none; `INVALID: no minimum repeat count`). It also ran with
`chunked_prefill_size=16384` against an 8192 production contract. Reuse it as context
only (`21-DONOR-RECORD.md` §5), never as a baseline, and never as a donor comparison.

Gates, all enforced mechanically in `bench/bb-campaign.sh` (rc=1 on any miss, verified
2026-11-22 with the matrix in `30-RUN-COMMANDS.md` §3.2):

1. every key of the card exists in `betterbench.config.Config.__dataclass_fields__`, and
   each card value reappears in the `config` block the tool echoes into `results.json` —
   `Config.load()` drops unknown keys **silently**, so a typo would otherwise look like
   a measured greedy run while sampling stayed at `temperature 0.7`;
2. `env.betterbench_version`, `env.corpus_hash`, `env.num_prompts` present and equal to
   the pins; missing version is a refusal (a 0.5.0 prefill number is the wrong
   measurement, see the warning above);
3. the arm note carries `arm=<name>`, and the `name` ends in the arm — this is what makes
   a ledger row attributable;
4. `quick` not set, and each level in `concurrency_levels` has a non-empty per-request
   rate list plus a non-null `aggregate_tps` — level 6 is the one most likely to come
   back empty, since the donor ran 1/2/4/8 and the queue's parser only maps those four
   (`flatten_betterbench`, follow-up task 662);
5. dump completeness: `server-info.json` present, `results.json` parseable, a prefill
   entry for every requested depth;
6. power read before and after against the expected cap (measured now: `power.limit`
   `300.00`, `power.default_limit` `600.00`; `--cap=350` refuses to run).

**One server per arm, runs sequential.** A second server would need another ~95 GiB of
the 97.9 GiB card, so the queue's `sequential: true` is the only possible shape;
alternating order (`A B A B A B`) is what the plan asks for and `campaign-20261121.json`
records it.

**Cold prefill vs cached prefix — two different quantities.** The 8K-chunk run with
HiCache on re-reads a long prefix that is already resident, which is a *re-use* number;
the 16k/32k numbers are cold by construction (`prefill_runs` with fresh fillers). The
script writes the cold reference as its own artifact (`prefill_cold_reference: true`) and
`RESULTS.md` keeps the two in separate tables; a row that mixes them is a defect, not a
result.

**Donor caveats to carry into every comparison:** the donor's `4 concurrent total` is
BetterBench level 4 aggregate, the donor's levels were 1/2/4/8 (no level 6 anywhere),
and the donor's 32 K prefill row (12,796 tok/s) disagrees with the only raw artifact on
this host (11,802 `pp_tps_med` at depth 32000) — see `21-DONOR-RECORD.md` §1, §3.

## Warmup, order, repeats

- **Engine warm per arm:** CUDA graphs and Triton JIT are built at startup, so the queue's
  `startup_s` (median 293 s, max 393 s over 47 rows) already covers that; additionally the
  tool discards `warmup 3` per category and `prefill_warmup 2` per depth. Do not reduce these.
- **One server start per arm** (donor rule: repeated measurements after restart are *not*
  independent; differences of a few percent are run-to-run noise). Repeats within one start
  are the tool's own passes; repeats across starts are separate jobs with `repeat=` in the
  job name.
- **Alternating arms:** order `A B A B A B` (never `A A A B B B`), so thermal and background
  drift splits evenly. Record the realized order in the row's `order` field.
- **Repeats:** ≥ 3 per arm per point. State `min / median / max` and the relative spread
  `(max-min)/median` next to every quoted number. Anything inside the spread is **not** a win;
  the queue's `target_mde_pct 1.0` is the effect size the design is allowed to claim.
- **A/B mechanics:** `betterbench ab` needs two live endpoints; one server already holds
  95.5 GiB of 97.9 GiB, so both do not fit at `mem_fraction_static 0.94`. Arms therefore run as
  separate `gpuq` jobs (the queue stops production, runs the arm, hands back, restores), and
  the pair is evaluated offline with `betterbench compare a/results.json b/results.json`.

## Noise gates (all must hold, per arm)

- no other agent session on the box, no `qt-*` tmux session (the queue's idle gate);
- engine idle before start: `/v1/loads` shows `num_running_reqs = 0` and `num_waiting_reqs = 0`;
- screen off / no desktop redraw (donor condition; the donor's own runner pinned
  `SERVE_DISPLAY_HZ=60` and recorded "also drives a 4K desktop" in the note — a desktop on the
  same GPU costs a few percent and must be declared, not hidden);
- one run at a time on the GPU (`flock` semantics are the queue's own);
- disk ≥ 80 GB on `/srv/llm-cache`, free memory ≥ 81920 MiB at hand-back (queue gates);
- nothing else writing the HiCache storage tree during the run.

## Feasibility (measured, not guessed)

From the donor artifact and 47 queue rows: measured request-seconds per full arm ≈ 182 s
(single stream, 160 runs) + 255 s (concurrency, 4 levels × 48) + 62 s (prefill TTFT, 40 runs)
= **~8.5 min**; the warmup passes add roughly a third; server start adds ~5 min; queue
stop/hand-back/restore adds ~5 min.

**One arm ≈ 20 min ⇒ two arms × 3 repeats ≈ 2 h.** A full C1/C2/C4/C6 + prefill A/B does fit
inside one `00:00–07:00` window with margin; the cold-prefill extension experiment (30) is a
second job in the same window.

## Acceptance for a row in `RESULTS.md`

A row is only admissible with:

1. `arm` = branch + `sglang_head` SHA **from the queue row**, never from a worktree name;
2. `image` + `image_digest` as recorded by the queue;
3. the four campaign numbers with the JSON field path and `n`;
4. `min/median/max` and relative spread over ≥ 3 repeats;
5. power: `enforced_limit_watts` read before and after, plus `power_max_w` / `power_mean_w`;
6. label: `measured-local` / `donor` / `historical` / `not-measured` (README rule);
7. raw artifact path: the queue's `dir` under `scripts/gpu-queue/results/<stamp>-<job>/`
   containing `results.json`, the server log, and the `/get_server_info` echo. Raw JSON is
   the evidence; a Markdown number without it is a rumour.
8. order of the run within the alternation.

If any of these is missing, the row stays in the ledger as `UNPRUEFBAR` with the reason —
never smoothed over, and never quoted in a release note.
