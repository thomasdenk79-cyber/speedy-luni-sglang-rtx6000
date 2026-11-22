# 30 — Run commands: the GPU half of campaign 2026-11-21

Everything here runs on t15. **Two hard rules decide who types what:**

1. `gpuq add` and `gpuq run` are the **owner's** job — the queue stops the
   production container `qwen38-rc2`, so a run is a production intervention
   (agent limit "`gpuq validate` only", recorded in
   `reports/report-2026-11-21-speedy-model-catalog.md`; the job README that
   should state it, `docs/gpuq-plan/jobs/README.md`, does not exist — see
   `00-SOURCE-INVENTORY.md`). This task's approval
   (`engine-hold-lift-rc2-2026-11-21` in `TASKS/APPROVALS.md`) says
   *GPU frei, kein Lease, keine gpuq-Runner*, so **no run was made here**; §1–§4
   are staged, tested and ready for the owner's window.
2. There is **no agent-held GPU lease**: `state/gpu-window/lease.json` is absent and
   `bin/gpu_window.py active` returns rc=1 (no live lease); `status` prints `no gpu window`
   (note: `status` always exits 0, so the **go/no-go signal is `active`**, not `status`). A baseline
   number produced outside the queue and the window is not a baseline.

## Contents
- §1 Pin table (what must be true before any run)
- §2 Pre-flight checks
- §3 Staging and validation (agent-allowed)
- §4 Running the window (owner only)
- §5 Artifacts, ledger rows, and what counts as a baseline
- §6 Abort rules

## 1. Pin table

| Object | Pinned value | Evidence |
|---|---|---|
| Reference source | `rc/qwen38-rc2` = `a090492091c7ed527d7f48faf35e9dec71e0f034` | `env -u GIT_INDEX_FILE git -C /srv/community-repos/sglang-wt-speedybase rev-parse HEAD` |
| Reference worktree | `/srv/community-repos/sglang-wt-speedybase` (private to this task, clean) | `git status --porcelain` empty |
| Container image | `docker.io/lmsysorg/sglang:nightly-dev-cu13-20261117-25c1c439` | job `IMAGE`, rule R9 also compares image **IDs** against `qwen38-rc2` |
| Bench tool | BetterBench `0.6.0` = tag `v0.6.0` (`d00ad5ec8098c06584a88ec3468bacd37d5ed098`), clean checkout, own venv at `/srv/community-repos/BetterBench-wt-v060/.venv/bin/betterbench` (sha256 `22ffad0b4024859a1192d19a46a11adda7f25717bab22eb23c3fa76b1df68470`) | `$BB --version` → `BetterBench 0.6.0`; `CORPUS_VERSION 1.0`, `RESULTS_SCHEMA 2`; `git status --porcelain` empty |
| Bench config | `bench/campaign-20261121.json`, sha256 `ef946cf2cd674d08bbc9c2ca2219e282f75873c86b8b6c5d371a874c3dd9a3a9` | asserted by the step before every run |
| Corpus | shipped `corpus/v1` (8 categories, **29 prompts**), file-tree sha256 `b31e9141ad5cf1ea29f62612afc528c96649cbb0dd1504926461cfcd75a6beed`, tool `content_hash` `e332dceff176d033` | tool hash == donor `env.corpus_hash`, see `21-DONOR-RECORD.md` §5; per-file hashes in `20-BENCH-PLAN.md` §3 |
| Ladder | c = 1, 2, 4, 6; greedy; seed 20261121; 48 requests/level (29 corpus prompts + 19 warmup-upfront copies, v0.6.0 `phase_warmup`); 5 prefill depths | `Config.load()` echo in `results.json` |
| Power | `nvidia-smi` enforced power limit asserted per run (`--cap=300`) | see §5 "power", and `21-DONOR-RECORD.md` for why this is a gate |
| Repeats | 3 per arm (`-r1/-r2/-r3`, distinct seeds; the pinned bench seed is constant on purpose) | job files in `bench/jobs/` |

## 2. Pre-flight checks

Run all of them; each prints a value that has to match. Anything non-matching is
a finding to report, not a warning to skip.

```bash
# 2.1 the queue itself, its window and the disk floor
cd /srv/community-repos/qwen38-flashnext-nvfp4-kv
scripts/gpu-queue/gpuq status
scripts/gpu-queue/gpuq disk                       # >= 80 GB free on /srv/llm-cache

# 2.2 production must be the reference build, and idle
env -u GIT_INDEX_FILE git -C /srv/community-repos/sglang-wt-rc2 rev-parse HEAD
curl -s -m 10 http://127.0.0.1:30000/get_server_info | python3 -c \
  'import json,sys; d=json.load(sys.stdin); print({k:d.get(k) for k in ("chunked_prefill_size","max_prefill_tokens","enable_mixed_chunk","max_running_requests","prefill_decode_interval","attention_backend","kv_cache_dtype","context_length")})'
curl -s -m 10 http://127.0.0.1:30000/v1/loads | head -c 400   # running=0 waiting=0 expected
tmux ls 2>/dev/null | grep -c '^qt-'                          # must be 0 (idle gate)

# 2.3 the launch path must be pinned too — examples/run-upstream.sh is the
#     GPUQ_LAUNCHER and it currently carries uncommitted changes (+128/-17)
env -u GIT_INDEX_FILE git -C /srv/community-repos/qwen38-flashnext-nvfp4-kv \
  status --porcelain -- examples/run-upstream.sh              # expect empty, else stop

# 2.4 the power cap actually set (0 = unlimited; measured history 600 W until
#     2026-11-18 14:51 CET, 300 W after, never 350 W — see 21-DONOR-RECORD.md)
nvidia-smi --query-gpu=power.limit,power.default_limit,power.draw,clocks.sm \
  --format=csv,noheader,nounits        # 300.00, 600.00, <draw>, <clocks> (measured 2026-11-22)

# 2.5 the bench tool and its pinned artifacts
betterbench --version                             # BetterBench 0.6.0
sha256sum docs/perf-2026-11-21/bench/campaign-20261121.json

# 2.6 the reference worktree is still what the task pinned
env -u GIT_INDEX_FILE git -C /srv/community-repos/sglang-wt-speedybase \
  rev-parse HEAD                                  # a090492091c7ed527d7f48faf35e9dec71e0f034
```

## 3. Staging and validation (agent-allowed, no GPU touched)

Job files live with the campaign in
`/srv/community-repos/speedy-luni-sglang-rtx6000/docs/perf-2026-11-21/bench/jobs/`
and follow the `gpuq-plan/jobs/` convention (one arm = one file = one server
start, `REPEAT=1`, unique `NAME`, own `HICACHE_ROOT`).

```bash
cd /srv/community-repos/speedy-luni-sglang-rtx6000
J=docs/perf-2026-11-21/bench/jobs

# 3.1 the queue's own loader: format, step names, ceilings
for f in $J/arm-a-rc2-r*.env $J/arm-c-chunk16k-r1.env; do
  echo "== $f"; /srv/community-repos/qwen38-flashnext-nvfp4-kv/scripts/gpu-queue/gpuq validate "$f"
done

# 3.2 isolation rules R1-R10 plus the live-production diff (dry-run only, no GPU)
cd /srv/community-repos/qwen38-flashnext-nvfp4-kv
python3 gpuq-plan/check_arm.py $J/*.env; echo "rc=$?"
```

Measured on 2026-11-22 (the state the owner inherits):

| File | `gpuq validate` | `check_arm.py` | DIFF vs `qwen38-rc2` |
|---|---|---|---|
| `arm-a-rc2-r1.env` | OK | OK | `only_arm=[] only_prod=[]` |
| `arm-a-rc2-r2.env` | OK | OK | `only_arm=[] only_prod=[]` |
| `arm-a-rc2-r3.env` | OK | OK | `only_arm=[] only_prod=[]` |
| `arm-b-cand-r1.env` | OK | **FAIL R5** (`SGLANG_SRC … sglang-wt-REPLACE-ME`) | template, owner fills the candidate |
| `arm-c-chunk16k-r1.env` | OK | OK | `only_arm=['16384'] only_prod=['8192']` |

Every file also shows `VALDIFF SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE=40G|200G`:
rule R8 forces an arm-sized cache cap, so an arm can never measure against the
production cache namespace. That is intended, not drift.

Step plan per run (`STEPS`):

| # | Step | What it measures |
|---|---|---|
| 1 | `startup` | `startup_s`, container provenance, log signature |
| 2 | `custom:…/bb-campaign.sh:--arm=…:--cap=300:--repeat=N` | decode c1/c2/c4/c6 + prefill ladder, via the pinned BetterBench config; asserts the power cap, archives `/get_server_info` |
| 3 | `custom:scripts/session-replay/step.sh:--sessions=8:--prefix-tokens=20000:--shared-tokens=10000:--turns=40:…:--seed=N` | tier1 session shape; per-turn `prompt_tokens` minus `cached_tokens` is the cold→cached split (needs R7 `--enable-cache-report`) |
| 4 | `custom:scripts/session-replay/step.sh:--sessions=1:--prefix-tokens=180000:--turns=8:…` | the missing measurement of `10-PREFILL-DECODE-HOUR.md` §4: one long session, first turn cold at ~180K, later turns extend a resident prefix |

Step 2 is a custom step because the queue's `bb-*` shortcuts have no `c6` and
force `concurrency_requests == prefill_runs`; the pinned config needs both
(`gpuq_lib.py` `BB_CONC_LEVELS`, `betterbench_setup`).

## 4. Running the window (owner only)

```bash
cd /srv/community-repos/qwen38-flashnext-nvfp4-kv
Q=scripts/gpu-queue/gpuq
$Q status
# queue must be empty of foreign jobs first:
ls scripts/gpu-queue/queue/pending/            # foreign jobs stay where they are

# 4.1 stage this campaign only, in the order of bench/jobs/window-sb.txt
while read -r f; do $Q add "/srv/community-repos/speedy-luni-sglang-rtx6000/$f"; done \
  < docs/perf-2026-11-21/bench/jobs/window-sb.txt    # run from the campaign repo root
$Q list

# 4.2 the proof run: plan every step, no process, no GPU
$Q run --dry-run

# 4.3 start inside the night window (default 00:00-07:00)
$Q run --once --window-only                    # one job, then stop
$Q run --window-only                           # or: the whole staged set, sequentially

# 4.4 watch, from a second shell
$Q status
tail -f /srv/llm-cache/qwen38-flashnext/logs/gpuq-runner.log 2>/dev/null || true
nvidia-smi --query-gpu=power.limit,power.draw,temperature.gpu,clocks.sm \
  --format=csv,noheader,nounits        # power.limit must stay 300.00
```

Budget, measured rather than guessed: `startup_s` median 293 s (max 393 s over
47 rows), the pinned BetterBench pass ≈ 500 s (182 s single-stream + 255 s
concurrency + 62 s prefill), the two replay steps are the long pole at the tier1
shape. `TIMEOUT_S=4200` per job with `STEP_TIMEOUTS="custom=1800"`. Five staged
jobs ≈ 2.5–3 h, which fits the 7 h window with the abort margin the queue
already enforces (`GPUQ_QUIESCE_TIMEOUT_S` 1800 s, hand-back ≥ 81920 MiB free).
Arms run **sequentially** — a second server would need another ~95 GiB of VRAM,
so `betterbench ab` (two live endpoints) is not available on this card; A/B is
compared offline with `betterbench compare` instead, and the alternation that
removes thermal drift is the repeat order r1 → r2 → r3 per arm, not interleaving.

## 5. Artifacts, ledger rows, and what counts as a baseline

Per run, the archive is `scripts/gpu-queue/results/<timestamp>-<NAME>/`:
`job.env` (the exact job), `results.json` (BetterBench, canonical fields),
`server-info.json`, `power-before.txt`, `power-after.txt`, `replay/requests.jsonl`,
`replay/summary.json`, `gpu.csv`, plus the runner's `row.json`.

The one row per run that the ledger reads comes from
`scripts/gpu-queue/results/results.jsonl`; canonical field names are fixed by
`gpuq_parsers.py`:

| Campaign number | Field | Rule |
|---|---|---|
| C1 decode | `decode_weighted_tps` | BetterBench's weighted median over single-stream categories — **not** `concurrency["1"].aggregate_tps`; the donor table was read this way (`docs/updates/2026-10-27.md:139`) |
| C2/C4/C6 total | `concurrency["2"|"4"|"6"].aggregate_tps` | sum of tokens over wall clock |
| C2/C4/C6 tail | `concurrency[…].per_req_tps_p1`, `.ttft_p99_ms` | queueing shows up here first |
| Cold prefill | `prefill[depth].pp_tps` at fresh depths | `unique_nonce`, nothing resident |
| Cached extension | `replay/requests.jsonl`: `prompt_tokens - cached_tokens` per turn | a resident prefix at ~180K vs the fresh 8K-chunk case |
| Power | `power-before.txt` first column | must equal the pinned `--cap` value |

A number enters `RESULTS.md` as **baseline** only if all of these hold; the
`Source` column says which:

```
window opened by gpuq (queue row exists)  +  power limit == pinned value
+  results.json tool/corpus/config echo == pin table  +  3 repeats
+  server-info shows the launch args that really ran  +  arm DIFF == expected
```

Anything measured on somebody else's running server, or with an unverified cap,
or from another machine's table, stays labelled `donor` / `historical` /
`not measured` — see `21-DONOR-RECORD.md` for the donor figures (C1 369.9,
C4 710.2, both author-reported and never reproduced on t15) and the 350 W
claim, which the measured history does not support.

```bash
# 5.1 the queue's own comparison table for the campaign (delta against the
#     reference arm); --list prints the tracked metric keys
Q=/srv/community-repos/qwen38-flashnext-nvfp4-kv/scripts/gpu-queue/gpuq
$Q report exp-sb --baseline exp-sb-a-rc2-r1 --repeats \
   --metrics startup_s,bb_decode_weighted_tps,bb_c1_aggregate_tps,bb_c2_aggregate_tps,bb_c4_aggregate_tps,bb_c1_ttft_p50_ms
$Q report --json exp-sb > /tmp/opencode/sb-rows.json      # machine copy for the ledger

# 5.2 C6 has to be read from results.json: flatten_betterbench only flattens the
#     levels 1/2/4/8 (gpuq-runner.py 1661), so there is no bb_c6_* column in the
#     ledger row. Use the queue's parser so the number is the same one the ledger
#     would have carried, only for level 6.
python3 - <<'PY'
import json, glob, sys
sys.path.insert(0, "/srv/community-repos/qwen38-flashnext-nvfp4-kv/scripts/gpu-queue")
import gpuq_parsers as P
for path in sorted(glob.glob("/srv/community-repos/qwen38-flashnext-nvfp4-kv/scripts/gpu-queue/"
                             "results/*exp-sb*/results.json")):
    d = P.parse_betterbench(path=path)
    c6 = (d.get("concurrency") or {}).get("6") or {}
    print(path.split("/")[-2][:46], "| C6 agg", c6.get("aggregate_tps"),
          "| per-req p1", c6.get("per_req_tps_p1"), "| ttft_p99", c6.get("ttft_p99_ms"))
PY
```

## 6. Abort rules

From `gpuq-plan/window-1-owner.md` §(e), unchanged, because it is the owner's
standing instruction for windows on this card:

- Two consecutive jobs `FAIL` on transport/runtime → `scripts/gpu-queue/gpuq abort`
  (discard the rest; production comes back up).
- Engine traceback or a divergence of production after restore → abort, write an
  incident note, do not restart except through a queue restore.
- `restore_timeout` is counted separately (`restore_timeout_count`,
  `sessions_stopped_by_restore_timeout` in the replay summary) and is **not** a
  global failure — it ends its own session, never the run.
- The step in §2.3/§2.4 that fails before the start is not an abort: it is the
  run not being allowed to begin, and the reason goes into the report.

Rollback for this whole task is documentation-only: the campaign files are new
files in `docs/perf-2026-11-21/`, the shared queue and the shared repos are
untouched, and no server was stopped, started or reconfigured by it.
