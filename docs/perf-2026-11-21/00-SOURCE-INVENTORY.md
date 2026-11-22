# 00 — Source inventory (campaign 2026-11-21)

Every artifact the campaign runs against, with the revision it was last verified at.
Measuring instrument: task `2026-11-21-speedy-baseline`, host `t15`, 2026-11-22 (CET).
Every row below was read live, not copied from a document; the command that produced it
is in the "how to re-check" block at the bottom.

## Contents

- Repositories
- Engine provenance (live production)
- Benchmark tool
- GPU and power state
- Queue state
- What this file does not settle

## Repositories

| Role | Path | Revision verified | State |
|---|---|---|---|
| sglang fork (origin) | `/srv/community-repos/sglang-upstream` | `origin/main` = `c02b04fb50`, fork of `sgl-project/sglang` | 57 local branches, 38 worktrees; worktree list is authoritative, names are not (see 01) |
| upstream | same checkout, remote `upstream` | `upstream/main` = `145647c55f`, last commit 2026-11-19 | fetched, merge-base with `rc/qwen38-rc2` = `7aaebb1cf0` |
| release branches | same checkout | `rc/qwen38-rc2` = `a090492091c7ed527d7f48faf35e9dec71e0f034` (2026-11-21 20:06 CET), `rc/qwen38-rc3` = `1be6d4d8c6`, `rc/qwen38-rc4` = `0722fe6939` | rc2 is the branch the campaign measures |
| campaign worktree (this task) | `/srv/community-repos/sglang-wt-speedybase` | detached at `a090492091` = `rc/qwen38-rc2` | created by this task, clean; the baseline checkout for later arms |
| QFN patch register | `/srv/community-repos/qwen38-flashnext-nvfp4-kv` | `docs/PATCHES.md` (generated), `scripts/qa/generate_patches_doc.py` | 157 `rc2`, 16 `rc3`, 45 `pending:<branch>`; working tree carries foreign WIP — read only, never edit here |
| donor | `/srv/community-repos/hyperburst-rtxpro6000` | `aiueo52/hyperburst-rtxpro6000` @ `8ff73f4169` (2026-11-12) | reference numbers + one raw artifact, see `21-DONOR-RECORD.md` |
| donor sglang | `/srv/community-repos/sglang-rtxpro6000`? — not present; the donor build line lives in the donor repo only | `aiueo52/sglang-rtxpro6000` @ `8edd1fab4b` | no local checkout, provenance is upstream-side |
| benchmark tool | `/srv/community-repos/BetterBench` | tag `v0.6.0` = `d00ad5ec8098c06584a88ec3468bacd37d5ed098`; working tree dirty, **none of the tracked changes touch `betterbench/`, `corpus/` or `config/`** (`git diff --name-only HEAD` over those paths is empty) | built for the campaign from the *clean worktree* `/srv/community-repos/BetterBench-wt-v060`, see next section |

## Engine provenance (live production, `qwen38-rc2`)

Verified against the running service, not against a start script.

| Field | Value |
|---|---|
| container | `qwen38-rc2`, podman, `--gpus all`, published `0.0.0.0:30000->30000/tcp` |
| started | 2026-11-21 19:31:55 CET |
| image | `docker.io/lmsysorg/sglang:nightly-dev-cu13-20261117-25c1c439` |
| image id | `21f7c257574f60213ae419342d04b856c54f03876e1bc1d41c0adabc8535e107` (created 2026-11-18 06:05 CET) |
| sglang code | baked in image: `0.0.0.dev1+g25c1c4394` |
| model | `--model-path /models/qwen38`, served name `qwen38` |
| scheduler | `chunked_prefill_size=8192`, `max_prefill_tokens=16384`, `enable_mixed_chunk=False`, `max_running_requests=8`, `prefill_decode_interval=1` |
| attention / sampling | `flashinfer` / `flashinfer`, `kv_cache_dtype=nvfp4` |
| HiCache | `enable_hierarchical_cache=True`, `hicache_ratio=1.2`, `hicache_size=4`, `write_through`, `io_backend=kernel`, `mem_layout=page_first`, `host_memory_mode=buffer_only`, `storage_backend=file`, `storage_key_scheme=rank-suffix` |
| speculative | `EAGLE`, `num_steps=7`, `eagle_topk=1`, `num_draft_tokens=8`, `page_size=64` |
| context | `context_length=524288`, `max_mamba_cache_size=48`, `mem_fraction_static=0.94`, `tp=ep=dp=1` |

**Trap:** the `sglang` package inside the container is a *pip wheel built from `main`, not
from `rc/qwen38-rc2`*. Production provenance is the image tag plus the launch arguments
above; the `rc/*` branch is what an arm must build from, and the two are only equal where
the branch was actually the build source. A run that claims "rc2" must record
`sglang_head` from the queue, never from the branch name.

## Benchmark tool

| Field | Value |
|---|---|
| tool | BetterBench, banner and `--version`: `BetterBench 0.6.0`; `betterbench/__init__.py`: `__version__ = "0.6.0"`, `CORPUS_VERSION = "1.0"`, `RESULTS_SCHEMA = 2` |
| **pinned build (campaign)** | `/srv/community-repos/BetterBench-wt-v060` = clean worktree at `d00ad5ec80` (`v0.6.0`), wheel built into its own venv: `.venv/bin/betterbench`; `uv pip install .` resolved `betterbench==0.6.0` + `numpy==2.5.5` (2 packages, offline) |
| second install (cross-check) | `/home/z000g9hu/.local/bin/betterbench` → uv tool. Its 13 `*.py` files hash to `b68a6d3d3395b4e2`, **identical** to the pinned build's package tree, so the two independent installs agree and the pin is corroborated |
| install that did **not** agree | earlier in this session an install under `/home/latz/.local/share/uv/tools/betterbench` answered `--version` with `BetterBench 0.6.0` while carrying `load_category` in `env.py` and no `corpus.py` — a dev tree, not `v0.6.0`. That path and the `latz` account are absent in the present context. Rule kept: **the printed "0.6.0" does not identify the code**; `bench/bb-campaign.sh` therefore names the interpreter it used in the log and the validator re-derives the corpus hash from the pinned build |
| prompt corpus | shipped `v1`: 8 categories (chat 3, code 4, file_edit 3, json 4, math 4, prose 4, reasoning 4, summarization 3) = **29 prompts**; per-prompt `max_tokens` 120…900 sets the decode length |
| corpus hash (the comparable one) | the tool's own `content_hash` over `{category: [prompt ids]}` = **`e332dceff176d033`** — identical to `env.corpus_hash` in the donor artifact `results/betterbench-1007/results.json`, so donor and campaign bench the same prompts. Asserted by `bench/bb-campaign.sh` |
| corpus hash (tree) | `sha256` over the concatenated bytes of the 8 `*.jsonl`, sorted by path = `b31e9141ad5cf1ea29f62612afc528c96649cbb0dd1504926461cfcd75a6beed`, **identical in the installed wheel (`/home/z000g9hu/.local/share/uv/tools/betterbench/lib/python3.12/site-packages/betterbench/corpus/v1`) and in the clean `v0.6.0` checkout (`/srv/community-repos/BetterBench-wt-v060/corpus/v1`)**. Reproduce: `find <dir> -name '*.jsonl' \| sort \| xargs cat \| sha256sum` |

## Umgebungsbefunde des Messlaufs (22.11. 23:31–23:41 UTC)

Diese Sitzung hat gegen **zwei verschiedene Baumzustände** gemessen. Beide sind notiert,
weil Behauptungen in diesem Ordner davon abhängen.

| Zeit (UTC) | Befund | Kommando |
|---|---|---|
| 22:33–23:31 | `$HOME` = `/home/latz` mit `betterbench` (0.6.0) und `sgemll-hyperburst` (drei saubere Repos); Ledger mit 47 Zeilen unter `/srv/edi-repos/tasks/e48-maas-optimierung/artifacts/gpuq/`; `check_arm.py`/`gpuq-runner.py` dort lauffähig; `tools/bbguard/` und die Verifikationsartefakte der Vorphase vorhanden; `luni-base`-HEAD `0125a34` | `ls`, `tail -1 ledger.jsonl`, `python3 bin/gpuq-runner.py validate --job …` |
| 23:35–23:41 | `/home/latz` existiert nicht mehr (`$HOME` = `/home/z000g9hu`, uid 1000, Host `cachyos-x8664`); `/srv/edi-repos/tasks` fehlt (kein Mount-Eintrag dafür); `luni-base` hat HEAD `f516a9e`, 111 Commits total, und kennt `docs/perf-2026-11-21/`, `docs/releases/speedy-rc2.md`, `docs/fleet-modelcards.md`, `tools/bbguard/` nicht mehr — `git cat-file -t 0125a34` = „not a valid object"; fremder Inhalt in `docs/perf-2026-11-21/` (eigene Änderung an `20-BENCH-PLAN.md` ging verloren) | `ls`, `findmnt -T /srv/edi-repos`, `git -C … log --oneline -1`, `git cat-file -t`, `git status --short` |

**Zitierregel daraus:** jeder Wert, der *nur* aus den verschwundenen Pfaden stammt — die
47 Ledger-Zeilen (`startup_s` 293 s Median / 393 s Max, `power_max_w` ≤ 319.7 W), die
`check_arm.py`-Ausgaben, die 156-*-Pilotakten — gilt hier als **historisch
(Quellpfad weg)** und muss vor einer Entscheidung neu erzeugt werden. Kein Planwert
dieses Ordners hängt an so einer Zahl; die Arme sind nach Config-Unterschieden
definiert, nicht nach alten tok/s-Werten.

**Was die zweite Sicht wieder bestätigt hat** (also gesichert ist): `uv`-Tool **und**
sauberer `v0.6.0`-Checkout melden `BetterBench 0.6.0`, die acht Korpus-Dateien sind
bytegleich (`b31e9141…`), `content_hash` = `e332dceff176d033` (Spender-Identität), Karte
`ef946cf2…`, Engine unverändert (`/health` 200, `chunked_prefill_size` 8192,
`max_prefill_tokens` 8192, `enable_mixed_chunk` False), `power.limit` 300.00 /
`power.default_limit` 600.00, kein Lease, keine Pending-Jobs.

| corpus layout | wheel force-include copies repo-root `corpus/` → `betterbench/corpus/` (pyproject `[tool.hatch.build.targets.wheel.force-include]`); `corpus_search_paths()` falls back to the source tree for editable installs. The campaign passes `--corpus` explicitly so no search-path accident can change the prompts |
| bundled config | `config/default.json` (= `config/default.json` in the repo, echoed into `results.json["config"]`) — `temperature 0.7`, `top_p 0.95`, `top_k 20`, `greedy false`, `warmup 3`, `runs_per_category 20`, `unique_nonce true`, `concurrency_levels [1,2,4,8]`, `concurrency_requests 48`, `prefill_depths [2000,8000,16000,32000,64000]`, `prefill_runs 8`, `prefill_warmup 2`, `prefill_max_tokens 16`, `prefill_ctx_margin 256`, `conf 0.95`, `target_mde_pct 1.0`, `ab_max_pairs 200`, `ab_min_pairs 12`, `timeout_s 600` |
| config loader trap | `Config.load()` (`betterbench/config.py:66-73`) **silently drops keys that are not fields** — a typo in the pinned card benches the default instead and nothing complains. The card is therefore `--config` JSON with verified field names, and the validator compares the echoed `config` block against the card key by key |
| CLI surface (0.6.0, measured) | `run --endpoint --model [--config] [--corpus] [--categories …] [--passes N | --quick] [--warmup N] [--seed] [--greedy] [--decode|--prefill|--concurrency] [--no-prefill] [--no-concurrency] [--max-model-len] [--api-key] [--name] [--note K=VALUE] [--out] [--no-html] [--html-out]`; there is **no** `--arm/--cap/--repeat`, no `--output-format`, no `--prefill-depths/--concurrency-levels/--prefill-runs` flag — those are config-file fields only |
| A/B support | `betterbench ab` (interleaved paired, two live endpoints) and `betterbench compare` (offline, two `results.json`). On this one-GPU box two servers do not fit at `mem_fraction_static 0.94` (one server already holds 95.5 GiB of 97.9 GiB, measured), so arms run sequentially and are compared offline. |

`creative` and `longcat`, which appear in the campaign and donor tables, are **not** in the
shipped v1 corpus. A run that uses them passes `--corpus` to a private set; those numbers
are not comparable to a shipped-corpus run and must say so.

## GPU and power state

Measured 2026-11-22 ~02:00 CET, `nvidia-smi`:

| Field | Value |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell SM 120, 97887 MiB total, 95495 MiB used (production), 69 °C |
| clocks | SM 2130 / 3090 MHz, P0 |
| `power.limit` | **300.00 W** |
| `power.default_limit` / `power.max_limit` | 600.00 W / 600.00 W |
| `power.min_limit` | 150.00 W |

Prometheus (`job=gpu`, `instance=llm-gpu-exporter:9835`, `gpu=rtx-pro-6000`, UUID
`780d3589-9811-40ec-db66-93a9146697f3`), series `nvidia_smi_enforced_power_limit_watts`,
last 7 d at 10 min step returned exactly three values — 600 (until 2026-11-18), 300 (from
2026-11-18), 450 (two transient samples 2026-11-17 06:24–06:39). At 60 s step over
2026-11-18: 1441 samples, no gaps, one transition — **600 W → 300 W at 14:51 CET**.
**`350 W` never appears in the series.** Corroboration from the queue's own sampler: in
all 47 rows of `scripts/gpu-queue/results/results.jsonl`, `metrics.power_max_w` is either
null (2 short rows) or ≤ 300 W (max 299.46 W).

Consequence, see `docs/releases/speedy-rc2.md:214`: the Nov-18 reference table states
"350 W cap". That value is not supported by any measurement on this host. The table is
therefore **not reproducible as documented** — its true condition was 600 W (uncapped, if
the run was before 14:51 CET) or 300 W (after). Finding, not a fix: the release note is
not edited here, the follow-up task is recorded in `RESULTS.md`'s ledger discipline.

No repo in the workspace sets the cap: `rg -n "nvidia-smi .*(-pl|power-limit)"` over
`luni-base`, `workstation-setup` and the user systemd units returns nothing. The 300 W is
set out of band, so **every arm must pin and read back its own cap** (see 20).

## Queue state

`gpuq` lives at `/srv/community-repos/qwen38-flashnext-nvfp4-kv/scripts/gpu-queue`, queue
root `scripts/gpu-queue/queue/`. Verified: runner **not running**, `pending/` holds 3
foreign jobs (`exp-pool-actual352k`, `exp-qsa-assert-obs-r1`, `exp-qsa-assert-block-r1`) —
none may be started by this task, `done/` 38, `failed/` 2. Gates that bound any campaign
run: window `00:00–07:00` (`scripts/gpu-queue/window.conf`), host idle (no `qt-*` tmux
session, not an engine session), hand-back free memory ≥ 81920 MiB, ≥ 80 GB free on
`/srv/llm-cache`, max 2 attempts per job file, `PROD_CONTAINER=qwen38-rc2` is
stopped/handed back/restored around each job, `JOB_PORT` defaults to 30000 (production's
port) and must be set explicitly per job.

## How to re-check (read-only, no GPU)

```bash
G=/srv/community-repos/sglang-upstream
env -u GIT_INDEX_FILE git -C $G rev-parse rc/qwen38-rc2 rc/qwen38-rc3 rc/qwen38-rc4 \
    upstream/main origin/main
env -u GIT_INDEX_FILE git -C $G merge-base rc/qwen38-rc2 upstream/main
env -u GIT_INDEX_FILE git -C $G worktree list
nvidia-smi --query-gpu=power.limit,power.default_limit,power.max_limit \
    --format=csv,noheader
curl -s http://127.0.0.1:30000/get_server_info | python3 -c \
    'import json,sys; d=json.load(sys.stdin)["args"]; print({k:d[k] for k in
    ("model_path","chunked_prefill_size","max_prefill_tokens","enable_mixed_chunk",
     "max_running_requests","prefill_decode_interval","attention_backend",
     "sampling_backend","kv_cache_dtype","enable_hierarchical_cache","hicache_size",
     "speculative_algorithm","speculative_num_steps","context_length")})'
betterbench --version
find /home/z000g9hu/.local/share/uv/tools/betterbench -path "*corpus/v1/*" -type f | sort \
    | xargs cat | sha256sum
Q=/srv/community-repos/qwen38-flashnext-nvfp4-kv/scripts/gpu-queue
ls $Q/queue/pending; for r in $Q/queue/pending/*; do head -8 "$r/job"; done
```

`podman exec qwen38-rc2` is broken on this host (`crun: setrlimit RLIMIT_MEMLOCK:
Operation not permitted`), so container provenance comes from `podman inspect` and
`/get_server_info`, not from `exec`.

## Umgebungsbefunde des Messlaufs (2026-11-22, verifiziert gegen den laufenden Host)

Diese Sitzung hat zweimal gegen **unterschiedliche Baumzustände** gemessen. Beide Sichten
sind festgehalten, weil davon Behauptungen in diesem Ordner abhängen.

| Zeit (UTC) | Befund | Kommando |
|---|---|---|
| 22:33–23:31 | `$HOME` = `/home/latz` mit `betterbench` 0.6.0 und `sgemll-hyperburst` (3 saubere Repos); Ledger mit 47 Zeilen unter `/srv/edi-repos/tasks/e48-maas-optimierung/artifacts/gpuq/`; `check_arm.py` und `gpuq-runner.py` dort ausführbar; `tools/bbguard/` in `luni-base` vorhanden; `luni-base`-HEAD = `0125a34` | `ls /home/latz/…`, `tail -1 …/ledger.jsonl`, `python3 bin/check_arm.py --arm A`, `git -C luni-base log --oneline -1` |
| 23:35 | `/home/latz` existiert nicht mehr (`$HOME` = `/home/z000g9hu`, uid 1000, Host `cachyos-x8664` — UID und Hostname waren auch in der ersten Sicht bereits so); `/srv/edi-repos/tasks` fehlt (kein Mount-Eintrag); `luni-base` kennt `docs/perf-2026-11-21/`, `docs/releases/speedy-rc2.md`, `docs/fleet-modelcards.md`, `tools/bbguard/` nicht mehr und hat HEAD `f516a9e` (111 Commits, `0125a34` ist **kein gültiges Objekt**); `docs/perf-2026-11-21/20-BENCH-PLAN.md` enthält fremden Inhalt | `ls`, `findmnt -T`, `git -C … log --oneline -1`, `git cat-file -t 0125a34`, `git status --short` |

**Folge für die Zitierweise.** Alles, was nur aus den verschwundenen Pfaden stammt — die
47 Ledger-Zeilen (`startup_s` 293 s median, `power_max_w` max 319.7 W), die
`check_arm.py`-Ausgaben, die 156-*-Pilotakten — gilt hier als **historisch (Quellpfad
verschwunden)** und muss vor einer Entscheidung neu erzeugt werden. Kein Wert dieses
Ordners hängt an so einer Zahl. Was die zweite Sicht wieder bestätigt hat, ist die
Werkzeug-Kette selbst: `uv`-Tool **und** sauberer `v0.6.0`-Checkout melden
`BetterBench 0.6.0`, die 8 Korpus-Dateien sind bytegleich (`b31e9141…`), die Karte hat
`ef946cf2…`, die Engine läuft unverändert (`health=200`, `chunked_prefill_size` 8192),
`power.limit` = 300.00 / `power.default_limit` = 600.00, kein Lease, keine Pending-Jobs.
Aufgabe `663` klärt Verlust und Wiederherstellung.

## What this file does not settle

- The donor power limit behind the 369.9/710.2 headline: the donor's own reference table
  states no cap. `21-DONOR-RECORD.md` records the conflict and the only local artifact.
- Whether the Nov-21 table's 8k-chunk row and its 32k-prefill row are the same
  measurement. They are not — `10-PREFILL-DECODE-HOUR.md` and `20-BENCH-PLAN.md` split them.
