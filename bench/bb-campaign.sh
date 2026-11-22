#!/usr/bin/env bash
# Campaign 2026-11-21 — BetterBench step at the fixed concurrency ladder 1/2/4/6.
#
# Why a custom step: the queue's built-in bb-* steps map to a shortcut table
# that has no c6 and forces concurrency_requests == prefill_runs. The campaign
# needs 6 in flight plus an independent prefill depth run, so this step calls
# BetterBench directly with the pinned config.
#
# Pinned contract (bench/campaign-20261121.json, sha256 checked below):
#   tool    BetterBench 0.6.0 built from source at the pinned commit
#           /srv/community-repos/BetterBench-wt-v060  git d00ad5ec8098c06584a88ec3468bacd37d5ed098
#           (tag v0.6.0), venv .venv/bin/betterbench; __version__ asserted = 0.6.0
#   config  ef946cf2cd674d08bbc9c2ca2219e282f75873c86b8b6c5d371a874c3dd9a3a9
#   corpus  shipped v1: 8 categories, 29 prompts; the tool's own
#           content_hash over prompt ids = e332dceff176d033 (== the donor artifact's
#           env.corpus_hash, so the corpus is identical to the one behind 369.9/710.2);
#           installed tree sha256 (8 *.jsonl) 25609238ea6badaf9551e9b648f54d1c227f9998fc5c3df9e2a1114428a93254
#   levels  1 2 4 6   greedy=true   seed=20261121   repeat=2
#   tokens  decode length comes from the corpus (per-prompt max_tokens 120..900);
#           the prefill sweep keeps decode tiny at prefill_max_tokens=16.
#           0.6.0 has no max_tokens_total / max_prompt_tokens knob - see 20-BENCH-PLAN.md.
#   power   the enforced cap is asserted before the run; a mismatch aborts
#
# Called by gpuq as:
#   custom:/srv/community-repos/speedy-luni-sglang-rtx6000/docs/perf-2026-11-21/bench/bb-campaign.sh:--arm=arm-a-rc2:--cap=300:--repeat=1
# gpuq exports GPUQ_URL, GPUQ_OUT, GPUQ_NAME, GPUQ_MODEL, GPUQ_CONTAINER,
# GPUQ_SGLANG_SRC, GPUQ_SGLANG_IMAGE and the step inherits the runner's PATH.
#
# Writes into $GPUQ_OUT (archived with the job, parsed by gpuq_parsers):
#   results.json          the canonical bench artifact — fixed name, do not rename
#   server-info.json      /get_server_info: the args the server actually ran with
#   power-before.txt      power.enforced.power.limit, power.draw, temp, sm clock
#   power-after.txt
#   bb-campaign.log       tool stdout
set -uo pipefail

: "${GPUQ_URL:?GPUQ_URL not set (gpuq exports it)}"
: "${GPUQ_OUT:?GPUQ_OUT not set (gpuq exports it)}"

CFG_SRC="/srv/community-repos/speedy-luni-sglang-rtx6000/docs/perf-2026-11-21/bench/campaign-20261121.json"
CFG_SHA="ef946cf2cd674d08bbc9c2ca2219e282f75873c86b8b6c5d371a874c3dd9a3a9"

ARM="unknown-arm"
REPEAT="0"
CAP=""
for tok in "$@"; do
  case "$tok" in
    --arm=*)    ARM="${tok#--arm=}" ;;
    --repeat=*) REPEAT="${tok#--repeat=}" ;;
    --cap=*)    CAP="${tok#--cap=}" ;;
    *) echo "[bb-campaign] FAIL: unknown argument '$tok'" >&2; exit 1 ;;
  esac
done

fail() { echo "[bb-campaign] FAIL: $*" >&2; exit 1; }

# The tool is pinned to a source build, not to whatever `betterbench` the
# interactive user happens to have in ~/.local/bin: two installs on this host
# both reported "0.6.0" while their module layout differed (env.load_category vs
# corpus.load_corpus), so the version string alone does not identify the code.
# Resolution order: $BB_BIN, the pinned venv, PATH, ~/.local/bin - and the tool
# that won is named in the log, together with the git sha of its source tree.
BB="${BB_BIN:-}"
if [ -z "$BB" ]; then
  if [ -x /srv/community-repos/BetterBench-wt-v060/.venv/bin/betterbench ]; then
    BB=/srv/community-repos/BetterBench-wt-v060/.venv/bin/betterbench
  elif command -v betterbench >/dev/null 2>&1; then
    BB="$(command -v betterbench)"
  elif [ -x "${HOME}/.local/bin/betterbench" ]; then
    BB="${HOME}/.local/bin/betterbench"
  else
    fail "betterbench not found (BB_BIN, pinned venv, PATH nor ~/.local/bin/betterbench)"
  fi
fi
[ -x "$BB" ] || fail "betterbench not executable: $BB"
command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi not on PATH"
BB_VERSION="$("$BB" --version 2>&1 | tail -1)" || true
echo "[bb-campaign] tool=$BB version='$BB_VERSION'"
case "$BB_VERSION" in
  *0.6.0*) : ;;
  *) fail "betterbench version is not 0.6.0: '$BB_VERSION' (pinned contract)" ;;
esac
# The validator below imports betterbench itself (to check the echoed config
# against the card), so it must run on the tool's own interpreter, not on the
# system python, which has no betterbench.  uv installs python3 beside the entry
# point; fall back to the system python only when that sibling is missing.
BB_ROOT="$(dirname "$(dirname "$BB")")"
BB_PY="$BB_ROOT/bin/python3"
[ -x "$BB_PY" ] || BB_PY="$(command -v python3)"
[ -f "$CFG_SRC" ] || fail "pinned config missing: $CFG_SRC"
have="$(sha256sum "$CFG_SRC" | awk '{print $1}')"
[ "$have" = "$CFG_SHA" ] || fail "pinned config sha mismatch: have $have want $CFG_SHA"

# Per-run copy: a concurrent job or a late edit can never swap the file under
# the tool between the check here and the run below.
CFG="$GPUQ_OUT/campaign-20261121.json"
cp -f "$CFG_SRC" "$CFG" || fail "cannot copy pinned config into $GPUQ_OUT"

# Corpus passed explicitly: the shipped v1 sits at the repo root and only
# reaches the package through the wheel's force-include, so an editable install
# or a venv with a different layout would silently bench another corpus. 8 *.jsonl
# files, 8 categories, 29 prompts; the content hash is checked after the run
# against the value recorded in 20-BENCH-PLAN.md.
CORPUS_DIR="${BB_CORPUS:-/srv/community-repos/BetterBench-wt-v060/corpus/v1}"
[ -d "$CORPUS_DIR" ] || fail "pinned corpus dir missing: $CORPUS_DIR"
n_jsonl=$(find "$CORPUS_DIR" -maxdepth 1 -name '*.jsonl' | wc -l)
[ "$n_jsonl" -eq 8 ] || fail "corpus $CORPUS_DIR has $n_jsonl *.jsonl, expected 8"

curl -fsS -m 10 "$GPUQ_URL/v1/models" >/dev/null 2>&1 || fail "engine not reachable at $GPUQ_URL"
# Provenance, not measurement: the server's own args are the only evidence that
# the lane under test ran with the flags the job file declares. Retry once; if it
# still fails the run continues (the numbers are good) but the missing file makes
# it unpublished - RESULTS.md must say so and 30-RUN-COMMANDS.md shows the check.
if ! curl -fsS -m 30 --retry 2 "$GPUQ_URL/get_server_info" -o "$GPUQ_OUT/server-info.json"; then
  rm -f "$GPUQ_OUT/server-info.json"
  echo "[bb-campaign] WARN: get_server_info failed - server-info.json missing," \
       "this run is NOT publishable until provenance is recovered" >&2
  date -Is > "$GPUQ_OUT/server-info.error"
fi

record_power() {
  # power.limit is the software-enforced cap nvidia-smi reports ("Requested Power
  # Limit" in -q -d POWER); power.enforced.power.limit is not a query field on
  # driver 620.70.06 and aborts the query with rc=2 - measured here 2026-11-22.
  nvidia-smi --query-gpu=power.limit,power.default_limit,power.draw,temperature.gpu,clocks.sm,utilization.gpu \
    --format=csv,noheader,nounits > "$1" || fail "nvidia-smi power query failed (see above)"
}
record_power "$GPUQ_OUT/power-before.txt"
# Live reference at pin time (2026-11-22 01:29 CET, no lease held): power.limit
# 300.00 W, default_limit 600.00 W, min 150 / max 600. The 350 W in the release
# docs is not the enforced value (see 21-DONOR-RECORD.md and the escalation task).
cap_now="$(awk -F', *' '{printf "%d", $1; exit}' "$GPUQ_OUT/power-before.txt")"
[ -n "$cap_now" ] || fail "cannot read enforced power limit from power-before.txt"
echo "[bb-campaign] arm=$ARM repeat=$REPEAT src=${GPUQ_SGLANG_SRC:-?} cap=${cap_now}W expected=${CAP:-unset}"

if [ -n "$CAP" ] && [ "$cap_now" != "$CAP" ]; then
  # Every number in the run would be non-comparable. The measured host history
  # shows 600 W until 2026-11-18 14:51 CET, 300 W after, and no 350 W sample -
  # so an assertion failure here is a real finding, not a flake. Fix the cap
  # (sudo nvidia-smi -c 5 -pl <W>) or amend the pinned value with evidence.
  fail "enforced power limit ${cap_now} W != pinned ${CAP} W - run aborted, no number entered"
fi

NOTE_ARGS=(--note "arm=$ARM" --note "repeat=$REPEAT"
           --note "src=${GPUQ_SGLANG_SRC:-none}" --note "image=${GPUQ_SGLANG_IMAGE:-none}"
           --note "cap=${CAP:-unset}W" --note "job=${GPUQ_NAME:-none}")
export BETTERBENCH_NO_UPDATE_CHECK=1

# --endpoint and --model are required by 0.6.0; the model name is the queue's
# own GPUQ_MODEL, which the runner has already matched against /v1/models
# (gpuq-runner.py 940-969), so the step can never bench a different model than
# the one the server answers for.
RUN_LABEL="$(printf '%s' "${ARM}-r${REPEAT}" | tr -c 'a-zA-Z0-9._-' '-')"
"$BB" run \
  --no-update-check \
  --endpoint "${GPUQ_URL%/}/v1" \
  --model "${GPUQ_MODEL:-flash-next}" \
  --name "$RUN_LABEL" \
  --config "$CFG" \
  --corpus "$CORPUS_DIR" \
  --out "$GPUQ_OUT/results.json" \
  --no-html \
  "${NOTE_ARGS[@]}" 2>&1 | tee "$GPUQ_OUT/bb-campaign.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || fail "betterbench run failed (see bb-campaign.log)"

record_power "$GPUQ_OUT/power-after.txt"

# Field check through the queue's own parser, so the numbers this step prints
# are the numbers the ledger row will carry (scripts/gpu-queue/gpuq_parsers.py).
# Reimplementing the schema here would let the step and the runner disagree.
"$BB_PY" - "$GPUQ_OUT/results.json" \
          /srv/community-repos/qwen38-flashnext-nvfp4-kv/scripts/gpu-queue <<'PY' \
  || fail "results.json failed field validation"
import json, sys
sys.path.insert(0, sys.argv[2])
import gpuq_parsers as P

path, qdir = sys.argv[1], sys.argv[2]
raw = json.load(open(path, encoding="utf-8-sig"))
data = P.parse_betterbench(path=path)

want_levels = {"1", "2", "4", "6"}
conc = data.get("concurrency") or {}
pref = data.get("prefill") or {}
cfg = raw.get("config") or {}

problems = []
if not data.get("betterbench_version"):
    problems.append("betterbench_version missing (an unversioned artifact is not comparable)")
if data.get("corpus_version") is None and raw.get("corpus_version") is None:
    problems.append("corpus_version missing")
if raw.get("quick"):
    problems.append("run was --quick (5 passes): a smoke check, never a publishable number")
phases = set(data.get("phases") or ())
if phases != {"single_stream", "prefill", "concurrency"}:
    problems.append("phases measured %s, expected all three (single_stream, prefill, concurrency)"
                    % sorted(phases))
notes = (raw.get("env") or {}).get("notes") or {}
for k in ("arm", "repeat", "src", "job"):
    if not notes.get(k):
        problems.append("note %s=... is empty - the artifact cannot be attributed to an arm" % k)

# Config.load() drops unknown keys SILENTLY (config.py:66-73), so a typo in the
# pinned card would quietly bench the default instead. Three checks close that
# hole: every card key must be a real field, the echoed config must equal the
# card, and the corpus hash must be the one the plan records.
import json as _json
card = _json.load(open(__import__("os").path.join(__import__("os").path.dirname(path),
                                                  "campaign-20261121.json")))
from betterbench.config import Config
fields = set(Config.__dataclass_fields__)
stray = sorted(k for k in card if not k.startswith("_") and k not in fields)
if stray:
    problems.append("card keys are not Config fields (silently ignored): %s" % stray)
from betterbench.env import content_hash
for k, v in card.items():
    if k.startswith("_") or k not in fields:
        continue
    if cfg.get(k) != v:
        problems.append("echoed config.%s is %r, card says %r" % (k, cfg.get(k), v))
env = raw.get("env") or {}
want_corpus = "e332dceff176d033"
if env.get("corpus_hash") != want_corpus:
    problems.append("env.corpus_hash %r != pinned %r (the prompts are not the campaign corpus)"
                    % (env.get("corpus_hash"), want_corpus))
if cfg.get("greedy") is not True:
    problems.append(f"config.greedy is {cfg.get('greedy')!r}, expected True")
if cfg.get("seed") != 20261121:
    problems.append(f"config.seed is {cfg.get('seed')!r}, expected 20261121")
if set(conc) != want_levels:
    problems.append(f"concurrency levels {sorted(conc)} != {sorted(want_levels)}")
for lvl in sorted(conc):
    blk = conc[lvl]
    if blk.get("aggregate_tps") is None or blk.get("requests") in (None, 0):
        problems.append(f"c{lvl} has no aggregate_tps/requests")
    if blk.get("per_req_tps_p1") is None:
        problems.append(f"c{lvl} has no per_req_tps_p1 tail")
if not pref:
    problems.append("prefill ladder empty")
if data.get("decode_weighted_tps") is None:
    problems.append("decode_weighted_tps (the C1 metric) is null")

if problems:
    sys.exit("FAILED: " + "; ".join(problems))

print("[bb-campaign] ok  tool=%s schema=%s corpus=%s endpoint=%s model=%s"
      % (data.get("betterbench_version"), data.get("schema"),
         raw.get("corpus_version"), data.get("endpoint"), data.get("model")))
print("    C1 decode_weighted_tps=%.2f  ttft_p50_ms=%s  update_p99_ms=%s"
      % (data["decode_weighted_tps"], data.get("ttft_p50_ms"), data.get("update_p99_ms")))
for lvl in sorted(conc, key=int):
    b = conc[lvl]
    print("    c%s agg=%s per_req_med=%s per_req_p1=%s ttft_p50=%s ttft_p99=%s req=%s"
          % (lvl, b.get("aggregate_tps"), b.get("per_req_tps_med"),
             b.get("per_req_tps_p1"), b.get("ttft_p50_ms"), b.get("ttft_p99_ms"),
             b.get("requests")))
for depth in sorted(pref, key=lambda x: int(x) if str(x).isdigit() else 0):
    print("    prefill %s: %s" % (depth, pref[depth]))
PY

echo "[bb-campaign] done -> $GPUQ_OUT/results.json"
