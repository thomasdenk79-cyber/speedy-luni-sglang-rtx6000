#!/usr/bin/env bash
# Guard matrix for bench/bb-campaign.sh — campaign 2026-11-21-speedy-baseline.
#
# Why this file exists: "the gates fire" is a claim about code, and a claim
# without a rerunnable probe is an anecdote.  Each case below asserts BOTH the
# exit code and the FAIL text of the step, so a gate that quietly stopped
# firing cannot pass.  Nothing here touches the GPU, the lease or production:
# bench/mock_engine.py is the only endpoint ever called, and the numbers it
# produces are meaningless by design — only the gates are under test.
#
# Usage:  bench/test-guards.sh                 # full matrix, ~10 min, artifacts in /tmp
#         bench/test-guards.sh --out DIR       # put all artifacts under DIR
#         bench/test-guards.sh --keep          # keep artifacts and name the dir
#         bench/test-guards.sh --port 39200    # another port if 39124 is taken
#         bench/test-guards.sh --only positive,foreign-module   # subset, for a quick
#                                        # re-check of one case (the pass/fail count
#                                        # then covers only the shown cases)
# Exit:   0 all cases behaved, 1 a gate misbehaved, 2/3 fixtures or mock failed.
set -uo pipefail

BENCH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEP_SRC="$BENCH/bb-campaign.sh"
MOCK="$BENCH/mock_engine.py"
PINNED_BB=/srv/community-repos/BetterBench-wt-v060/.venv/bin/betterbench
CORPUS=/srv/community-repos/BetterBench-wt-v060/corpus/v1
PORT=39124
OUT_DIR=""
KEEP=0
ONLY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --out)  OUT_DIR="${2:?--out needs a directory}"; shift 2 ;;
    --keep) KEEP=1; shift ;;
    --port) PORT="${2:?--port needs a number}"; shift 2 ;;
    --only) ONLY="${2:?--only needs a comma separated case list}"; shift 2 ;;
    *) echo "unknown argument '$1' (use --out/--keep/--port/--only)" >&2; exit 2 ;;
  esac
done
WORK="${OUT_DIR:-${TMPDIR:-/tmp}/sb-guards-$$}"
[ -x "$PINNED_BB" ] || { echo "missing pinned tool: $PINNED_BB" >&2; exit 2; }
[ -d "$CORPUS" ]    || { echo "missing pinned corpus: $CORPUS" >&2; exit 2; }
rm -rf "$WORK"; mkdir -p "$WORK/fake/bin" "$WORK/other/bin" "$WORK/step"
# Run a COPY of the step, not the file in the working tree: bash reads a script
# lazily, so an edit that lands while the matrix is running shifts the byte
# offset and the interpreter starts executing fragments of the changed lines.
# That hit this harness for real on 2026-11-22 - one case died with
# "Syntaxfehler beim unerwarteten Symbol" at line 187, and the failure was mine,
# not the gate's. The copy also carries the card beside it, because the step
# locates its card relative to its own file.
cp -f "$STEP_SRC" "$WORK/step/bb-campaign.sh"
cp -f "$BENCH/campaign-20261121.json" "$WORK/step/campaign-20261121.json"
chmod +x "$WORK/step/bb-campaign.sh"
STEP="$WORK/step/bb-campaign.sh"
[ "$KEEP" = 1 ] || trap 'rm -rf "$WORK"' EXIT
BASE_ENV=(GPUQ_MODEL=flash-next GPUQ_NAME=sb-guards
          GPUQ_SGLANG_SRC=/srv/community-repos/sglang-wt-speedybase
          GPUQ_SGLANG_IMAGE=none)

# --- injector: the real tool runs, then the named fault lands in the artifact
cat > "$WORK/fake/bin/betterbench" <<'WRAP'
#!/usr/bin/env bash
REAL=__PINNED_BB__
OUT="" prev=""
for a in "$@"; do [ "$prev" = "--out" ] && OUT="$a"; prev="$a"; done
"$REAL" "$@"; rc=$?
if [ "${1:-}" != "--version" ] && [ -n "$OUT" ] && [ -n "${INJECT:-}" ]; then
  OUT="$OUT" INJECT="$INJECT" python3 - <<'PY'
import copy, json, os, sys
path = os.environ["OUT"]
d = json.load(open(path, encoding="utf-8-sig"))
card = os.path.join(os.path.dirname(path), "campaign-20261121.json")

def row(level):
    for r in d["concurrency"]:
        if str(r.get("level")) == str(level):
            return r
    return None

for f in os.environ["INJECT"].split(","):
    if f == "greedy_false":   d["config"]["greedy"] = False
    elif f == "seed_wrong":   d["config"]["seed"] = None
    elif f == "levels_wrong":                      # drop 6, invent 8 (the donor shape)
        d["concurrency"] = [r for r in d["concurrency"] if str(r.get("level")) != "6"]
        eight = copy.deepcopy(row("4")); eight["level"] = 8
        d["concurrency"].append(eight)
    elif f == "prefill_empty": d["prefill"] = []
    elif f == "c6_no_tail":    row("6")["decode_tps"] = []
    elif f == "agg_null":      row("6")["aggregate_tps"] = None
    elif f == "req_zero":      row("2")["requests"] = 0
    elif f == "corpus_hash":   d["env"]["corpus_hash"] = "0123456789abcdef"
    elif f == "no_version":    d["betterbench_version"] = None
    elif f == "decode_null":                       # C1 metric dies, phases intact
        for recs in (d.get("single_stream") or {}).values():
            for r in recs:
                if isinstance(r, dict):
                    r["decode_tps"] = None
    elif f == "card_stray":                        # card edited under the sha gate
        c = json.load(open(card))
        c["bogus_key"] = 1
        c["max_tokens_total"] = 5
        json.dump(c, open(card, "w"))
    else: sys.exit("unknown INJECT %r" % f)
json.dump(d, open(path, "w"))
PY
  [ $? -ne 0 ] && exit 9
fi
exit $rc
WRAP
sed -i "s#__PINNED_BB__#$PINNED_BB#" "$WORK/fake/bin/betterbench"
chmod +x "$WORK/fake/bin/betterbench"
# The step derives its validator interpreter from the tool location, so a real
# venv python has to sit beside the wrapper. A symlink is not enough: python
# resolves its own executable before looking for pyvenv.cfg, so a linked venv
# python lands on the base interpreter without betterbench. Measured 2026-11-22
# - the whole injected half of this matrix died on ModuleNotFoundError.
mkdir -p "$WORK/fake2/bin"
cp -f "$WORK/fake/bin/betterbench" "$WORK/fake2/bin/betterbench"   # no python3 sibling: forces the step's fallback
PINNED_PY="$(dirname "$PINNED_BB")/python3"
[ -x "$PINNED_PY" ] || { echo "missing pinned interpreter: $PINNED_PY" >&2; exit 2; }
printf '#!/bin/sh\nexec %s "$@"\n' "$PINNED_PY" > "$WORK/fake/bin/python3"
chmod +x "$WORK/fake/bin/python3"

printf '#!/usr/bin/env bash\necho "BetterBench 0.5.9"\n' > "$WORK/other/bin/betterbench"
# Ein zweites Install mit derselben Versionszeile: --version luegt nicht (das
# Wrapping druckt sie brav), aber das Modul, das der Validator importiert, liegt
# ausserhalb des gepinnten Baums. Ohne Modulpfad-Pruefung waere das ein stiller
# Vergleich zweier verschiedenen Codes - das Spender-Artefakt saehe aus wie eine
# rc2-Messung. Das delegieren ohne PYTHONPATH haelt die Versions-Zeile glaubwuerdig.
mkdir -p "$WORK/foreign/bin"
printf '#!/usr/bin/env bash\nif [ "${1:-}" = "--version" ]; then echo "BetterBench 0.6.0"; exit 0; fi\nexec env -u PYTHONPATH %s "$@"\n' \
  "$PINNED_BB" > "$WORK/foreign/bin/betterbench"
chmod +x "$WORK/foreign/bin/betterbench"
mkdir -p "$WORK/foreign/lib/python3.12/site-packages/betterbench"
printf '"""Foreign 0.6.0 stand-in: same version string, different tree."""\n__version__ = "0.6.0"\n' \
  > "$WORK/foreign/lib/python3.12/site-packages/betterbench/__init__.py"
chmod +x "$WORK/other/bin/betterbench"

# tampered card beside a copy of the step: tests the sha gate, which is
# otherwise unreachable now that the step locates its own card
mkdir -p "$WORK/tampered"
cp -f "$STEP" "$WORK/tampered/bb-campaign.sh"
python3 - "$BENCH/campaign-20261121.json" "$WORK/tampered/campaign-20261121.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
c["seed"] = 1                                  # a late edit under the sha gate
json.dump(c, open(sys.argv[2], "w"), sort_keys=True)
PY

rm -rf "$WORK/corpus-7"; mkdir -p "$WORK/corpus-7"
i=0; for f in "$CORPUS"/*.jsonl; do i=$((i+1)); [ "$i" -le 7 ] && cp -f "$f" "$WORK/corpus-7/"; done
[ "$i" -eq 8 ] || { echo "fixture error: corpus has $i *.jsonl, expected 8" >&2; exit 2; }

python3 "$MOCK" "$PORT" >"$WORK/mock.log" 2>&1 &
MOCK_PID=$!
trap 'kill $MOCK_PID 2>/dev/null' EXIT
for _ in $(seq 1 30); do
  curl -fsS -m 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
  sleep 1
done
curl -fsS -m 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 \
  || { echo "mock engine did not come up on :$PORT (see $WORK/mock.log)" >&2; exit 3; }

pass=0; failed=0; bad=""
run_case() {            # name  want_rc  want_text  [env assignments...]
  local name=$1 want_rc=$2 want_text=$3; shift 3
  if [ -n "$ONLY" ]; then
    case ",$ONLY," in *",$name,"*) : ;; *) return 0 ;; esac
  fi
  local out="$WORK/out-$name" log="$WORK/log-$name.txt" step="$STEP"
  local args=(--arm=arm-a-rc2 --cap=300 --repeat=1) extra=() a
  for a in "$@"; do
    case "$a" in
      STEP=*) step="${a#STEP=}" ;;
      ARGS=*) read -r -a args <<< "${a#ARGS=}" ;;
      *) extra+=("$a") ;;
    esac
  done
  rm -rf "$out"; mkdir -p "$out"
  env -u INJECT -u BB_BIN -u BB_CORPUS "${BASE_ENV[@]}" \
      GPUQ_URL="http://127.0.0.1:$PORT" GPUQ_OUT="$out" "${extra[@]}" \
      bash "$step" "${args[@]}" >"$log" 2>&1
  local rc=$? verdict=PASS
  [ "$rc" = "$want_rc" ] || verdict=WRONG-RC
  { [ -z "$want_text" ] || grep -qF -- "$want_text" "$log"; } || verdict=WRONG-TEXT
  printf '%-15s rc=%-3s want=%-3s %-10s | %s\n' "$name" "$rc" "$want_rc" "$verdict" \
     "$(grep -m1 -oE '(FAIL|FAILED): .*' "$log" | cut -c1-125)"
  if [ "$verdict" = PASS ]; then pass=$((pass+1)); else
    failed=$((failed+1)); bad="$bad $name"; fi
}

echo "== guard matrix: $(basename "$STEP") vs mock engine :$PORT  ($(date -u +%FT%TZ)) =="
FAKE="$WORK/fake/bin/betterbench"
run_case positive        0 ""
run_case sha-mismatch    1 "pinned config sha mismatch"  STEP="$WORK/tampered/bb-campaign.sh"
run_case cap-mismatch    1 "enforced power limit"        ARGS="--arm=arm-a-rc2 --cap=350 --repeat=1"
run_case unknown-arg     1 "unknown argument"            ARGS="--arm=arm-a-rc2 --cap=300 --oops=1"
run_case foreign-tool    1 "version is not 0.6.0"        BB_BIN="$WORK/other/bin/betterbench"
run_case corpus-7        1 "expected 8"                  BB_CORPUS="$WORK/corpus-7"
run_case engine-down     1 "engine not reachable"        GPUQ_URL="http://127.0.0.1:39999"
run_case no-url          1 "GPUQ_URL not set"            GPUQ_URL=
run_case greedy-false    1 "config.greedy is False"                      INJECT=greedy_false  BB_BIN="$FAKE"
run_case seed-wrong      1 "config.seed"                                 INJECT=seed_wrong     BB_BIN="$FAKE"
run_case levels-wrong    1 "concurrency levels"                          INJECT=levels_wrong   BB_BIN="$FAKE"
run_case prefill-empty   1 "prefill ladder empty"                        INJECT=prefill_empty  BB_BIN="$FAKE"
run_case c6-no-tail      1 "has no per_req_tps_p1 tail"                  INJECT=c6_no_tail     BB_BIN="$FAKE"
run_case agg-null        1 "no aggregate_tps/requests"                   INJECT=agg_null       BB_BIN="$FAKE"
run_case req-zero        1 "no aggregate_tps/requests"                   INJECT=req_zero       BB_BIN="$FAKE"
run_case corpus-hash     1 "!= pinned"                                   INJECT=corpus_hash    BB_BIN="$FAKE"
run_case no-version      1 "betterbench_version missing"                 INJECT=no_version     BB_BIN="$FAKE"
run_case decode-null     1 "decode_weighted_tps (the C1 metric) is null" INJECT=decode_null    BB_BIN="$FAKE"
run_case card-stray      1 "not Config fields"                           INJECT=card_stray     BB_BIN="$FAKE"
# Same fault, but with no usable interpreter beside the wrapper: the step must
# fall back to the pinned venv python and still refuse the artifact.
run_case no-sibling-py   1 "config.greedy is False"                      INJECT=greedy_false  BB_BIN="$WORK/fake2/bin/betterbench"
# Zweites Install, gleiche Versionszeile: das Wrapping druckt 'BetterBench 0.6.0'
# brav, also kann nur der Modulpfad entlarven. Der Wegfall des Falls mit BB_BIN=$FAKE
# war ein Konstruktionsfehler: dort crashte --version schon an der PYTHONPATH-Blende
# und die rc kam aus der falschen Gate-Zeile (gemessen 2026-11-22, WRONG-TEXT).
run_case foreign-module  1 "not the pinned tree"  BB_BIN="$WORK/foreign/bin/betterbench" \
    PYTHONPATH="$WORK/foreign/lib/python3.12/site-packages"

echo "== $pass pass, $failed fail${bad:+ ->$bad} =="
echo "   artifacts: $WORK (per-case logs log-<case>.txt, artifacts out-<case>/)"
[ "$failed" -eq 0 ]
