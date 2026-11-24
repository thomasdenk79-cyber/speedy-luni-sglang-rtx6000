#!/usr/bin/env python3
"""Deterministic coding-shaped replay: task novelty, repo shift, exact repeat."""

import argparse
import json
import statistics
import time
import urllib.request
from collections import defaultdict
from pathlib import Path


DOMAINS = (
    ("canopy", "ParcelLedger", "route_bucket", "zone", "weight"),
    ("quartz", "FrameArchive", "clip_catalog", "codec", "duration"),
    ("harbor", "BerthJournal", "mooring_index", "pier", "draft"),
    ("glacier", "ProbeRegister", "transect_rollup", "ridge", "depth"),
    ("orchard", "HarvestManifest", "cultivar_batch", "plot", "volume"),
    ("relay", "DispatchNotebook", "dispatch_span", "lane", "priority"),
)


def ticket(domain, round_id, task):
    repo, cls, fn, key, value = domain
    symbols = [f"{repo}_{fn}_{i:02}" for i in range(13)]
    module = f"src/{repo}/{fn}.py"
    test = f"tests/{repo}/test_{fn}.py"
    issue = (
        f"Ignore rows with a missing {key}; preserve the first-seen order of other buckets."
        if task == 0 else
        f"Treat boolean {value} as invalid and skip it without changing behavior for integers."
    )
    files = f"$ sed -n '1,120p' {module}\nfrom dataclasses import dataclass\n\n"
    files += f"@dataclass(frozen=True)\nclass {cls}:\n    {key}: str | None\n    {value}: int\n\n"
    files += f"def {fn}(records):\n    totals = {{}}\n    for record in records:\n"
    files += f"        bucket = record.{key}\n        totals[bucket] = totals.get(bucket, 0) + record.{value}\n    return totals\n\n"
    for symbol in symbols:
        files += f"def {symbol}(records):\n    return tuple(sorted({fn}(records).items()))\n\n"
    case = f"{cls}(None, 4)" if task == 0 else f"{cls}('left', True)"
    header = f"REPLAY-{round_id}-{repo}-{task}: Coding issue in {repo}. {issue}\n"
    history = header + f"Repository files collected by tools:\n{files}"
    yield "inspect", history + "\nExplain the defect and propose one focused test."
    history += f"\nAssistant checkpoint: inspected {module}; keep the patch local and cover the regression."
    history += f"\n$ sed -n '1,80p' {test}\nfrom {repo}.{fn} import {cls}, {fn}\n"
    history += f"def test_regression():\n    assert {fn}([{case}, {cls}('right', 2)]) == {{'right': 2}}\n"
    yield "edit", history + f"\nReturn a minimal unified diff for {module}. Do not add a dependency."
    history += f"\nTool result: applied a candidate edit in {module}.\n$ pytest -q {test}\n1 passed\n"
    history += f"$ git diff --stat\n {module} | 3 ++-\n"
    yield "review", history + "\nReview the edit for typing, ordering and unintended behavior; suggest one edge case."
    yield "compact", header + f"\nPrevious session summary: {test} passed after an edit in {module}. Relevant function: {fn}.\nWrite a short handoff with the test command and any remaining risk."


def make_plan(round_id):
    first, second = DOMAINS[2 * (round_id - 1):2 * round_id]
    novel_task = int(round_id == 2)
    plan = (
        ("first_repo", first, novel_task), ("same_repo_new_ticket", first, 1 - novel_task),
        ("repo_switch", second, novel_task), ("switched_repo_new_ticket", second, 1 - novel_task),
        ("exact_replay", first, novel_task),
    )
    return [(phase, stage, prompt) for phase, domain, task in plan for stage, prompt in ticket(domain, round_id, task)]


def post(url, path, data):
    request = urllib.request.Request(url + path, json.dumps(data).encode(), {"Content-Type": "application/json"})
    return urllib.request.urlopen(request, timeout=180)


def generate(url, prompt):
    # Apply the server's chat template; raw /generate text may stop at prefill.
    # Tokenization is outside the request timer and does not populate the KV cache.
    with post(url, "/tokenize", {"model": "flash-next", "messages": [{"role": "user", "content": prompt}]}) as response:
        token_ids = json.load(response)["tokens"]
    data = {"input_ids": token_ids, "stream": True, "sampling_params": {"temperature": 0, "max_new_tokens": 256}}
    begin = time.monotonic()
    first = None
    meta = {}
    done = False
    with post(url, "/generate", data) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            data = line[6:].strip()
            if data == b"[DONE]":
                done = True
                break
            event = json.loads(data)
            if "error" in event:
                raise RuntimeError(event["error"])
            if first is None and event.get("output_ids"):
                first = time.monotonic()
            meta = event.get("meta_info", meta)
    end = time.monotonic()
    if not done or not meta.get("finish_reason") or meta["finish_reason"].get("type") == "abort":
        raise RuntimeError("generation did not finish normally")
    for key in ("completion_tokens", "prompt_tokens", "cached_tokens"):
        if key not in meta:
            raise RuntimeError(f"missing donor metric: {key}")
    steps = meta.get("spec_verify_ct", 0)
    drafts = meta.get("spec_num_correct_drafts")
    if drafts is None and (steps or meta["completion_tokens"] > 1):
        raise RuntimeError(f"missing draft counters with {meta['completion_tokens']} generated tokens")
    return {
        "prompt_tokens": meta["prompt_tokens"], "cached_tokens": meta["cached_tokens"],
        "completion_tokens": meta["completion_tokens"], "verify_steps": steps,
        "accepted_draft_tokens": drafts or 0, "ttft_s": first - begin if first is not None else None,
        "wall_s": end - begin, "completion_tps": meta["completion_tokens"] / (end - begin),
    }


def min_med_max(xs):
    return [min(xs), statistics.median(xs), max(xs)] if xs else None


def summarize(rows):
    phases = defaultdict(list)
    for row in rows:
        phases[row["phase"]].append(row)
    return {
        phase: {
            "requests": len(rr),
            "requests_with_verification": sum(r["verify_steps"] > 0 for r in rr),
            "accepted_draft_tokens_per_verify": sum(r["accepted_draft_tokens"] for r in rr) / steps if (steps := sum(r["verify_steps"] for r in rr)) else None,
            "accepted_drafts_per_verify_min_med_max": min_med_max([r["accepted_draft_tokens"] / r["verify_steps"] for r in rr if r["verify_steps"]]),
            "ttft_s_min_med_max": min_med_max([r["ttft_s"] for r in rr if r["ttft_s"] is not None]),
            "completion_tps_min_med_max": min_med_max([r["completion_tps"] for r in rr]),
            "kv_cached_ratio": sum(r["cached_tokens"] for r in rr) / sum(r["prompt_tokens"] for r in rr),
            "first_turn_cached_tokens": [r["cached_tokens"] for r in rr if r["stage"] == "inspect"],
        } for phase, rr in phases.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8001", choices=["http://127.0.0.1:8001"])
    parser.add_argument("--round", required=True, choices=[1, 2, 3], type=int)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plan = make_plan(args.round)
    (args.out / "prompts.json").write_text(json.dumps(plan, indent=2) + "\n")
    if args.plan_only:
        return
    rows = []
    with (args.out / "requests.jsonl").open("w", buffering=1) as out:
        for phase, stage, prompt in plan:
            row = {"phase": phase, "stage": stage, "ts": time.time(), **generate(args.url, prompt)}
            print(json.dumps(row), file=out)
            rows.append(row)
    summary = summarize(rows)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
