#!/usr/bin/env python3
"""Generate the fork patch register (docs/PATCHES.md) from git.

Source of truth: commits on the release branches of the sglang fork.
Stable patch numbers and hand-maintained status live in patches/register.yaml
(keyed by commit subject, which survives rebases). New commits get the next
free number; existing entries keep theirs.

Usage: scripts/patch_register.py [--sglang /srv/community-repos/sglang-wt-rc3]
       [--check]   # exit 1 if docs/PATCHES.md or the register would change
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
REGISTER = ROOT / "patches" / "register.yaml"
OUT = ROOT / "docs" / "PATCHES.md"
OWN_AUTHORS = {"thomasdenk79-cyber", "Thomas Denk"}
UPSTREAM = "https://github.com/sgl-project/sglang/pull/"

# (label, range) in release order; pending branches are reported, not released.
LINES = [
    ("rc2", "25c1c4394..rc/qwen38-rc2"),
    ("rc3", "rc/qwen38-rc2..rc/qwen38-rc3"),
]
PENDING = [
    "feat/draft-vocab-head-buckets",
    "perf/hb-lowrow-kernels",
    "perf/hb-prune-pdl",
    "perf/profile-spans",
    "feat/draft-embed-bind-before-pool",
]
TRAILER = re.compile(r"^(Co-authored-by|Signed-off-by|Reviewed-by|Change-Id):", re.I)
ENV = re.compile(r"\b(SGLANG_[A-Z0-9_]{3,})\b")


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout


def commits(repo: Path, rng: str) -> list[dict]:
    try:
        raw = git(repo, "log", "--reverse", "--no-merges",
                  "--format=%H%x1f%ad%x1f%an%x1f%s%x1f%b%x1e", "--date=short", rng)
    except subprocess.CalledProcessError:
        return []
    out = []
    for rec in raw.split("\x1e"):
        rec = rec.strip("\n")
        if not rec:
            continue
        h, date, author, subject, body = (rec.split("\x1f") + [""] * 5)[:5]
        files = git(repo, "show", "--name-only", "--format=", h).split()
        diff_add = "\n".join(
            l for l in git(repo, "show", "--format=", "-U0", h).splitlines()
            if l.startswith("+") and not l.startswith("+++")
        )
        stat = git(repo, "show", "--shortstat", "--format=", h).strip()
        out.append(dict(
            sha=h, date=date, author=author, subject=subject,
            body="\n".join(l for l in body.strip().splitlines() if not TRAILER.match(l)).strip(),
            files=files, stat=stat,
            tests=[f for f in files if "/test" in f or f.startswith("test")],
            env=sorted(set(ENV.findall(diff_add))),
        ))
    return out


def origin(c: dict) -> tuple[str, str]:
    prs = re.findall(r"\(#(\d{4,6})\)", c["subject"])
    if c["author"] in OWN_AUTHORS and c["subject"].startswith("[Backport]") and prs:
        return "BACKPORT", ",".join(f"#{p}" for p in prs)
    if c["author"] in OWN_AUTHORS:
        ported = re.findall(r"(?:PR|pull/)\s*#?(\d{2,6})", c["subject"] + " " + c["body"])
        return ("OWN" if not ported else "OWN-PORT", ",".join(f"#{p}" for p in ported))
    if prs:
        return "BACKPORT", ",".join(f"#{p}" for p in prs)
    ref = re.findall(r"pull/(\d{4,6})", c["body"])
    if ref:
        return "BACKPORT", ",".join(f"#{p}" for p in ref)
    return "BACKPORT-UNMERGED", ""


def kind(subject: str) -> str:
    m = re.match(r"\[(\w+)\]", subject)
    return m.group(1) if m else "-"


def load_register() -> dict:
    if REGISTER.exists():
        return yaml.safe_load(REGISTER.read_text()) or {"patches": []}
    return {"patches": []}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sglang", default="/srv/community-repos/sglang-wt-rc3", type=Path)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    reg = load_register()
    by_subject = {p["subject"]: p for p in reg["patches"]}
    next_no = max((p["no"] for p in reg["patches"]), default=0) + 1

    rows = []
    seen = set()
    for label, rng in LINES + [(f"pending:{b}", f"rc/qwen38-rc3..{b}") for b in PENDING]:
        for c in commits(a.sglang, rng):
            if c["subject"] in seen:
                continue
            seen.add(c["subject"])
            org, prs = origin(c)
            p = by_subject.get(c["subject"])
            if p is None:
                p = dict(no=next_no, subject=c["subject"], status="new",
                         gpu_validated="pending" if org.startswith("OWN") else "n/a",
                         upstream="deferred" if org.startswith("OWN") else ("upstream-open" if org == "BACKPORT-UNMERGED" else "is-upstream"),
                         note="")
                next_no += 1
                reg["patches"].append(p)
                by_subject[c["subject"]] = p
            p["release"] = label
            if org.startswith("BACKPORT") and p.get("upstream") in ("candidate", "deferred"):
                p["upstream"], p["gpu_validated"] = "is-upstream", "n/a"
            p["origin"] = org
            if org == "BACKPORT-UNMERGED" and p.get("upstream") == "is-upstream":
                p["upstream"] = "upstream-open"
            p["prs"] = prs
            rows.append((p, c))

    reg["patches"].sort(key=lambda p: p["no"])
    new_reg = yaml.safe_dump(reg, sort_keys=False, allow_unicode=True, width=120)
    md = render(rows)

    if a.check:
        old_reg = REGISTER.read_text() if REGISTER.exists() else ""
        old_md = OUT.read_text() if OUT.exists() else ""
        drift = old_reg != new_reg or old_md != md
        print("patch register: " + ("DRIFT" if drift else "up to date"))
        return 1 if drift else 0
    REGISTER.parent.mkdir(exist_ok=True)
    REGISTER.write_text(new_reg)
    OUT.write_text(md)
    print(f"wrote {OUT.relative_to(ROOT)} ({len(rows)} patches)")
    return 0


def pid(p: dict) -> str:
    return f"QFN-{p['no']:04d}"


def render(rows: list[tuple[dict, dict]]) -> str:
    from collections import Counter
    cnt = Counter((p["release"].split(":")[0], p["origin"]) for p, _ in rows)
    L = [
        "# Patch register — Qwen3.8 Flash-Next SGLang fork",
        "",
        "> Generated by `scripts/patch_register.py` from git; do not edit by hand.",
        "> Status, upstream plan and GPU validation: `patches/register.yaml`.",
        "",
        "## Contents",
        "- [Legend](#legend)",
        "- [Summary](#summary)",
        "- [Index](#index)",
        "- [Own patches in detail](#own-patches-in-detail)",
        "",
        "## Legend",
        "| Field | Meaning |",
        "|---|---|",
        "| `QFN-NNNN` | Stable patch number (survives rebases; keyed by subject) |",
        "| origin `OWN` | Developed in this fork |",
        "| origin `OWN-PORT` | Own port of an external PR/design (PR named) |",
        "| origin `BACKPORT` | Cherry-pick of an upstream sglang PR (link) |",
        "| origin `BACKPORT-UNMERGED` | Foreign commit without upstream PR number (open PR or branch) — rebase risk |",
        "| release | `rc2` production, `rc3` candidate, `pending:<branch>` not yet integrated |",
        "| gpu | GPU validation: `pending`, `pass`, `fail`, `n/a` |",
        "| upstream | Plan for own patches: `deferred` (no upstream PR before extended field testing), `candidate`, `submitted #N`, `fork-only`, `drop`; backports: `is-upstream`, `upstream-open` |",
        "",
        "## Summary",
        "| Release | Origin | Patches |",
        "|---|---|---|",
    ]
    for (rel, org), n in sorted(cnt.items()):
        L.append(f"| {rel} | {org} | {n} |")
    L += ["", "## Index",
          "| ID | Release | Origin | PR | Type | Subject | Commit | Tests | Env flags | GPU | Upstream |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for p, c in rows:
        prs = " ".join(f"[{x}]({UPSTREAM}{x[1:]})" for x in p["prs"].split(",") if x) if p["origin"].startswith("BACKPORT") else p["prs"]
        L.append(
            f"| {pid(p)} | {p['release']} | {p['origin']} | {prs} | {kind(c['subject'])} | "
            f"{c['subject'].replace('|', '/')} | `{c['sha'][:10]}` | {len(c['tests'])} | "
            f"{' '.join(f'`{e}`' for e in c['env'][:3])} | {p['gpu_validated']} | {p['upstream']} |"
        )
    L += ["", "## Own patches in detail", ""]
    for p, c in rows:
        if not p["origin"].startswith("OWN"):
            continue
        L += [
            f"### {pid(p)} — {c['subject']}",
            "",
            f"- Release: `{p['release']}` · commit `{c['sha'][:12]}` · {c['date']} · {c['stat']}",
            f"- Origin: {p['origin']}{' ' + p['prs'] if p['prs'] else ''} · GPU validation: {p['gpu_validated']} · Upstream: {p['upstream']}",
            f"- Env flags: {', '.join(f'`{e}`' for e in c['env']) or 'none'}",
            f"- Tests: {', '.join(f'`{t}`' for t in c['tests']) or '**none in this commit**'}",
            f"- Files: {', '.join(f'`{f}`' for f in c['files'] if f not in c['tests'])}",
        ]
        if p.get("note"):
            L.append(f"- Note: {p['note']}")
        L += ["", c["body"] or "_No description in commit message._", ""]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())
