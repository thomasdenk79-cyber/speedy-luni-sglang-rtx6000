# AGENTS.md — speedy-luni-sglang-rtx6000 (shared-tree working rules)

This fork is worked on by several concurrent agent runs in ONE shared
worktree on ONE branch (`main`). Two consequences are binding for every run.

## Push is team state, not run state

A peer's `git push` publishes the commits of ALL runs sharing this tree,
including yours. A report that says "landed locally, no push" is wrong here —
it cannot tell your work from a peer's transport. Reports MUST state the
landing as the commit that sits on the ref (`git rev-parse --short HEAD`
plus the path list of the commit), never as "kein Push".

## Mechanical gate before every push

`scripts/pushguard` (pre-push hook) protects the core refs (`main`, `dev`)
and LKG tags: direct pushes need the owner's blessing in the environment,
force-pushes and ref deletions are always refused, every decision is
appended to `log/push-gate.jsonl`:

```bash
LUNI_PUSH_GATE=1 \
LUNI_PUSH_REASON="Betreiber-Auftrag $(date -u '+%Y-%m-%d %H:%M'): <purpose>" \
  git push origin main
```

Hooks live in `.git/hooks` and do NOT survive a fresh clone. Re-arm after
any clone with the documented installer:

```bash
/srv/luni-repos/luni-ai-operator/scripts/pushguard-hook install \
  --repo /srv/community-repos/speedy-luni-sglang-rtx6000
/srv/luni-repos/luni-ai-operator/scripts/pushguard-hook status \
  --repo /srv/community-repos/speedy-luni-sglang-rtx6000
```

## Provenance of a landing

Because a peer's push can publish your work before you push yourself, the
attribution must not live in the reflog. Every run records each landing —
commit SHA + explicit path list — in
`/srv/luni-repos/luni-ai-operator/log/actions.log` (via `bin/alog`) and in
its task report. That record survives whatever a peer pushes.

## Shared-tree etiquette

- Commit only your own explicit paths: derive a private index with
  `/srv/luni-repos/luni-ai-operator/scripts/repo_index.sh \
  /srv/community-repos/speedy-luni-sglang-rtx6000 <task-slug>` and
  `git add -- <paths>`; never `git add -A`, never a bare commit.
- Dirty files of other runs stay untouched; a `git checkout`/`clean` in this
  tree is a destructive act against live work (incident 2026-11-22).
- `git` in scripts always wrapped as `env -u GIT_INDEX_FILE git ...` unless
  you deliberately point at your private index.
