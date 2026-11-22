#!/usr/bin/env python3
"""Guard tests for scripts/pushguard — the pre-push gate of this shared tree.

Why this file exists (task 665): the gate protects the core refs (`main`,
`dev`) against unblessed, forced and deleting pushes and audits every
decision to `log/push-gate.jsonl`. A gate that quietly stops firing is worse
than no gate, so each case asserts the exit code AND the refusal text, and
runs against a throwaway git repo (never the shared one, never a real push).

Run:  python3 -m unittest discover -s tests -v   (no GPU, no lease, no network)
"""
from __future__ import annotations

import datetime
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO_TOP = Path(__file__).resolve().parent.parent
PUSHGUARD = REPO_TOP / "scripts" / "pushguard"
ZERO = "0" * 40


def git(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.pop("GIT_INDEX_FILE", None)  # never touch the caller's run index
    return subprocess.run(
        ["git", *args], cwd=cwd, env=env,
        capture_output=True, text=True, check=True,
    )


class PushGuardCase(unittest.TestCase):
    """Shared fixture: one throwaway repo, one commit, clean gate env."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="pushguard-test-")
        self.repo = Path(self._tmp.name)
        git(self.repo, "init", "-q", "-b", "main")
        git(self.repo, "config", "user.email", "t@t")
        git(self.repo, "config", "user.name", "t")
        (self.repo / "f").write_text("1\n")
        git(self.repo, "add", "--", "f")
        git(self.repo, "commit", "-q", "-m", "base")
        self.head = git(self.repo, "rev-parse", "HEAD").stdout.strip()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def call(self, lines: list[str], *, gate: str = "0", reason: str = "",
             extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
        env = dict(os.environ)
        env.pop("GIT_INDEX_FILE", None)
        env.pop("LUNI_REPO_TOP", None)
        env["LUNI_PUSH_GATE"] = gate
        env["LUNI_PUSH_REASON"] = reason
        for k in ("PUSH_PROTECTED_REFS", "PUSH_LKG_PREFIXES"):
            env.pop(k, None)
        if extra_env:
            env.update(extra_env)
        return subprocess.run(
            [str(PUSHGUARD), "origin", "https://example.invalid/repo.git"],
            cwd=self.repo, env=env, input="\n".join(lines) + "\n",
            capture_output=True, text=True, timeout=30,
        )

    # --- the four behaviours the shared tree depends on -------------------

    def test_unblessed_push_to_main_is_refused(self) -> None:
        p = self.call([f"refs/heads/main {self.head} refs/heads/main {ZERO}"])
        self.assertEqual(p.returncode, 1, p.stdout + p.stderr)
        self.assertIn("VERWEIGERT", p.stderr)
        self.assertIn("Betreiber-Freigabe", p.stderr)

    def test_blessed_push_to_main_is_allowed_and_audited(self) -> None:
        today = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        p = self.call(
            [f"refs/heads/main {self.head} refs/heads/main {ZERO}"],
            gate="1", reason=f"Betreiber-Auftrag {today}: land task 665 probe",
        )
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertIn("erlaubt", p.stdout)
        ledger = self.repo / "log" / "push-gate.jsonl"
        self.assertTrue(ledger.is_file(), "audit line missing")
        row = json.loads(ledger.read_text().splitlines()[-1])
        self.assertEqual(row["aktion"], "erlaubt")
        self.assertEqual(row["ref"], "refs/heads/main")

    def test_blessing_without_reason_or_timestamp_is_refused(self) -> None:
        p = self.call([f"refs/heads/main {self.head} refs/heads/main {ZERO}"],
                      gate="1", reason="")
        self.assertEqual(p.returncode, 1)
        p = self.call([f"refs/heads/main {self.head} refs/heads/main {ZERO}"],
                      gate="1", reason="Betreiber-Auftrag: no stamp here")
        self.assertEqual(p.returncode, 1, "timestamp duty not firing")
        self.assertIn("Zeitstempel", p.stderr)

    def test_future_stamp_is_refused(self) -> None:
        future = (datetime.datetime.now()
                  + datetime.timedelta(days=2)).strftime("%Y-%m-%d %H:%M")
        p = self.call([f"refs/heads/main {self.head} refs/heads/main {ZERO}"],
                      gate="1", reason=f"Betreiber-Auftrag {future}: made up")
        self.assertEqual(p.returncode, 1, "future-stamp guard not firing")
        self.assertIn("Zukunft", p.stderr)

    def test_delete_of_main_is_refused_even_with_blessing(self) -> None:
        today = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        p = self.call([f"refs/heads/main {ZERO} refs/heads/main {self.head}"],
                      gate="1", reason=f"Betreiber-Auftrag {today}: delete try")
        self.assertEqual(p.returncode, 1)
        self.assertIn("geloescht", p.stderr)

    def test_non_fast_forward_is_refused(self) -> None:
        # a second, unrelated commit as remote_sha: not an ancestor of HEAD
        git(self.repo, "checkout", "-q", "--orphan", "side")
        (self.repo / "g").write_text("2\n")
        git(self.repo, "add", "--", "g")
        git(self.repo, "commit", "-q", "-m", "side")
        foreign = git(self.repo, "rev-parse", "HEAD").stdout.strip()
        git(self.repo, "checkout", "-q", "main")
        today = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        p = self.call([f"refs/heads/main {self.head} refs/heads/main {foreign}"],
                      gate="1", reason=f"Betreiber-Auftrag {today}: rewind try")
        self.assertEqual(p.returncode, 1)
        self.assertIn("non-fast-forward", p.stderr)

    def test_unprotected_branch_passes_without_gate(self) -> None:
        p = self.call([f"refs/heads/feature {self.head} refs/heads/feature {ZERO}"])
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)


if __name__ == "__main__":
    unittest.main()
