#!/usr/bin/env python3
"""Smoke test for the second-brain skill: proves save, dedupe, gated
retrieval and forget all work on this machine. Run:

    python3 scripts/selftest.py

Exits 0 and prints PASS when every check holds, 1 with the first failure.
Uses a throwaway database in a temp dir; touches nothing else."""

import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
BRAIN = os.path.join(HERE, "brain.py")

CHECKS = []


def check(name, ok, detail=""):
    CHECKS.append((name, ok, detail))
    print(("%s %s %s" % ("PASS" if ok else "FAIL", name, detail)).rstrip())
    return ok


def run(db, *args, input_text=None):
    return subprocess.run(
        [sys.executable, BRAIN, "--db", db] + list(args),
        input=input_text, capture_output=True, text=True,
    )


def main():
    tmp = tempfile.mkdtemp(prefix="second-brain-selftest-")
    db = os.path.join(tmp, "brain.db")
    ok = True

    r = run(db, "save", "--title", "Deploy checklist",
            "--text", "Deploy the worker with uv run pywrangler deploy, "
                      "then set secrets with wrangler secret put.")
    ok &= check("save", r.returncode == 0 and "saved #1" in r.stdout, r.stdout.strip())

    r = run(db, "save", "--title", "Deploy checklist",
            "--text", "Deploy the worker with uv run pywrangler deploy, "
                      "then set secrets with wrangler secret put.")
    ok &= check("dedupe", "duplicate of #1" in r.stdout, r.stdout.strip())

    r = run(db, "save", "--title", "Pickle recipe",
            "--text", "Sun-dry raw mango for two days, then mix with mustard "
                      "oil and spices and rest a week.")
    ok &= check("second save", "saved #2" in r.stdout, r.stdout.strip())

    r = run(db, "retrieve", "how do I deploy the worker?")
    ok &= check("relevant retrieval",
                r.returncode == 0 and "Deploy checklist" in r.stdout,
                r.stdout.strip() or r.stderr.strip())

    r = run(db, "retrieve", "what is the capital of France?")
    ok &= check("irrelevant query gated to NO_MATCH",
                r.returncode == 1 and r.stdout.strip() == "NO_MATCH",
                r.stdout.strip())

    r = run(db, "retrieve", "how long does the pickle rest?")
    ok &= check("no cross-contamination",
                r.returncode == 0 and "Pickle recipe" in r.stdout
                and "Deploy checklist" not in r.stdout,
                r.stdout.strip())

    r = run(db, "forget", "2")
    ok &= check("forget", "forgot #2" in r.stdout, r.stdout.strip())

    r = run(db, "stats")
    ok &= check("stats", "1 items" in r.stdout, r.stdout.strip())

    print()
    print("PASS" if ok else "FAIL", "-", sum(1 for _, good, _ in CHECKS if good),
          "of", len(CHECKS), "checks")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
