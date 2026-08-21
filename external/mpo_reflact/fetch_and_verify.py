#!/usr/bin/env python3
"""Reproduce + hash-verify the vendored MPO assets under external/mpo_reflact/.

external/ is entirely gitignored in this repo (see .gitignore), the same as
external/reflexgrad and external/ADaPT -- so a fresh clone does not carry
these files. Since we deliberately did not add a git-tracked exception for
this one directory (kept the exclusion pattern uniform across every
external/ vendor), this script is the reproducibility path instead: pin the
exact upstream commit, fetch each file, and refuse to proceed if what
arrives doesn't hash-match what react_reflact_anchor was implemented and
tested against.

Usage:
    python3 external/mpo_reflact/fetch_and_verify.py          # fetch if missing/mismatched
    python3 external/mpo_reflact/fetch_and_verify.py --verify-only  # never fetch, just check
    python3 external/mpo_reflact/fetch_and_verify.py --force       # re-fetch even if already OK

Exits non-zero (and prints exactly which file and which hash mismatched) on
any verification failure -- never silently accepts different content than
what is pinned here, even if the fetch itself "succeeds" (e.g. upstream
force-pushed over the commit, or a proxy/cache served something else).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request

MPO_SOURCE_REPO = "https://github.com/WeiminXiong/MPO"
MPO_SOURCE_COMMIT = "5529eca70ab352eb7e56533ca44cb60fbf34bef1"
_RAW_BASE = f"https://raw.githubusercontent.com/WeiminXiong/MPO/{MPO_SOURCE_COMMIT}"

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# (upstream path, local path relative to this directory, expected sha256)
# Hashes computed once, at implementation time (2026-07-29), from the files
# alfworld_runs_ae/mpo_prompts.py and its tests were actually written and
# verified against -- see ae/baselines/tests/test_react_reflact_anchor.py
# and alfworld_runs_ae/tests/test_mpo_prompts.py.
FILES = [
    (
        "prompt/instructions/alfworld_inst.txt",
        "prompt/instructions/alfworld_inst.txt",
        "bd4f1f6bb04c4dbac216453bf19f1f70e1c54613e6801d3237758856eab82217",
    ),
    (
        "prompt/icl_examples/alfworld_icl.json",
        "prompt/icl_examples/alfworld_icl.json",
        "5912687e70cbe0c434d26e43d3f79dfdc50bf116c89e2a2e7fde12bc51eb8c9a",
    ),
    (
        "prompt/templates.py",
        "prompt/templates.py",
        "9e3b93653121a1578c592181a08dd009ba888a31f854f6777f9070749fd5bb33",
    ),
    (
        "envs/alfworld_env.py",
        "envs/alfworld_env_reference.py",
        "f5350a680b6d396bcf6fc18c57b594d7ccaf187c244606e084d0d65b151c3296",
    ),
    (
        "LICENSE",
        "LICENSE",
        "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
    ),
]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_local(local_path: str) -> bytes | None:
    full = os.path.join(_THIS_DIR, local_path)
    if not os.path.exists(full):
        return None
    with open(full, "rb") as f:
        return f.read()


def _fetch(upstream_path: str) -> bytes:
    url = f"{_RAW_BASE}/{upstream_path}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        return resp.read()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--verify-only", action="store_true",
                    help="Never fetch over the network; only check what's already on disk.")
    p.add_argument("--force", action="store_true",
                    help="Re-fetch even if the local file already hash-matches.")
    args = p.parse_args()

    ok = True
    for upstream_path, local_path, expected_sha in FILES:
        local_bytes = None if args.force else _read_local(local_path)
        local_sha = _sha256_bytes(local_bytes) if local_bytes is not None else None

        if local_sha == expected_sha:
            print(f"OK      {local_path}  (already matches pinned commit {MPO_SOURCE_COMMIT[:12]})")
            continue

        if args.verify_only:
            ok = False
            status = "MISSING" if local_bytes is None else "MISMATCH"
            print(f"{status:8s}{local_path}  expected sha256={expected_sha}"
                  + (f" got={local_sha}" if local_sha else " (file not found)"))
            continue

        print(f"FETCH   {local_path}  <- {MPO_SOURCE_REPO}@{MPO_SOURCE_COMMIT[:12]}/{upstream_path}")
        try:
            data = _fetch(upstream_path)
        except Exception as e:
            ok = False
            print(f"FAIL    {local_path}  could not fetch: {e}")
            continue

        actual_sha = _sha256_bytes(data)
        if actual_sha != expected_sha:
            ok = False
            print(f"FAIL    {local_path}  fetched content does NOT match the pinned hash -- "
                  f"refusing to write it. expected={expected_sha} got={actual_sha}")
            continue

        full = os.path.join(_THIS_DIR, local_path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "wb") as f:
            f.write(data)
        print(f"WROTE   {local_path}  sha256={actual_sha} (matches pinned commit)")

    if not ok:
        print("\nVerification FAILED -- do not proceed with react_reflact_anchor runs "
              "until every file above reads OK.")
        return 1
    print(f"\nAll {len(FILES)} vendored MPO files verified against pinned commit "
          f"{MPO_SOURCE_COMMIT}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
