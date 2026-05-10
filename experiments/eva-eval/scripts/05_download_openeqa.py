"""Phase: clone openeqa repo, copy questions JSON, verify HM3D bundle URL.

Usage:
    python scripts/05_download_openeqa.py \
        --openeqa-repo-dir <path>            # where to clone the repo
        --out-questions-json <path>          # where to copy the questions JSON
        [--bundle-url <URL>]                 # override the default bundle URL
        [--no-bundle-check]                  # skip the URL HEAD check

This script does NOT download the episode bundle itself — that's done
on demand per episode by 06_preprocess_openeqa.py.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

OPENEQA_REPO_URL = "https://github.com/facebookresearch/open-eqa.git"
DEFAULT_QUESTIONS_RELPATH = "data/open-eqa-v0.json"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--openeqa-repo-dir", type=Path, required=True)
    ap.add_argument("--out-questions-json", type=Path, required=True)
    ap.add_argument("--bundle-url", default=None,
                    help="If set, HEAD-check this URL to confirm the HM3D "
                         "pre-rendered bundle is reachable.")
    ap.add_argument("--no-bundle-check", action="store_true")
    args = ap.parse_args()

    repo = args.openeqa_repo_dir.resolve()
    if repo.exists() and (repo / ".git").exists():
        print(f"[skip clone] {repo} already exists; pulling latest")
        subprocess.run(["git", "-C", str(repo), "pull", "--ff-only"], check=True)
    else:
        repo.parent.mkdir(parents=True, exist_ok=True)
        print(f"[clone] {OPENEQA_REPO_URL} -> {repo}")
        subprocess.run(["git", "clone", "--depth", "1", OPENEQA_REPO_URL, str(repo)], check=True)

    src_q = repo / DEFAULT_QUESTIONS_RELPATH
    if not src_q.exists():
        print(f"ERROR: expected questions JSON not found at {src_q}", file=sys.stderr)
        print(f"       Inspect the repo and update DEFAULT_QUESTIONS_RELPATH", file=sys.stderr)
        sys.exit(2)
    args.out_questions_json.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src_q, args.out_questions_json)
    print(f"[copy] {src_q} -> {args.out_questions_json}")

    if args.bundle_url and not args.no_bundle_check:
        print(f"[head] {args.bundle_url}")
        try:
            req = urllib.request.Request(args.bundle_url, method="HEAD")
            with urllib.request.urlopen(req, timeout=15) as resp:
                if resp.status >= 400:
                    raise RuntimeError(f"HTTP {resp.status}")
                print(f"[ok] bundle URL reachable (HTTP {resp.status})")
        except Exception as e:
            print(f"ERROR: bundle URL not reachable: {e}", file=sys.stderr)
            print(
                "Cannot proceed with the planned design (Habitat-GT depth+pose).\n"
                "Fallback options: (1) use a different mirror, (2) install Habitat-sim\n"
                "and HM3D scans and render trajectories yourself (out of scope here).",
                file=sys.stderr,
            )
            sys.exit(3)

    print("done.")


if __name__ == "__main__":
    main()
