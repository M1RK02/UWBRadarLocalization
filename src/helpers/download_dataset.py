"""Download the UWB dataset from HuggingFace into ./multi-person-localization/.

The dataset is gitignored, so a fresh clone won't have it. Run this once after
cloning to fetch it.

Prereqs:
    .venv/bin/pip install huggingface_hub

Auth (the dataset is gated — you need to be granted access first):
    - Get a token at https://huggingface.co/settings/tokens (read scope), then
      either run `huggingface-cli login` once, or set HF_TOKEN in your env, or
      pass --token. This script also falls back to an interactive prompt.

Usage:
    .venv/bin/python src/helpers/download_dataset.py
"""

import argparse
import os
from pathlib import Path

from huggingface_hub import login, snapshot_download

REPO_ID = "HAEEAI/multi-person-localization"
# Repo root
DEFAULT_DEST = (
    Path(__file__).resolve().parent.parent.parent / "multi-person-localization"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="HuggingFace access token (defaults to $HF_TOKEN, then cached login).",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help=f"Destination directory (default: {DEFAULT_DEST}).",
    )
    args = parser.parse_args()

    # If no token is available anywhere, prompt interactively; otherwise this
    # writes the token to the local HF cache so it's remembered next time.
    login(token=args.token)

    args.dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {REPO_ID} -> {args.dest} ...")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=args.dest,
    )
    print("Done.")


if __name__ == "__main__":
    main()
