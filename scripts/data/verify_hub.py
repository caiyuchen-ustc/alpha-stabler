#!/usr/bin/env python3
"""Verify local data hashes; optionally compare an explicitly configured Hub mirror."""

import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def digest(path, git=False):
    result = hashlib.sha1() if git else hashlib.sha256()
    if git:
        result.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=os.environ.get("DATA_ROOT", str(ROOT / "data")))
    parser.add_argument("--pins", default=str(ROOT / "configs/datasets.json"))
    parser.add_argument("--domains", nargs="+", default=["math", "science", "code", "instruction"])
    parser.add_argument("--remote", action="store_true")
    args = parser.parse_args()
    pins = json.loads(Path(args.pins).read_text())
    for domain in args.domains:
        folder = Path(args.data_dir) / domain
        manifest = json.loads((folder / "manifest.json").read_text())
        files = None
        if args.remote:
            from huggingface_hub import HfApi

            pin = pins[domain]
            repo = os.environ.get(f"DATASET_{domain.upper()}_ID") or pin.get("repo_id")
            revision = os.environ.get(f"DATASET_{domain.upper()}_REVISION") or pin.get("revision")
            if not repo:
                raise ValueError(f"Set DATASET_{domain.upper()}_ID before remote verification")
            files = {
                entry.rfilename: entry
                for entry in HfApi().dataset_info(repo, revision=revision, files_metadata=True).siblings
            }
        for split, record in manifest["splits"].items():
            filename = split + ".parquet"
            path = folder / filename
            actual = digest(path)
            if actual != record["sha256"] or actual != record.get("source_sha256", actual):
                raise ValueError(f"Hash mismatch: {domain}/{split}")
            if files:
                entry = files[filename]
                expected = entry.lfs.sha256 if entry.lfs else entry.blob_id
                if expected != (actual if entry.lfs else digest(path, git=True)):
                    raise ValueError(f"Remote hash mismatch: {domain}/{split}")
            print(f"PASS {domain}/{filename}", flush=True)


if __name__ == "__main__":
    main()
