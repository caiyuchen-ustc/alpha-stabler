#!/usr/bin/env python3
"""Stage original Parquet files or download from an explicitly configured mirror."""

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOMAINS = {
    "math": {
        "files": {
            "train": "math/train.parquet",
            "teacher_train": "math/teacher_train.parquet",
            "validation": "math/validation.parquet",
            "aime2025": "math/aime2025.parquet",
        }
    },
    "science": {"files": {"train": "science/train.parquet", "validation": "science/validation.parquet"}},
    "code": {
        "files": {
            "train": "code/train.parquet",
            "validation": "code/validation.parquet",
            "eurus_validation": "code/eurus_validation.parquet",
        }
    },
    "instruction": {"files": {"train": "instruction/train.parquet", "validation": "instruction/validation.parquet"}},
}


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def prepare(args):
    import pyarrow.parquet as pq

    for domain in args.domains:
        destination = Path(args.output) / domain
        destination.mkdir(parents=True, exist_ok=True)
        manifest = {"domain": domain, "splits": {}}
        prompts = {}
        for split, filename in DOMAINS[domain]["files"].items():
            source = Path(args.source) / filename
            target = destination / (split + ".parquet")
            if source.resolve() == target.resolve():
                raise ValueError("Source and staging directory must differ")
            if target.exists():
                if sha256(source) != sha256(target):
                    raise FileExistsError(f"Different file already exists: {target}")
            else:
                shutil.copy2(source, target)
            source_hash, target_hash = sha256(source), sha256(target)
            if source_hash != target_hash:
                raise RuntimeError(f"Copy hash mismatch: {split}")
            parquet = pq.ParquetFile(target)
            prompt_hashes = set()
            for batch in parquet.iter_batches(batch_size=256, columns=["prompt"]):
                for row in batch.to_pylist():
                    value = json.dumps(row["prompt"], sort_keys=True, ensure_ascii=False)
                    prompt_hashes.add(hashlib.sha256(value.encode()).hexdigest())
            prompts[split] = prompt_hashes
            manifest["splits"][split] = {
                "rows": parquet.metadata.num_rows,
                "sha256": target_hash,
                "source_sha256": source_hash,
                "unique_prompts": len(prompt_hashes),
            }
            print(f"{domain}/{split}: {parquet.metadata.num_rows:,} rows, original bytes preserved", flush=True)
        manifest["train_validation_exact_prompt_overlap"] = len(prompts["train"] & prompts["validation"])
        (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def download(args):
    from huggingface_hub import snapshot_download

    pins = json.loads(Path(args.pins).read_text())
    for domain in args.domains:
        pin = pins[domain]
        repo = os.environ.get(f"DATASET_{domain.upper()}_ID") or pin.get("repo_id")
        revision = os.environ.get(f"DATASET_{domain.upper()}_REVISION") or pin.get("revision")
        if not repo:
            raise ValueError(f"Set DATASET_{domain.upper()}_ID to a review-safe mirror or stage local files")
        snapshot_download(
            repo,
            repo_type="dataset",
            revision=revision,
            local_dir=str(Path(args.output) / domain),
            allow_patterns=["*.parquet", "manifest.json"],
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "download"])
    parser.add_argument("--source", default="input_data")
    parser.add_argument("--output", default=os.environ.get("DATA_ROOT", str(ROOT / "data")))
    parser.add_argument("--domains", choices=list(DOMAINS), nargs="+", default=list(DOMAINS))
    parser.add_argument("--pins", default=str(ROOT / "configs/datasets.json"))
    args = parser.parse_args()
    {"prepare": prepare, "download": download}[args.action](args)


if __name__ == "__main__":
    main()
