#!/usr/bin/env python3
"""Check the exported review artifact for common metadata and path leaks."""

import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BLOCKED_PARTS = {"analysis", "data", "models", "artifacts", "outputs", "wandb", ".nltk_data"}
BLOCKED_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors", ".parquet", ".arrow", ".pdf", ".png", ".jpg", ".zip"}
SECRETS = re.compile(
    r"(?:hf_|ghp_)[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
)
PRIVATE_PATH = re.compile(r"/(?:home|Users|workspace)/[A-Za-z0-9_.-]+")
DASHBOARD = re.compile(r"https?://(?:api\.)?wandb\.ai/")


def source_files():
    if (ROOT / ".git").exists():
        return [ROOT / name for name in subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()]
    return [
        path
        for path in ROOT.rglob("*")
        if path.is_file() and not set(path.relative_to(ROOT).parts) & {"__pycache__", ".pytest_cache", ".ruff_cache"}
    ]


def main():
    failures = []
    files = source_files()
    for path in files:
        relative = path.relative_to(ROOT)
        # scripts/data is implementation code, not a downloaded dataset directory.
        if relative.parts[0] in BLOCKED_PARTS or ".nltk_data" in relative.parts or path.suffix in BLOCKED_SUFFIXES:
            failures.append(f"Artifact payload: {relative}")
        if path.is_symlink():
            failures.append(f"Symlink: {relative}")
            continue
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            failures.append(f"Unexpected binary file: {relative}")
            continue
        for label, pattern in (
            ("Credential", SECRETS),
            ("Personal filesystem path", PRIVATE_PATH),
            ("Dashboard", DASHBOARD),
        ):
            if pattern.search(text):
                failures.append(f"{label}: {relative}")
    for name in ("teachers", "datasets"):
        config = json.loads((ROOT / "configs" / f"{name}.json").read_text())
        for task, record in config.items():
            if record.get("teacher") or record.get("repo_id") or record.get("revision"):
                failures.append(f"Embedded private resource: {name}/{task}")
    if (ROOT / ".git").exists():
        remotes = subprocess.check_output(["git", "remote"], cwd=ROOT, text=True).strip()
        count = subprocess.check_output(["git", "rev-list", "--count", "HEAD"], cwd=ROOT, text=True).strip()
        authors = subprocess.check_output(["git", "log", "--format=%an|%ae|%cn|%ce"], cwd=ROOT, text=True).splitlines()
        if (
            remotes
            or count != "1"
            or any(
                line != "Anonymous Authors|anonymous@example.invalid|Anonymous Authors|anonymous@example.invalid"
                for line in authors
            )
        ):
            failures.append("Git metadata must contain one anonymous commit and no remote")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"PASS: {len(files)} source files; resource placeholders; credentials; paths; Git metadata")


if __name__ == "__main__":
    main()
