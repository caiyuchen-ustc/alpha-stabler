import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def launcher():
    spec = importlib.util.spec_from_file_location("review_launcher", ROOT / "scripts/train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resource_config_contains_no_author_hosted_defaults():
    for name in ("datasets", "teachers"):
        config = json.loads((ROOT / "configs" / f"{name}.json").read_text())
        assert set(config) == {"math", "science", "code", "instruction"}
        for entry in config.values():
            assert not entry.get("repo_id")
            assert not entry.get("teacher")
            assert not entry.get("revision")


def test_teacher_path_supplied_by_environment(monkeypatch):
    run = launcher()
    monkeypatch.setenv("TEACHER_MODEL_PATH", "models/review-teacher")
    args = run.parser().parse_args(["opd", "science"])
    config = run.compose_config(run.build_command(args, []))
    assert config.actor_rollout_ref.ref.model.path == "models/review-teacher"


def test_anonymous_download_requires_explicit_mirror(tmp_path):
    spec = importlib.util.spec_from_file_location("review_data", ROOT / "scripts/data/manage.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from types import SimpleNamespace

    args = SimpleNamespace(pins=ROOT / "configs/datasets.json", domains=["science"], output=tmp_path)
    with pytest.raises(ValueError, match="DATASET_SCIENCE_ID"):
        module.download(args)
