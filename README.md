# Anonymous Review Artifact

Training code for reinforcement learning, on-policy and offline distillation,
trainable representation vectors, and Alpha-Stabler. The implementation uses verl,
FSDP and synchronous vLLM.

## Installation

Use Python 3.10+ with compatible CUDA, PyTorch, vLLM and FlashAttention builds.

```bash
pip install -e .
pip install -r requirements-opv.txt
python -m nltk.downloader punkt punkt_tab averaged_perceptron_tagger_eng
```

The package includes third-party source code. Its copyright and license notices
remain intact; see [LICENSE](LICENSE), [Notice.txt](Notice.txt), and source headers.
These upstream attributions do not identify the submitting authors.

## Models and data

The artifact contains source code, configurations and tests. Model weights,
datasets, training checkpoints and analysis assets are supplied separately.
No submitting-author repository identifiers or private resource locations are
embedded in the configuration.

| Task | Public base model | Training / evaluation |
| --- | --- | --- |
| Math | Qwen3-4B | DAPO teacher training; DeepMath distillation; AIME evaluation |
| Science | DeepSeek-R1-Distill-Qwen-1.5B | SciKnowEval |
| Code | Qwen3-8B | Eurus / LiveCodeBench |
| Instruction | DeepSeek-R1-Distill-Qwen-7B | Nemotron instruction following / IFBench |

Public base-model identifiers are in `configs/teachers.json`. Provide RL teacher
weights through `--teacher` or `TEACHER_MODEL_PATH`; the fallback location is
`models/<task>-teacher`. Provide original Parquet files in this layout:

```text
data/
  math/          train.parquet, teacher_train.parquet, validation.parquet, aime2025.parquet
  science/       train.parquet, validation.parquet
  code/          train.parquet, validation.parquet, eurus_validation.parquet
  instruction/   train.parquet, validation.parquet
```

Files keep their original columns and nested metadata. `--data-dir`, `--train-file`
and `--val-files` select alternate local locations. To make byte-identical copies
from a separate input directory with the same layout:

```bash
python scripts/data/manage.py prepare --source input_data --output data
python scripts/data/verify_hub.py
```

For a separately provided anonymous data mirror, set `DATASET_SCIENCE_ID` (or the
corresponding task name) and optionally `DATASET_SCIENCE_REVISION` in the environment.
Then use `python scripts/data/manage.py download --domains science`. No account or
mirror is selected automatically. Original file hashes are recorded only in local
manifests, which are excluded from the code artifact.

## Training

Run commands from the repository root. `TASK` is `math`, `science`, `code`, or
`instruction`; `--model` can select another model or a local checkpoint.

```bash
# RL teacher
bash scripts/rl_teacher/run.sh science --method full
bash scripts/rl_teacher/run.sh math --rl-algorithm dapo

# On-policy distillation
bash scripts/opd/run.sh science --teacher models/science-teacher --method vector
bash scripts/opd/run.sh science --teacher models/science-teacher --method lora
bash scripts/opd/run.sh science --teacher models/science-teacher \
  --method sequential --vectors 8 --stage-steps 100

# Offline teacher trajectories and distillation
bash scripts/offpolicy/generate.sh science --teacher models/science-teacher
python scripts/merge_rollouts.py \
  --inputs outputs/science-generate-full/round_0.parquet \
  --output data/science/teacher_responses.parquet
bash scripts/offpolicy/run.sh science --teacher models/science-teacher \
  --method vector --offline-data data/science/teacher_responses.parquet

# Alpha-Stabler, default task: science
bash scripts/alpha_stabler/run.sh --check
bash scripts/alpha_stabler/run.sh
```

`--method` supports `full`, `lora`, `vector`, `sequential`, and `gated` for the
applicable workflows. Layer ranges are zero-based and inclusive. Offline
distillation uses teacher trajectories with the existing reverse-KL surrogate.

The Alpha-Stabler launcher checks dependencies, visible GPUs, model files and dataset
schemas. It downloads public base weights if missing. Local datasets or an explicitly
configured anonymous mirror are required. `--no-download` enforces local resources.
See [Alpha-Stabler](docs/alpha_stabler.md) for configuration details.

All launchers accept `--dry-run`, `--config-only`, and Hydra `key=value` overrides.
Use `--gpus`, `--tp`, `--batch-size`, `--steps` and `--lr` to adjust training.
`PYTHON=/path/to/python` selects the environment for the Alpha-Stabler shell entry.
The default logging backends are console and local JSONL; network experiment
tracking requires an explicit `--wandb` option.

## Repository layout

```text
configs/                  public model defaults; anonymous resource placeholders
scripts/rl_teacher/       RL training
scripts/opd/              on-policy distillation
scripts/offpolicy/        teacher generation and offline distillation
scripts/alpha_stabler/     monitoring and gradient control
examples/representation/  parameterization variants and local-path templates
recipe/dapo/              DAPO trainer
recipe/code_sandbox/      code verifier execution service
verl/                    training library
tests/opv/               focused regression tests
```

## Checks

```bash
python -m pytest -q tests/opv
python scripts/check_repository.py
python scripts/audit_anonymity.py
```
