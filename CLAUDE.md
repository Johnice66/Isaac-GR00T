# CLAUDE.md — Isaac GR00T N1.7

## Project overview

Isaac GR00T N1.7 is an open vision-language-action (VLA) model for generalized humanoid robot skills.
The repo contains the model, training pipeline, evaluation harness, and deployment tooling.

- **Language:** Python 3.10 (dGPU, Orin); Python 3.12 (Thor, DGX Spark — see deployment dir)
- **Package manager:** [uv](https://docs.astral.sh/uv/)
- **Build system:** setuptools (see `pyproject.toml`)
- **CI:** internal GitLab CI (`.gitlab-ci.yml` + includes under `ci/`, not shipped to the public GitHub EA repo); public GitHub Actions (`.github/workflows/`)

## Quick-start commands

```bash
# Install (dev mode with all extras)
uv sync --all-extras

# Lint and format (uses ruff via pre-commit)
pre-commit run --all-files

# Run CPU tests
python -m pytest tests/ -m "not gpu" -v --timeout=300

# Run GPU tests
python -m pytest tests/ -m gpu -v --timeout=300

# Build package
uv build

# Validate lockfile
uv lock --locked
```

## Code style

- Formatter: `ruff format` (double quotes, spaces, line-length 100)
- Linter: `ruff check` with rules E, F, I (ignores E501)
- Config lives in `pyproject.toml` under `[tool.ruff]`
- Run `pre-commit run --all-files` before committing

## Directory layout

```
gr00t/              # Main package
  configs/          #   Training, data, and model configs
  data/             #   Data loading, embodiment tags, dataset processing
  eval/             #   Evaluation (run_gr00t_server.py)
  experiment/       #   Training pipeline (launch_finetune.py, trainer.py)
  model/            #   Model architecture (N1.7, base, modules)
  policy/           #   Policy inference (Gr00tPolicy, server/client)
examples/           # Per-embodiment example configs and READMEs
scripts/            # Deployment, conversion, and utility scripts
  deployment/       #   Platform install scripts (dgpu, orin, thor, spark)
tests/              # pytest suite (markers: gpu, not gpu)
getting_started/    # User-facing guides and notebooks
```

## Key entry points

- **Fine-tune:** `bash examples/finetune.sh --base-model-path <path> --dataset-path <path> --embodiment-tag <tag> --output-dir <dir>`
- **Inference server:** `python gr00t/eval/run_gr00t_server.py --model-path <path> --embodiment-tag <tag>`
- **ONNX export:** `python scripts/deployment/export_onnx_n1d7.py`
- **TensorRT build:** `python scripts/deployment/build_trt_pipeline.py`
- **Benchmark:** `python scripts/deployment/benchmark_inference.py`

## Testing

- Test markers: `gpu` (requires GPU), default is CPU-safe
- Fixtures live in `tests/fixtures/` and `demo_data/`
- CI runs CPU and GPU tests in separate jobs with 300s timeout

## Deployment platforms

- **dGPU (H100, A100, RTX):** CUDA 12.8 — install via `scripts/deployment/dgpu/install_deps.sh`, container via top-level `docker/Dockerfile` (supports x86_64 and aarch64)
- **Jetson Orin:** CUDA 12.6 — install via `scripts/deployment/orin/install_deps.sh`, container via `scripts/deployment/orin/Dockerfile`
- **Jetson Thor:** CUDA 13.0 — install via `scripts/deployment/thor/install_deps.sh`, container via `scripts/deployment/thor/Dockerfile`
- **DGX Spark:** CUDA 13.0 — install via `scripts/deployment/spark/install_deps.sh`, container via `scripts/deployment/spark/Dockerfile`

Each Jetson/Spark platform ships an `activate_*.sh` helper (`scripts/activate_orin.sh`, `scripts/activate_spark.sh`, `scripts/activate_thor.sh`) that exports platform-specific library paths. For dGPU, the standard `source .venv/bin/activate` is sufficient.

## Documentation and change logs

- `AGENTS.md` is a symlink to this file, so project agent rules should be written here.
- Every code change must be recorded in a project change log.
- Chinese change logs live under `docs/zh-cn/`.
- Small changes should be appended to the latest `docs/zh-cn/change-log-YYYY-MM-DD.md`.
- Independent topics, multi-day changes, or cross-module changes should get a new `docs/zh-cn/change-log-YYYY-MM-DD.md`.
- New change logs must be linked from `docs/zh-cn/README.md`.
- Each change-log entry should include:
  - date,
  - files touched,
  - what changed,
  - implementation principle,
  - intended functionality,
  - validation or why validation was not run,
  - known limitations and follow-ups.
- Documentation-only and comment-only changes should still be recorded briefly unless the user explicitly asks not to update documentation.

## Current project notes

- On the remote training/inference server, do not default to `uv run`; dependency syncing has previously failed because the flash-attn download source was unstable. Use the activated venv `python` directly.
- In each new remote shell, set the venv cuDNN path before training or inference:

```bash
source /root/gpufree-data/Isaac-GR00T/.venv/bin/activate
export LD_LIBRARY_PATH=/root/gpufree-data/Isaac-GR00T/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
```

- The current AgiBot G01 task controls dual-arm 14 joints plus left/right grippers. Do not control base, head, or waist for the fixed-point task.
- If a checkpoint was trained with `config_subtask.py`, the client observation language key must be `sub_task`, not `annotation.human.task_description`.
