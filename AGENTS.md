# Repository Guidelines

## Project Structure & Module Organization
This repository is organized around simulation and training scripts. Core environment and shared helpers live at the root: `env.py`, `dynamics.py`, `simulation.py`, `utils.py`, and `params.py`. Algorithm-specific code is grouped in `DDPG/`, `GPS/`, and `Linear/`. Entry-point scripts such as `trainDDPG.py`, `trainGPS.py`, `try_rollout.py`, and `test_noise.py` run experiments and evaluations. Plotting utilities live in `tests/`, but they are analysis scripts rather than an automated test suite. Generated artifacts such as `results_*`, `*.npz`, and `*.pth` are intentionally ignored by Git.

## Build, Test, and Development Commands
Use the environment files already checked in:

- `conda env create -f env.yml` creates the recommended Python 3.9 environment.
- `pip install -r requirements.txt` installs the pip-based dependency set.
- `python3 trainDDPG.py` runs DDPG training and writes outputs under `results_ddpg/`.
- `python3 trainGPS.py` runs Guided Policy Search training and writes outputs under `results_gps/`.
- `python3 test_noise.py` evaluates a saved policy under noisy and noiseless rollouts.
- `python3 tests/plot_results.py --file-array path/to/results.npz --name policy_cost` plots stored experiment arrays.

Some scripts expect precomputed checkpoints under ignored `results_*` folders; verify paths in `params.py` before running.

## Coding Style & Naming Conventions
Follow the existing Python style: 4-space indentation, module-level constants in `UPPER_CASE`, functions and variables in `snake_case`, and classes in `CamelCase` such as `QuadcopterEnv` and `DDPGagent`. Keep new scripts consistent with the repository’s script-first layout and prefer small, importable helpers over duplicated experiment code. There is no formatter or linter configured, so match surrounding style closely.

## Testing Guidelines
There is no formal `pytest` or `unittest` suite at present. Validate changes by running the relevant training or rollout script and checking generated plots, saved arrays, and final rewards. When adding new analysis scripts, keep names descriptive and runnable directly, following patterns like `test_noise.py` or `tests/plot_results.py`.

## Commit & Pull Request Guidelines
Recent commits use short, imperative, lowercase subjects such as `added expert` and `removed convex_hull`. Keep commit messages brief, focused on one change, and under roughly 60 characters when possible. Pull requests should state which algorithm or module changed, list any `params.py` edits, and attach the key evidence for behavior changes: reward curves, rollout plots, or output file paths. Do not commit ignored artifacts, credentials, or large checkpoint files.
