# TRExFitter hackathon

We are building a starting point for teaching a small language model to operate a high-energy-physics analysis environment through agentic tools.

We will release several small, verified one-turn datasets: TRExFitter config work, ROOT-file inspection and modification, ATLAS Open Data knowledge, and execution/result interpretation. Each supported config task has two semantic renderings: a Qwen Code-native coding-agent trajectory plus a direct natural-language-to-config response. Codex and OpenCode may supply verified source runs, but Qwen SFT calls use Qwen Code's own tool names and arguments. The selected checkpoint tokenizer—not the dataset—creates Qwen-specific tokens. All trajectory variants share a logical task ID and split. Compatible family releases are then composed into a separate long-horizon dataset.

Our first starting point is the working H→γγ config at [data/configs/examples/hyy.config](data/configs/examples/hyy.config).

## Clone it

`verl` is retained as a Git submodule for the later RL stage. Clone it with the repository:

```bash
git clone --recurse-submodules git@github.com:JO5HO4/ncsa_psc_hackathon_2026.git
```

For an existing clone, initialize the dependency:

```bash
git submodule update --init --recursive
```

Submodules use public HTTPS URLs by default. Contributors who prefer SSH can
override them locally; see [the ATLAS workflow](docs/ATLAS_WORKFLOW.md).

`bash training/bootstrap_verl.sh` remains available when only the training
dependency needs to be initialized. Download the reference training Parquet
files directly from Hugging Face when needed:

```bash
bash data/fetch_reference_datasets.sh
```

## Host-side Python environment

Use the committed `pyproject.toml`, `uv.lock`, and Python 3.11 pin for host-side
analysis tooling. On NERSC:

```bash
module load python
uv sync --locked
uv run --locked pytest -v --color=yes
```

Default dependencies are NumPy, Awkward, and Pydantic, plus pytest in the
development group. Optional extras separate ROOT-file metadata (`inputs`),
histogramming (`coffea`), and the optional ATLAS collection schema
(`atlas-schema`). Select extras explicitly in both `uv sync` and `uv run`.
These dependencies do not install ROOT or TRExFitter; those remain in the
existing container. Active SFT and later RL use the pinned VERL environment
documented in `training/README.md`.
The local `.venv/` is ignored; commit dependency changes with the lockfile.

## Main folders

| Folder | What it is for |
| --- | --- |
| [data/](data/README.md) | Dataset families, fixtures, and publishing conventions |
| [trex_fitter/](trex_fitter/) | The code that checks and runs configs |
| [training/](training/README.md) | Qwen native-agent SFT and later RL with VERL |
| [inference/](inference/README.md) | Testing a trained model |
| [docs/ATLAS_WORKFLOW.md](docs/ATLAS_WORKFLOW.md) | Reproduce the ATLAS command benchmark |
| [Hyy TRExFitter agent dataset](data/datasets/hyy-trexfitter-agent-trajectories/README.md) | Replay-approved Qwen Code-native trajectories, Parquet mirrors, and dataset audit tools |
| [docs/](docs/HACKATHON.md) | The plan and task list |

Start with the [hackathon plan](docs/HACKATHON.md), then pick a task from the [task board](docs/TASK_BOARD.md).
