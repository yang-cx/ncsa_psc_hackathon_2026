# TRExFitter hackathon

We are building a starting point for teaching a small language model to operate a high-energy-physics analysis environment through agentic tools.

We will release several small, verified one-turn datasets: TRExFitter config work, ROOT-file inspection and modification, ATLAS Open Data knowledge, and execution/result interpretation. Each supported config task has three aligned renderings: Codex and OpenCode tool-use harnesses, plus a direct natural-language-to-config response for the model endpoint. The direct response contains only an insertable config snippet; the agent versions use bounded tools to inspect, repair, validate, and run. All sibling renderings share a logical task ID and split. Compatible family releases are then composed into a separate long-horizon dataset.

Our first starting point is the working H→γγ config at [data/configs/examples/hyy.config](data/configs/examples/hyy.config).

## Clone it

`verl` is a Git submodule. Clone it with the repository:

```bash
git clone --recurse-submodules git@github.com:JO5HO4/ncsa_psc_hackathon_2026.git
```

For an existing clone, initialize the dependency:

```bash
git submodule update --init --recursive
```

## Host-side Python

Host-side analysis tools use `uv`; TRExFitter and ROOT remain in the pinned
container. For static config verification:

```bash
module load python
uv sync --locked
uv run --locked python -m trex_fitter.coffea_backend.verify \
  data/configs/examples/hyy.config
```

On a compute node, install and use the optional Coffea backend with:

```bash
uv sync --locked --extra coffea
uv run --locked --extra coffea python trex_fitter/runner.py \
  data/configs/examples/hyy.config --backend coffea --actions n
```

Add `--extra atlas-schema` to both commands only when that optional schema is
needed.

`bash training/bootstrap_verl.sh` remains available when only the training
dependency needs to be initialized. Download the reference training Parquet
files directly from Hugging Face when needed:

```bash
bash data/fetch_reference_datasets.sh
```

## Main folders

| Folder | What it is for |
| --- | --- |
| [data/](data/README.md) | Dataset families, fixtures, and publishing conventions |
| [trex_fitter/](trex_fitter/README.md) | The code that checks and runs configs |
| [training/](training/README.md) | Training with verl |
| [inference/](inference/README.md) | Testing a trained model |
| [docs/](docs/HACKATHON.md) | The plan and task list |

Start with the [hackathon plan](docs/HACKATHON.md), then pick a task from the [task board](docs/TASK_BOARD.md).
