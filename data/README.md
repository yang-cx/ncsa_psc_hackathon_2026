# Data

`reference/` holds the two reference datasets after they are downloaded from
Hugging Face. They are not stored in this repository. Fetch the needed Parquet
files with:

```bash
bash data/fetch_reference_datasets.sh
```

TRExFitter data is split by role:

| Location | Contents |
| --- | --- |
| [`configs/examples/`](configs/examples/) | Fifteen runnable, versioned TRExFitter `.config` examples. |
| [`samples/examples/`](samples/examples/) | Shared TRExFitter example samples. |
| [`samples/hyy/`](samples/hyy/) | Large H→γγ `Data/` and `MC/` ROOT inputs, which are not versioned. |

`configs/` is the first of several small dataset families. Verified Codex and
OpenCode runs may provide teacher provenance, but the active Hyy SFT release
normalizes accepted trajectories into a Qwen Code view. A separate direct
natural-language-to-config modality may use an insertable config snippet with
an empty `tools` list; it is not part of the current main-agent mixture. Other
families will cover ROOT-file work, ATLAS Open Data knowledge, TRExFitter
execution, and fit-result interpretation. Each family owns small reviewed
source-task records, fixtures, and a verifier; family releases are later
combined into a separate long-horizon dataset.

The first starting config is `configs/examples/hyy.config`. It is a working H→γγ TRExFitter config and should be the base for our first example tasks.

`harbor/` is reserved for Harbor data when it arrives.

For now, `configs/examples/` contains only runnable TRExFitter configs. Task
records, schemas, and generated training splits are maintained separately and
are not part of this repository layout.

The Hyy Qwen training view uses the
[`qwen-code-native-tools/v1`](../docs/TOOL_CONTRACT.md) contract. Native Codex
and OpenCode evaluations use their own harness interfaces while keeping the
user task and external scorer fixed. Do not create one-off domain tools, MCP
servers, or custom wrappers for individual dataset families.

## Dataset publishing convention

Treat a Hugging Face dataset repository as the canonical home for every
reviewed dataset we create: individual task-family releases, Codex/OpenCode
agent and direct-config SFT renderings, RL prompts, held-out evaluation tasks,
merged long-horizon tasks, and inference prompt sets. Keep only schemas, small examples, and download
scripts in this Git repository. Large or generated dataset files should be
published to the Hub and fetched by dataset ID and revision.

For inference, a Hub dataset can use any string prompt column; pass its name to
`inference/run_prompts.py --prompt-field`. The dataset's README should state
the family, schema, agent harness/template versions, direct prompt-template
version, tool-manifest revision, splits, source fixture version, verifier,
intended use, and immutable commit or tag to use for evaluations.

When a reviewed local split is ready, publish it with a logged-in Hugging Face
account instead of committing it here:

```bash
python data/publish_dataset.py \
  --source data/configs/splits/train.jsonl \
  --repo-id ho22joshua/trex-config-tasks \
  --split train
```

Run the command once for each split. It accepts `.json`, `.jsonl`, and
`.parquet`; use `--config-name` when one dataset repository hosts multiple
schemas.

### Hyy TRExFitter agent trajectories

The current native-tool TRExFitter trajectory release is the public Hugging
Face dataset [`cxyang-ucb/hyy-sft`](https://huggingface.co/datasets/cxyang-ucb/hyy-sft).
It is registered as a Git submodule, matching
`data/datasets/root-sft-dataset`, so the parent repository can pin an exact
dataset revision. Its dataset card documents the replay gate, splits, Qwen
Code-native three-tool contract, and Parquet loading interface.

```bash
git submodule update --init \
  data/datasets/hyy-trexfitter-agent-trajectories
```

The dataset is public, so HTTPS checkout does not require authentication. A
contributor who needs to push can select an SSH URL locally without editing
`.gitmodules`:

```bash
  git config \
  submodule.data/datasets/hyy-trexfitter-agent-trajectories.url \
  git@hf.co:datasets/cxyang-ucb/hyy-sft
```

Dataset builders, schemas, authoring tests, and release tools live in the
dataset repository itself. They generate ignored `.build/` workspaces; hidden
gold states and sealed test trajectories remain local and are never included
in the public release payload.

## ATLAS Open Data

Use `fetch_atlas_opendata.sh` to download complete Open Data skims directly
from `ho22joshua/atlas_opendata`. Files are stored under `atlas_opendata/`,
preserving the Hub layout, and are ignored by git.

```bash
# One 2025 skim
bash data/fetch_atlas_opendata.sh 2to4lep

# The 2025 diphoton collection
bash data/fetch_atlas_opendata.sh GamGam

# Every available ROOT file; this is a very large download
bash data/fetch_atlas_opendata.sh all
```

The H→γγ fixture needs only a small diphoton subset. Use
`bash trex_fitter/scripts/fetch_hyy_inputs.sh` for that instead of downloading
the full GamGam skim.
