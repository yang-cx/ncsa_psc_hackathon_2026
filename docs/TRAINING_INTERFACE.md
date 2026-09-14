# Training and testing plan

## What we will train

We will train selected Qwen3.5 checkpoints on the datasets made in this project:

- Qwen3.5 0.8B for fast iteration
- Qwen3.5 9B for the larger comparison

Each supported config task has two semantic versions: coding-agent
and direct natural-language-to-config. Codex and OpenCode are source samplers,
not different training formats. The direct version teaches the prompt endpoint
to return only a config snippet, without tools. All trajectory variants of a
task remain in the same data split.

## What we will compare

Use the same held-out tasks to compare three groups:

| Group | Models |
| --- | --- |
| Untrained baseline | Qwen3.5 0.8B and 9B before training on our data |
| Trained models | The same Qwen3.5 checkpoints after training on our data |
| Strong reference models | Available state-of-the-art models, tested without training on our data |

For every model, report results separately for each dataset family and
modality: Qwen agent and direct config. Deployment may additionally be broken
down by Codex and OpenCode adapter. Do not use any held-out task, or any of its
trajectory variants, during training.

## What the training data needs

The dataset builder turns each reviewed task into model-neutral conversational
JSONL. Agent records include the user request, generic
tools, tool calls and their results, and the final answer. The selected Qwen
checkpoint tokenizer renders the native control tokens. A direct-config record
includes
a human-readable user request and one assistant answer containing only the
reviewed config snippet; its `tools` column is `[]`. The builder makes one
Qwen task record and one direct-config record; it may retain multiple verified
source trajectories with the same logical task ID for controlled augmentation.

People review the source tasks and the results from the two verifiers.
Canonical records contain no tokenizer tokens or loss masks. VERL reads an
Arrow-stable Parquet materialization, passes `enable_thinking=false` to Qwen
only at rendering time, and creates assistant-only loss masks from the
structured turns. A hard preflight proves that VERL's turn-wise rendering
matches the selected tokenizer's complete-conversation rendering.

## What to save

For each run, save the model name and size, resolved model/tokenizer revision,
chat-template hash, installed VERL/Transformers
versions, dataset version, training command,
and test results. For each test task, save whether an agent output has valid
Qwen tool syntax, whether its harness adapter executed it, and, for config
tasks in every modality, whether the TRExFitter config is valid and runs when
required. Validate a direct response
by inserting the snippet into its documented template.

Reinforcement learning is a later step only if this supervised-training
comparison works first.
