# Dataset design

## Dataset programme

We are building a collection of small, independently reviewable **one-turn datasets**, not one monolithic TRExFitter-config dataset. Each supported configuration task is published in two aligned modalities: a Qwen coding-agent episode and a direct natural-language-to-config response. Codex and OpenCode may both sample candidate trajectories, but they are provenance sources rather than different model-facing modalities. Initial families include:

- TRExFitter configuration repair, synthesis, and explanation.
- ROOT-file inspection and modification.
- ATLAS Open Data knowledge and analysis tasks.
- TRExFitter execution, artifact inspection, and result interpretation.

Here, *one turn* means one user task and one bounded response. An agent response may contain several tool calls, their results, and a final answer; a direct response is a single config snippet. Each family has its own source-task schema, fixtures, expected outcome, and train/validation/test split, and uses the two shared project verifiers. Keep final test tasks out of every training family and preserve the split when families are combined.

Later, compose compatible one-turn episodes into long-horizon tasks—for example, inspect a ROOT file, write a config, run TRExFitter, and interpret the fit. Those composed tasks are a distinct dataset with their own held-out evaluation, not a replacement for the component datasets. They are the eventual basis for long-horizon SFT and RL.

## Two target modalities

The trained Qwen model has two complementary uses: operation through any harness that implements the canonical coding-tool adapter, and a prompt endpoint that turns a human-readable request into a TRExFitter config snippet.

An agentic record contains:

- a system instruction that establishes the task boundary and tool-use rules;
- the user request and any supplied workspace context;
- assistant tool-call turns, with stable call IDs and JSON arguments;
- tool-result turns that refer to their call ID; and
- a concise final assistant answer, or a justified no-action answer.

Every agentic record carries the single, versioned Qwen tool manifest defined in [TOOL_CONTRACT.md](TOOL_CONTRACT.md). The canonical task declares required capabilities—`read`, `modify`, and `execute`—and the SFT exporter represents them as `shell`, `read_file`, `search_files`, and `apply_patch`. Codex and OpenCode names are normalized at the dataset boundary and retained only in raw provenance. Do not train on invented tool output or a hidden verifier result.

The direct `direct_config` modality is not an abbreviated agent transcript. Its prompt states the physics goal, supported config subset, constraints, and necessary local context. Its assistant target contains only the reviewed, self-contained config snippet: no tools, diff fences, explanation, or verifier output. The `messages` field uses the same typed chat structure as agent rows and `tools` is an empty list (`[]`). A consumer can insert the snippet at the documented location in a config template.

### Canonical source and aligned renderings

Author and review one canonical config task, then render it two ways:

| Field | Coding agent | Direct config |
| --- | --- | --- |
| `logical_task_id` | source task ID | same source task ID |
| `modality` | `coding_agent` | `direct_config` |
| `capabilities` | `read` / `modify` / `execute` | `[]` |
| `tools` | generic four-tool manifest | omitted or `[]` |
| `messages` | OpenAI-style semantic tool trajectory | natural-language request followed by one config-only answer |
| split, fixture revision, config verification | preserved | preserved |

The target checkpoint tokenizer owns protocol-specific tokens and message-template details. Do not put literal Qwen XML, Codex wrappers, OpenCode wrappers, token IDs, or loss masks in the semantic source record. Candidate source traces can differ, but after normalization they expose the same generic tool manifest and must reach the same reviewed final state. Multiple accepted trajectories for one task are variants, not independent examples; balance and split by `logical_task_id`, never by sampler or rendered row ID.

Before publishing an agent trajectory, replay or externally verify its source run against the fixture, normalize it to the canonical contract, and preflight it through the exact target Qwen checkpoint tokenizer. Before publishing a direct renderer, insert its target snippet into the documented fixture/template and run the config verifier. Store source-harness, source-model, tokenizer revision, and prompt-template provenance.

## Common dataset shape

All published family datasets and the eventual merged dataset should expose the following common columns, in addition to family-specific fields:

```text
id                  unique rendered-record ID, e.g. trex-repair-001--qwen-agent
logical_task_id     ID shared by all renderings of one authored task
dataset_family      trex_config | root_io | open_data | trex_execution | ...
modality            coding_agent | direct_config
source_harness      codex | opencode | generated | none (provenance only)
tool_contract       canonical-code-tools/v1 | none
capabilities         subset of read, modify, execute; [] for direct_config
split               train | validation | test
messages             typed chat/tool trajectory
tools                optional structured generic tool manifest
fixture              immutable input/environment revision
verification         verifier name, status, and non-secret evidence
provenance           source, authoring/review, and renderer metadata
```

Use typed semantic chat records: assistant tool calls have a function name and object-valued JSON arguments; a tool result has the matching `tool_call_id`. Keep JSONL as the canonical representation and publish verified Parquet mirrors for Hugging Face. The local VERL transport may JSON-encode heterogeneous Arrow leaves, but its loader must restore object-valued arguments before tokenization. In a `direct_config` record, the assistant has one final message containing only the config snippet and no tool-call messages. The active TRExFitter files are exported under `artifacts/native-sft/`; the selected tokenizer renders them only in VERL's transient training view.

## First family: TRExFitter configuration repair

The first task is:

```text
physics goal + starting .config file + diagnostic
                         ->
                 bounded agent repair episode
```

Start with `data/configs/examples/hyy.config`, our working H→γγ config. A repair task declares `read`, `modify`, and `execute` capabilities. Every Qwen rendering exposes the same canonical generic tools. Source harnesses invoke task-provided validation, TRExFitter, and result-inspection commands only in the sandbox; the exporter maps their calls to the canonical contract. The model should use supplied tools rather than emit an out-of-band patch answer.

For every supported repair or synthesis task, also create the direct rendering. Phrase its request for a human, provide only necessary local context, and target the smallest valid config block—not a patch or the complete fixture file.

The reviewed source-task schema is maintained with the released task dataset rather than this config-only directory. A dataset builder adds the common columns and produces both model-facing renderings. Later config synthesis and multi-edit repair fit this same family; ROOT and Open Data tasks should get small family-specific source schemas rather than being forced into a TRExFitter-config record.

### Single-block reconstruction

The native-development layer also includes a distinct `block_reconstruction`
objective. A *block* is one named TRExFitter section, such as `Sample: "WH"`,
together with all of its indented settings. The fixture is made by removing
exactly one complete block from an otherwise valid config. The user prompt
describes the missing block's physics role, exact identifiers and unique input
values, plus relationships that can be inferred from sibling blocks. It does
not paste the target block or prescribe a patch.

These tasks teach more than local value replacement. The agent must:

1. inspect the surrounding text with normal coding tools;
2. identify the appropriate TRExFitter block syntax and reusable patterns;
3. synthesize all required settings from the prose and local evidence;
4. edit only `analysis.config`; and
5. run the ordinary config-verification command through `shell`.

The current development catalogue covers Fit, Region, Sample, and NormFactor
blocks. Region tasks exercise structured selection reconstruction; Sample
tasks combine unique filenames with shared MC conventions; NormFactor tasks
exercise cross-references between samples and regions. Job-block removal is
deferred because removing the sole global Job block gives too little local
evidence and makes the prompt mostly a transcription exercise.

Scoring is setting-level rather than a whole-file byte comparison. Every
parsed setting in the omitted block must equal its reviewed expected value,
the final config must pass the static verifier, and every parsed key outside
that block must remain unchanged. Block position and surrounding whitespace
are not scored. The public train and validation partitions are disjoint by
missing block identity, so a
paraphrase of the same target cannot cross splits. No public reconstruction
task is labeled as a sealed test: final evaluation must remove unseen blocks
from a private alternate config whose answers are absent from the repository.

Expression-valued settings need stronger treatment than ordinary strings.
The scorer first accepts exact values, then applies a conservative expression
normal form that proves identities such as `A > B` versus `B < A`, reordered
Boolean terms, redundant parentheses, and reordered addition or multiplication.
It records when this equivalence path was used. It does not guess at arbitrary
algebraic equivalence. This proof-based normalization, contract checking,
unrelated-setting preservation, and `config_verify` form the deterministic SFT
acceptance gate. SFT records do not need a TRExFitter execution reward.

Histogram-level operational equivalence belongs entirely to RL. Run both the
candidate and reference configs with the pinned TRExFitter environment and the
same private inputs. Require matching output-object coverage and compare every
per-region/per-sample nominal histogram: axis definitions, regular and flow-bin
contents, and sum-of-weights-squared variances. Then compare the requested fit
artifacts and health checks. Matching only total yields is insufficient because
different selections can accidentally integrate to the same count. Agreement
on finite inputs is operational evidence rather than a mathematical proof, so
retain the input revision and numerical tolerances with every reward. Do not
use a synthetic probe evaluator as a substitute for the authoritative
TRExFitter run.

## Verification and rewards

Each task must state its fixture/environment revision, permitted actions, and
expected observable outcome. The project has exactly two verifiers:

1. a Qwen-interface verifier, which checks the semantic tool schema, call/result
   pairing, canonical tool names, and rendering through the target checkpoint's
   `apply_chat_template`; and
2. a TRExFitter-config verifier, which checks that a changed config is valid
   and runs when the task requires a run. For a direct rendering, it first
   inserts the snippet into its documented template.

ROOT inventories, documentation answers, and fit results are expected task
outcomes saved as evidence; they do not create separate verifier projects.

The stages deliberately use increasingly strong checks:

1. **SFT deterministic check:** the requested contract is satisfied, unrelated
   settings are preserved, and the config passes the static verifier.
2. **RL execution check:** the pinned StatAnalysis container runs the candidate
   and reference with identical inputs and their requested artifacts agree.
3. **RL physics result:** the run is healthy and meets its stated goal, such as
   a target significance, without rewarding needless config complexity.

The static verifier catches cheap structural errors before a run, but the pinned
StatAnalysis container remains the authority for workspace construction and
physics results. For later RL, reward a healthy successful fit and the stated physics
goal, and penalize failures, instability, missing output, or needless
complexity.

## Publishing and merging

Keep reviewed source records as JSON and publish generated, versioned dataset splits to the Hub. Do not hand-edit Parquet. A dataset README must name its family, source schema, Qwen checkpoint/tokenizer revisions, source-harness adapter versions, direct prompt-template version, tool manifest revision, fixture revision, split policy, verifier, and intended use.

The merge step takes selected released family versions, validates that their common columns and modality contracts are compatible, checks every model-facing tool manifest against `canonical-code-tools/v1`, balances families and modalities deliberately, and writes a manifest of every source dataset and revision. It retains family, `logical_task_id`, modality, source harness, capabilities, split, provenance, and verification fields so contamination checks and per-family evaluation remain possible. Model, tokenizer, template, and package revisions belong to each training-run manifest rather than the canonical rows.
