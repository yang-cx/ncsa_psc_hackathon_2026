# Task board

Keep each dataset small and useful on its own; we will join them into longer
tasks later.

## Status key for assigned tasks

- ⚪ Not started
- 🟢 Complete
- 🔴 Blocked

Only assigned tasks use these circles. Unassigned tasks stay as empty
checkboxes until someone takes them.

## 1. Make the small datasets

| Owner | Status | Task |
| --- | --- | --- |
| Chengxi | ⚪ | Turn the TRExFitter documentation into H→γγ operational tasks. |
| Joshua | ⚪ | Turn ATLAS Open Data documentation into small tasks. |
| Dongwon | 🟢 | Make ROOT-file tasks: inspect objects and variables, then describe them in plain language. |

- [ ] Choose the first small part of the TRExFitter config language to support.
- [ ] Make simple config tasks: a physics goal, a starting config, a known good answer, and broken examples.
- [x] Add single-block reconstruction development tasks: remove one Fit, Region, Sample, or NormFactor block, describe it in natural language, and score restoration plus preservation. Keep the final reconstruction test on a private alternate fixture.
- [ ] Make small tasks for running TRExFitter and reading its results.
- [ ] Keep training, validation, and final test tasks separate from the start.
- [ ] Find a faster way to histogram data: evaluate a new backend and already-histogrammed input data.

## 2. Use Qwen Code tools, two modalities, and two verifiers

| Owners | Status | Task |
| --- | --- | --- |
| Joshua, Chengxi | ⚪ | Agree on one basic record format for every dataset: task ID, dataset name, modality, split, starting files, available tools, tool calls and their results when agentic, final config snippet when direct, and check result. |

- [ ] From each supported config task, make a Qwen-agent version and a direct natural-language-to-config version. Keep every source trajectory with the same logical task ID and split.
- [x] Configure and test the [native tool contracts](TOOL_CONTRACT.md): Qwen Code-native functions, Qwen checkpoint chat-template rendering, and native Codex/OpenCode comparisons. No MCP or domain-specific wrapper.
- [ ] Define the direct-config prompt template and snippet insertion context. Its target must contain only a valid config snippet and its `tools` value must be `[]`.
- [ ] Build a verifier that checks the canonical semantic record and its exact Qwen checkpoint rendering.
- [ ] Build a verifier that checks whether a TRExFitter config is valid and can run when needed, including a direct snippet after insertion into its documented template. Save a clear pass/fail result, error message, time limit, log, and—when applicable—significance.
- [ ] Test both verifiers and both output modalities on a few known good and bad tasks before publishing data.

## 3. Release, join, and train datasets

| Owner | Status | Task |
| --- | --- | --- |
| Joshua | ⚪ | Convert the checked records into the files verl needs and provide one simple training command for Qwen 1.5B and Qwen 7B. |
| Joshua | ⚪ | Compare untrained Qwen 0.8B/9B, trained Qwen 0.8B/9B, and strong reference models on held-out tasks. Report Qwen-agent and direct-config results, plus Codex/OpenCode adapter results where deployed. |

- [ ] Release each checked one-turn dataset separately.
- [ ] Join released datasets into a separate long task: read a ROOT file → write a config → run TRExFitter → explain the result. Keep the source dataset and split recorded for every step.
- [ ] Add RL only after SFT works, using the final result from the verifier.

## 4. Score physics results

| Owners | Status | Task |
| --- | --- | --- |
| Charlie, Dongwon | ⚪ | Use TRExFitter run logs and artifacts to come up with physically meaningful metrics/rewards to evaluate the quality of the config. Decide which fit outputs show a healthy result: run status, significance, yields, uncertainties, pulls, and correlations. Turn those outputs into one simple score that rewards a healthy fit meeting the goal and penalizes failures or unstable results.|
- [ ] Test the score on known good and bad configs, and save the full report before sending its final score to RL.

## 5. Later: Harbor data

- [ ] Add the Harbor export under `data/harbor/`, choose useful examples, and keep final test examples out of training.
