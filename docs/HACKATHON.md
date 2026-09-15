# Hackathon plan

## Goal

Train a small language model that can help with common high-energy-physics
analysis work. A user should be able to give it a new prompt, such as asking
what is inside a ROOT file, how to change a TRExFitter config, how to run a
fit, or how to explain a result.

The final model must work in these two ways:

- through a simple model endpoint or command that accepts a human-readable
  request and returns the requested TRExFitter config snippet directly;
- behind each model family's native coding-agent harness.

For the agent version, Qwen uses Qwen Code's native tool vocabulary and its
checkpoint chat template. Codex and OpenCode retain their own native tools.
The user prompt and external scorer are identical across harnesses; tool
descriptions and message envelopes follow each harness.

## What we will build

We will first make several small, checked datasets. Each example is one user
request and the agent's complete response to it, including any tool calls and
tool results. The first datasets cover:

- changing and checking TRExFitter configs;
- reconstructing a complete missing config block from a natural-language description and nearby patterns;
- looking inside ROOT files and describing their contents;
- answering questions about ATLAS Open Data;
- running TRExFitter and explaining its output.

For supported config tasks, we will make a Qwen coding-agent version and a
direct config version of every reviewed task. Codex and OpenCode may each
provide verified source trajectories, but both are normalized to the same
Qwen tool contract. The direct version contains a natural-language request
and only the config snippet as its answer; it has no tool calls. All variants
of one task stay in the same train, validation, or test split.

Once the small datasets work, we will combine them into longer tasks, such as:

```text
inspect a ROOT file → write a config → run TRExFitter → explain the fit
```

## Checks before training

Every example must pass the checks that apply to it. We need two checkers:

1. A Qwen-interface verifier: confirms the canonical tool schema, call/result
   pairing, and rendering with the selected checkpoint tokenizer.
2. A TRExFitter-config verifier: confirms that a changed config is valid and,
   when needed, that TRExFitter can run it. For direct responses, it first
   inserts the snippet into the task's documented template.

The second checker saves a clear pass/fail result, errors, logs, time used, and
the significance when a fit is run. We will test both checkers with known good
and bad examples before using a dataset for training.

## Work plan

1. Pick a small part of the TRExFitter config language and create simple
   correct and broken examples from the H→γγ fixture.
2. Build the ROOT-file, Open Data, TRExFitter-running, and result-explanation
   datasets.
3. Find a faster histogramming path, either through another backend or by
   using data that is already histogrammed.
4. Define the Qwen Code-native SFT format plus the direct-config version and
   source-harness provenance converters.
5. Build and test the two checkers.
6. Train Qwen 1.5B and Qwen 7B with the checked datasets using verl.
7. Test held-out tasks with untrained Qwen, trained Qwen, and strong reference
   models.
8. Package the trained model for prompt-based use and for Codex/OpenCode use.
9. Only after supervised training works, try reinforcement learning using the
   pinned TRExFitter execution result as the score. SFT admission uses only the
   deterministic task contract, preservation check, and static config verifier;
   runtime/physics equivalence is not required to construct the SFT records.

The detailed, assigned work is in [TASK_BOARD.md](TASK_BOARD.md).

## Final deliverable

The final demo should include:

- a trained small model exported in a reusable format;
- a simple way to send it different prompts (a local command or endpoint);
- a prompt endpoint that returns config snippets directly, plus working Codex
  and OpenCode integrations that can use the same model for tool calls;
- a small held-out test report showing which tasks it can complete; and
- the dataset and model versions, training command, and checker results needed
  to repeat the demo.

## What counts as success

For a held-out agent task, success means the model emits valid Qwen tool calls,
the harness adapter executes them correctly, and the run reaches the requested
result. For a held-out direct-config task, success means it returns
only an insertable config snippet that passes config validation in its template.
For a config task, that also means TRExFitter runs when required. For a fit
task, report whether the physics goal was met as well as whether the run
succeeded.
