# Qwen coding-agent tool contract (v1)

The model-facing contract is independent of the trajectory-collection harness.
Codex and OpenCode are samplers, not separate SFT modalities. Their raw event
streams remain immutable provenance and are normalized into one semantic
record before training Qwen3.5.

## Qwen training representation

Each record contains an OpenAI-style `tools` array and chronological
`system`, `user`, `assistant`, and `tool` messages. Assistant calls use
`tool_calls[].function.name` and object-valued `arguments`; tool results carry
the matching `tool_call_id`. We do not write Qwen XML into JSONL.

At tokenization time, pass both `messages` and `tools` to the exact target
checkpoint's `apply_chat_template`. Qwen3.5 then renders its preferred
`<tool_call><function=...><parameter=...>` syntax and converts tool-result
messages to `<tool_response>` content. This keeps source data semantic and
lets the checkpoint tokenizer own special tokens and template revisions.

The initial native-agent study uses `enable_thinking=false`. Planning text and
tool calls are supervised, but proprietary hidden reasoning from a sampling
model is never treated as a Qwen reasoning trace. A thinking-mode study needs
separately generated, reviewed Qwen-compatible `reasoning_content`.

## Canonical generic tools

Every TRExFitter coding-agent row exposes the same small tool manifest:

| Tool | Purpose |
| --- | --- |
| `shell` | Run a command in the isolated task workspace. |
| `read_file` | Read a workspace-relative file, optionally with an offset and limit. |
| `search_files` | Search contents, path globs, or a directory. |
| `apply_patch` | Apply a bounded patch to an allowlisted workspace file. |

These names are our versioned application interface; Qwen itself does not
mandate function names. They are deliberately generic coding operations, not
TRExFitter APIs. `config_verify` and the TRExFitter runner are ordinary shell
commands, not model-facing functions. No MCP server, `list_blocks`,
`search_settings`, `read_config`, or `verify_config` is exposed.

## Source-harness adapters

The capture exporter records the source harness and translates only the call
envelope. Observable results are retained from the raw event stream.

| Source event | Canonical Qwen call |
| --- | --- |
| Codex `command_execution` | `shell` |
| Codex `file_change` | `apply_patch` reconstructed from the verified before/after files |
| OpenCode `bash` | `shell` |
| OpenCode `read` | `read_file` |
| OpenCode `glob`, `grep`, or `list` | `search_files` with an explicit mode |
| OpenCode `edit` or `apply_patch` | `apply_patch` |

At inference, a harness adapter performs the reverse mapping: it parses
Qwen3.5's output through the checkpoint-supported Qwen tool parser, validates
the canonical JSON arguments, and invokes the corresponding native harness
operation. The trained model therefore sees one protocol even when deployed
behind different harnesses.

## Sandbox and acceptance rules

- Every command runs in a private fixture-derived workspace. Only
  `analysis.config` may be modified.
- Agent-visible network, web, MCP, subagent, credential, and external-directory
  access is disabled. Provider transport and authentication remain outside the
  task sandbox.
- Paths are normalized workspace-relative. Commands, reads, outputs, errors,
  exit codes, and edits are bounded before release.
- A raw trajectory is eligible only after the agent exits and an external
  scorer verifies the requested semantic change, unrelated-setting
  preservation, and required config evidence.
- Raw source events and their hashes are retained. The export records every
  normalization, including a reconstructed Codex patch.
- Train, validation, and test are split by `logical_task_id`, never by sampler
  or trajectory variant.
