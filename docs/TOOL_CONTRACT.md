# Native coding-agent tool contracts

The user task and external scorer are harness-independent. The model-facing
tool contract is not: Codex uses Codex tools, OpenCode uses OpenCode tools, and
Qwen3.5 is trained and evaluated with Qwen Code's native tools. Raw teacher
events remain immutable provenance and are translated into Qwen Code calls
only when constructing Qwen SFT examples.

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

## Qwen Code native tools

Every Qwen TRExFitter coding-agent row exposes this small subset of Qwen Code:

| Tool | Purpose |
| --- | --- |
| `read_file` | Read a file with Qwen Code's `file_path`, `offset`, and `limit` arguments. |
| `edit` | Replace `old_string` with `new_string` in `file_path`. |
| `run_shell_command` | Run a command with the required `is_background` decision. |

These are Qwen Code function names and argument shapes, not repository-owned
aliases. Tool descriptions may change with the pinned Qwen Code release; the
checkpoint chat template renders the structured calls. `config_verify` remains
an ordinary shell command. No MCP server or TRExFitter-specific function is
exposed.

## Source-harness adapters

The capture exporter records the source harness and translates only the call
envelope. Observable results are retained from the raw event stream.

| Source event | Canonical Qwen call |
| --- | --- |
| Codex `command_execution` | `run_shell_command` |
| Codex `file_change` | `edit` reconstructed from the verified before/after files |
| OpenCode `bash` | `run_shell_command` |
| OpenCode `read` | `read_file` |
| OpenCode `glob`, `grep`, or `list` | bounded `run_shell_command` inspection |
| OpenCode `edit` | `edit` |

At inference there is no reverse-mapping adapter. Qwen Code sends its own tool
schemas to the OpenAI-compatible endpoint, parses the model's native calls,
executes them, and emits native tool-result messages. Codex and OpenCode are
run separately through their own CLIs with the exact same user prompt.

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
  normalization, including a reconstructed Codex exact-replacement edit.
- Train, validation, and test are split by `logical_task_id`, never by sampler
  or trajectory variant.
