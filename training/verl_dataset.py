"""VERL dataset adapter for model-neutral native coding-agent trajectories.

The canonical release remains JSONL with structured ``messages`` and optional
OpenAI-style ``tools``. The VERL materialization stores tools and tool-call
arguments as JSON strings only to avoid unstable Arrow union schemas; this
adapter restores those objects before the selected tokenizer renders them.
"""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pandas as pd
import torch

from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
from verl.utils.py_functional import convert_nested_value_to_list_recursive
from verl.utils.tokenizer.chat_template import apply_chat_template, extract_system_prompt_and_generation


class TReXNativeToolSFTDataset(MultiTurnSFTDataset):
    """Load nested SFT records safely and decode their JSON-encoded tools."""

    def __init__(self, parquet_files, tokenizer, config, processor=None, max_samples=-1):
        self.require_native_tool_contract = bool(
            (config or {}).get("require_native_tool_contract", True)
        )
        super().__init__(parquet_files, tokenizer, config, processor=processor, max_samples=max_samples)

    @staticmethod
    def decode_tools(value):
        if isinstance(value, str):
            value = json.loads(value)
        if not isinstance(value, list):
            raise TypeError(
                "The tools column must contain a JSON array or Python list; "
                f"received {type(value).__name__}."
            )
        return value

    @staticmethod
    def decode_tool_call_arguments(messages):
        """Restore canonical object arguments after Arrow-safe JSON encoding.

        The checked-in JSONL follows the common Hugging Face tool-call shape,
        where ``function.arguments`` is an object.  Parquet stores that field
        as a string to avoid an unstable union of per-tool Arrow structs.
        Model templates such as Qwen3.5 require the object form at render time.
        """
        for message in messages:
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    decoded = json.loads(arguments)
                    if not isinstance(decoded, dict):
                        raise TypeError("Tool-call arguments must decode to a JSON object.")
                    function["arguments"] = decoded
        return messages

    def _read_files_and_process(self):
        # VERL's default ``dtype_backend="pyarrow"`` loader aborts when Pandas
        # indexes nested struct/list columns in the ATLAS command Parquet files.
        # Default Pandas objects preserve those values and support safe ``iloc``
        # access in the StatefulDataLoader.
        frames = [pd.read_parquet(path) for path in self.parquet_files]
        self.dataframe = pd.concat(frames, ignore_index=True)

        total = len(self.dataframe)
        print(f"dataset len: {total}")
        if 0 < self.max_samples < total:
            if self.shuffle:
                rng = np.random.default_rng(self.seed) if self.seed is not None else np.random.default_rng()
                indices = rng.choice(total, size=self.max_samples, replace=False)
            else:
                indices = np.arange(self.max_samples)
            self.dataframe = self.dataframe.iloc[indices.tolist()].reset_index(drop=True)
            print(f"selected {self.max_samples} random samples out of {total}")

        self.messages = [
            self.decode_tool_call_arguments(messages)
            for messages in self.dataframe[self.messages_key]
            .apply(convert_nested_value_to_list_recursive)
            .tolist()
        ]
        if self.tools_key in self.dataframe.columns:
            self.tools = [self.decode_tools(value) for value in self.dataframe[self.tools_key].tolist()]
        else:
            self.tools = None
        if self.require_native_tool_contract:
            self.validate_native_tool_contract()
        self.enable_thinking = (
            self.dataframe[self.enable_thinking_key].tolist()
            if self.enable_thinking_key in self.dataframe.columns
            else None
        )
        self.system_prompt, self.generation_prompt = extract_system_prompt_and_generation(
            self.tokenizer, **self.apply_chat_template_kwargs
        )

    def validate_native_tool_contract(self):
        """Require the model-neutral generic coding-tool contract."""
        required_columns = {"tool_contract"}
        missing_columns = required_columns - set(self.dataframe.columns)
        if self.tools is None or missing_columns:
            raise ValueError(
                f"native-tool SFT is missing tools or columns: {sorted(missing_columns)}"
            )
        allowed = {"shell", "read_file", "search_files", "apply_patch"}
        forbidden = {
            "list_blocks", "search_settings", "read_config", "verify_config",
            "bash", "read", "edit", "write", "glob", "grep", "list",
        }
        for index, (manifest, messages) in enumerate(zip(self.tools, self.messages, strict=True)):
            if self.dataframe.iloc[index]["tool_contract"] != "canonical-code-tools/v1":
                raise ValueError(f"row {index}: unexpected native tool contract")
            names = {item.get("function", {}).get("name") for item in manifest}
            if names & forbidden or names != allowed:
                raise ValueError(f"row {index}: invalid canonical tool manifest: {sorted(names)}")
            calls = [call for message in messages for call in (message.get("tool_calls") or [])]
            called = {call.get("function", {}).get("name") for call in calls}
            if not called <= names:
                raise ValueError(f"row {index}: tool call is absent from its manifest: {sorted(called - names)}")
            if any(not isinstance((call.get("function") or {}).get("arguments"), dict) for call in calls):
                raise ValueError(f"row {index}: tool arguments must be restored as JSON objects")
            call_ids = [call.get("id") for call in calls]
            result_ids = [message.get("tool_call_id") for message in messages if message.get("role") == "tool"]
            if None in call_ids or Counter(call_ids) != Counter(result_ids):
                raise ValueError(f"row {index}: tool calls and results are not paired by tool_call_id")

    def _build_messages(self, example):
        """Build a row and restore Arrow-encoded tool arguments used by it."""
        return self.decode_tool_call_arguments(super()._build_messages(example))

    def __getitem__(self, item):
        # VERL passes tools only while processing turn zero.  This adapter
        # renders cumulative context for every later turn, so retain the row's
        # tool schema for those renders as well.
        self._active_tools = self.tools[item] if self.tools is not None else None
        try:
            return super().__getitem__(item)
        finally:
            self._active_tools = None

    def _process_single_message(self, index, message, full_message, tools=None, enable_thinking=None):
        """Render each turn with enough preceding context for Qwen's template."""
        tools = getattr(self, "_active_tools", tools)
        # Qwen 3.5 rejects an isolated system message. It is included in the
        # following user/assistant contexts, so it needs no standalone tokens.
        if message["role"] == "system":
            empty = torch.empty(0, dtype=torch.long)
            return empty, empty.clone(), empty.clone(), {}

        processor = self.processor if self.processor is not None else self.tokenizer
        kwargs = {**self.apply_chat_template_kwargs}
        if enable_thinking is not None:
            kwargs["enable_thinking"] = enable_thinking

        context = full_message[: index + 1]
        inputs = dict(
            apply_chat_template(
                processor,
                messages=context,
                tools=tools,
                add_generation_prompt=False,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                **kwargs,
            )
        )
        input_ids = inputs.pop("input_ids")[0]
        attention_mask = inputs.pop("attention_mask")[0]

        # Keep only tokens introduced by this turn. The first user turn needs
        # the system context, while assistant and later user turns can subtract
        # their valid preceding conversation.
        prefix = full_message[:index]
        if any(turn["role"] == "user" for turn in prefix):
            prefix_inputs = dict(
                apply_chat_template(
                    processor,
                    messages=prefix,
                    tools=tools,
                    add_generation_prompt=False,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    **kwargs,
                )
            )
            prefix_length = prefix_inputs["input_ids"].shape[1]
            input_ids = input_ids[prefix_length:]
            attention_mask = attention_mask[prefix_length:]

        loss_mask = torch.ones_like(attention_mask) if message["role"] == "assistant" else torch.zeros_like(attention_mask)
        if message["role"] == "assistant":
            loss_mask[: len(self.generation_prompt)] = 0
        return input_ids, loss_mask, attention_mask, inputs


# Historical imports used this name. Keep it as a compatibility alias while
# the active launcher names the model-neutral class above.
TReXConfigSFTDataset = TReXNativeToolSFTDataset
