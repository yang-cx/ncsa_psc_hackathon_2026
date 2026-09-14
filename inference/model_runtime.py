"""Small, local Hugging Face runtime used by the hackathon inference scripts."""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any


def resolve_model(model: str | Path) -> str:
    """Resolve a local verl/HF export or LoRA adapter, or retain a Hub ID."""
    source = str(model)
    candidate = Path(source).expanduser()
    if not candidate.is_dir():
        if source.startswith((".", "/", "~")):
            raise ValueError(f"Local model directory does not exist: {candidate}")
        return source

    candidate = candidate.resolve()
    export = candidate / "huggingface"
    if export.is_dir():
        candidate = export

    is_adapter = (candidate / "adapter_config.json").is_file()
    has_nested_adapter = (candidate / "lora_adapter" / "adapter_config.json").is_file()
    if not (candidate / "config.json").is_file() and not is_adapter and not has_nested_adapter:
        raise ValueError(
            f"{candidate} is not a Transformers model or LoRA adapter directory. Pass a directory "
            "containing config.json, adapter_config.json, or a verl checkpoint containing huggingface/."
        )
    return str(candidate)


def _adapter_directory(model_dir: Path) -> Path | None:
    """Return a direct or nested PEFT adapter directory, if present."""
    if (model_dir / "adapter_config.json").is_file():
        return model_dir
    nested = model_dir / "lora_adapter"
    if (nested / "adapter_config.json").is_file():
        return nested
    return None


def _has_model_weights(model_dir: Path) -> bool:
    return any((model_dir / name).is_file() for name in (
        "model.safetensors",
        "pytorch_model.bin",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    ))


def _read_model_id(path: Path, keys: tuple[str, ...]) -> str | None:
    try:
        contents = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(contents, dict):
        return None
    for key in keys:
        value = contents.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _adapter_base_model(adapter_dir: Path, export_dir: Path) -> str | None:
    """Find the base model recorded by PEFT or the adjacent VERL export."""
    base = _read_model_id(adapter_dir / "adapter_config.json", ("base_model_name_or_path",))
    if base:
        return base
    return _read_model_id(export_dir / "config.json", ("_name_or_path", "base_model_name_or_path"))


def select_device(requested: str) -> str:
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("--device cuda was requested, but CUDA is not available.")
    return requested


def load_model(
    model: str | Path,
    device: str = "auto",
    trust_remote_code: bool = False,
    base_model: str | Path | None = None,
) -> tuple[Any, Any, str, str]:
    """Load a full HF model or apply a local LoRA adapter to its base model."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForMultimodalLM, AutoProcessor, AutoTokenizer

    resolved = resolve_model(model)
    selected_device = select_device(device)
    model_dir = Path(resolved)
    adapter_path = _adapter_directory(model_dir) if model_dir.is_dir() else None
    adapter_only = adapter_path is not None and not _has_model_weights(model_dir)
    load_source = resolved
    if adapter_only:
        base = str(base_model) if base_model is not None else _adapter_base_model(adapter_path, model_dir)
        if not base:
            raise ValueError(
                f"{adapter_path} is a LoRA adapter without an identifiable base model. "
                "Pass --base-model, for example --base-model Qwen/Qwen3.5-0.8B."
            )
        load_source = resolve_model(base)

    config = AutoConfig.from_pretrained(load_source, trust_remote_code=trust_remote_code)
    is_qwen35 = config.model_type in {"qwen3_5", "qwen3_5_moe"}
    if is_qwen35:
        tokenizer = AutoProcessor.from_pretrained(load_source, trust_remote_code=trust_remote_code)
    else:
        tokenizer = AutoTokenizer.from_pretrained(load_source, trust_remote_code=trust_remote_code)

    underlying_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    if underlying_tokenizer.pad_token_id is None:
        underlying_tokenizer.pad_token = underlying_tokenizer.eos_token

    dtype = torch.bfloat16 if selected_device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    if selected_device == "cpu":
        dtype = torch.float32
    model_class = AutoModelForMultimodalLM if is_qwen35 else AutoModelForCausalLM
    load_kwargs: dict[str, Any] = {"dtype": dtype, "trust_remote_code": trust_remote_code}
    if selected_device == "cuda":
        # Stream checkpoint shards directly to the available accelerator rather
        # than materializing a full CPU copy before model.to(cuda). This avoids
        # an otherwise large host-memory spike for Qwen3.5-9B.
        load_kwargs.update(low_cpu_mem_usage=True, device_map="auto")
        if is_qwen35:
            # Qwen3.5-9B fits comfortably in BF16 on Perlmutter's 40 GB A100.
            # Keep roughly 10 GB for generation and allocator headroom while
            # streaming the checkpoint directly to the accelerator.
            gpu_limit = os.environ.get("QWEN35_GPU_MEMORY_GIB", "30")
            load_kwargs["max_memory"] = {
                **{index: f"{gpu_limit}GiB" for index in range(torch.cuda.device_count())},
                "cpu": "160GiB",
            }
    model = model_class.from_pretrained(load_source, **load_kwargs)
    # VERL stores the frozen base-layer weights in the HF export and writes the
    # learned low-rank tensors beside them in ``lora_adapter``.  The nested
    # adapter therefore remains necessary even though full base weights exist.
    if adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
    if selected_device != "cuda":
        model.to(selected_device)
    model.eval()
    description = f"{load_source} + {adapter_path}" if adapter_only else resolved
    return model, tokenizer, description, selected_device


def _uses_multimodal_processor(runtime: Any) -> bool:
    """Return whether the runtime is a Qwen3.5-style AutoProcessor."""
    return hasattr(runtime, "image_processor") and hasattr(runtime, "tokenizer")


def _move_inputs_to_model(inputs: Any, device: str) -> dict[str, Any]:
    """Move a tokenizer/processor batch to the generation device."""
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in dict(inputs).items()}


def _decode(runtime: Any, token_ids: Any, *, skip_special_tokens: bool = True) -> str:
    """Decode generated IDs using either an AutoTokenizer or AutoProcessor."""
    return runtime.decode(token_ids, skip_special_tokens=skip_special_tokens)


def generation_kwargs(
    tokenizer: Any,
    max_new_tokens: int,
    temperature: float,
    top_p: float | None,
    model: Any | None = None,
    *,
    top_k: int | None = None,
    min_p: float | None = None,
    repetition_penalty: float = 1.0,
) -> dict[str, Any]:
    underlying_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    eos_token_ids = []
    configured_eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    for token_id in (
        configured_eos if isinstance(configured_eos, (list, tuple)) else [configured_eos]
    ):
        if token_id is not None and token_id not in eos_token_ids:
            eos_token_ids.append(token_id)
    if underlying_tokenizer.eos_token_id not in eos_token_ids:
        eos_token_ids.append(underlying_tokenizer.eos_token_id)
    kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": underlying_tokenizer.pad_token_id,
        "eos_token_id": eos_token_ids,
        "do_sample": temperature > 0,
    }
    if temperature > 0:
        kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p
        if top_k is not None:
            kwargs["top_k"] = top_k
        if min_p is not None:
            kwargs["min_p"] = min_p
    if repetition_penalty != 1.0:
        kwargs["repetition_penalty"] = repetition_penalty
    return kwargs


def generate_text(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float | None = None,
    top_k: int | None = None,
    min_p: float | None = None,
    repetition_penalty: float = 1.0,
) -> str:
    """Generate a completion for a literal text prompt."""
    import torch

    if _uses_multimodal_processor(tokenizer):
        inputs = _move_inputs_to_model(tokenizer(text=prompt, return_tensors="pt"), model.device)
    else:
        inputs = _move_inputs_to_model(tokenizer(prompt, return_tensors="pt"), model.device)
    input_ids = inputs["input_ids"]
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            **generation_kwargs(
                tokenizer, max_new_tokens, temperature, top_p, model,
                top_k=top_k, min_p=min_p, repetition_penalty=repetition_penalty,
            ),
        )
    completion_ids = generated[0, input_ids.shape[-1] :]
    return _decode(tokenizer, completion_ids)


def generate_chat(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float | None = None,
    top_k: int | None = None,
    min_p: float | None = None,
    repetition_penalty: float = 1.0,
    enable_thinking: bool = False,
    tools: list[dict[str, Any]] | None = None,
) -> str:
    """Generate one assistant turn with the checkpoint's native chat template."""
    import torch

    if _uses_multimodal_processor(tokenizer):
        inputs = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            # Qwen3.5 otherwise spends completion tokens on a visible
            # reasoning trace. ROOT benchmark answers are concise finals.
            enable_thinking=enable_thinking,
        )
        inputs = _move_inputs_to_model(inputs, model.device)
    else:
        inputs = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=enable_thinking,
        )
        inputs = _move_inputs_to_model(inputs, model.device)
    input_ids = inputs["input_ids"]
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            **generation_kwargs(
                tokenizer, max_new_tokens, temperature, top_p, model,
                top_k=top_k, min_p=min_p, repetition_penalty=repetition_penalty,
            ),
        )
    completion_ids = generated[0, input_ids.shape[-1] :]
    # Tool-capable templates may encode call delimiters as special tokens.
    # Preserve them whenever a schema is present; the agent parser removes any
    # terminal chat-control tokens after extracting the call.
    return _decode(tokenizer, completion_ids, skip_special_tokens=tools is None)
