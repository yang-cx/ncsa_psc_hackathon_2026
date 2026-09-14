import json
from types import SimpleNamespace

from inference.model_runtime import _adapter_base_model, generation_kwargs, resolve_model


def test_resolve_verl_adapter_only_export_and_base_model(tmp_path):
    export = tmp_path / "global_step_10" / "huggingface"
    adapter = export / "lora_adapter"
    adapter.mkdir(parents=True)
    (export / "config.json").write_text(json.dumps({"_name_or_path": "Qwen/Qwen3.5-0.8B"}))
    (adapter / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": None}))

    assert resolve_model(export.parent) == str(export.resolve())
    assert _adapter_base_model(adapter, export) == "Qwen/Qwen3.5-0.8B"


def test_resolve_direct_adapter(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "Qwen/Qwen3.5-0.8B"}))

    assert resolve_model(adapter) == str(adapter.resolve())
    assert _adapter_base_model(adapter, adapter) == "Qwen/Qwen3.5-0.8B"


def test_generation_preserves_model_chat_stop_tokens():
    tokenizer = SimpleNamespace(pad_token_id=1, eos_token_id=1)
    model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=[1, 130073]))

    kwargs = generation_kwargs(tokenizer, 256, 0.0, None, model)

    assert kwargs["eos_token_id"] == [1, 130073]


def test_generation_sampling_parameters_are_explicit():
    tokenizer = SimpleNamespace(pad_token_id=1, eos_token_id=1)
    model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=1))

    kwargs = generation_kwargs(
        tokenizer, 512, 0.7, 0.8, model,
        top_k=20, min_p=0.0, repetition_penalty=1.1,
    )

    assert kwargs == {
        "max_new_tokens": 512,
        "pad_token_id": 1,
        "eos_token_id": [1],
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "repetition_penalty": 1.1,
    }


def test_generation_greedy_ignores_sampling_only_parameters():
    tokenizer = SimpleNamespace(pad_token_id=1, eos_token_id=1)
    model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=1))

    kwargs = generation_kwargs(
        tokenizer, 512, 0.0, 0.8, model, top_k=20, min_p=0.1,
    )

    assert kwargs["do_sample"] is False
    assert "temperature" not in kwargs
    assert "top_p" not in kwargs
    assert "top_k" not in kwargs
    assert "min_p" not in kwargs
