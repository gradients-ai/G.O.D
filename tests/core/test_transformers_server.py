from contextlib import nullcontext

import pytest
from fastapi.testclient import TestClient

from core.pvp.transformers_server import TransformersRuntime
from core.pvp.transformers_server import create_app
from core.pvp.transformers_server import parse_cli_args


class _FakeCuda:
    def __init__(self):
        self.seeds = []

    def is_available(self):
        return True

    def manual_seed_all(self, seed):
        self.seeds.append(seed)


class _FakeTorch:
    def __init__(self):
        self.cuda = _FakeCuda()
        self.seeds = []

    def manual_seed(self, seed):
        self.seeds.append(seed)

    @staticmethod
    def inference_mode():
        return nullcontext()


class _FakeTokenizer:
    eos_token_id = 7
    pad_token_id = 0
    all_special_tokens = ["<|endoftext|>"]

    def __init__(self):
        self.template_messages = None
        self.template_kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.template_messages = messages
        self.template_kwargs = kwargs
        return {"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]}

    @staticmethod
    def decode(token_ids, *, skip_special_tokens, **kwargs):
        assert token_ids == [10, 11]
        assert kwargs == {"clean_up_tokenization_spaces": False}
        if skip_special_tokens:
            return ""
        return '<function_calls>play(card="ace")</function_calls><|endoftext|>'


class _FakeModel:
    def __init__(self):
        self.generate_kwargs = None

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return [[1, 2, 3, 10, 11]]


def _runtime():
    model = _FakeModel()
    tokenizer = _FakeTokenizer()
    torch_module = _FakeTorch()
    runtime = TransformersRuntime(
        model=model,
        tokenizer=tokenizer,
        torch_module=torch_module,
        model_id="allenai/Olmo-Hybrid-Instruct-SFT-7B",
        device="cuda",
        default_seed=42,
    )
    return runtime, model, tokenizer, torch_module


def test_chat_completion_uses_native_template_and_returns_openai_tool_calls():
    runtime, model, tokenizer, torch_module = _runtime()
    request = {
        "model": "submitted-model",
        "messages": [
            {"role": "user", "content": "play"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_old",
                        "type": "function",
                        "function": {"name": "remember", "arguments": '{"slot": 1}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_old", "content": "done"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "play",
                    "description": "Play a card",
                    "parameters": {"type": "object"},
                },
            }
        ],
        "tool_choice": "auto",
        "temperature": 0.7,
        "top_p": 0.8,
        "max_tokens": 17,
        "seed": 99,
    }

    response = runtime.chat_completion(request)

    assert tokenizer.template_messages[1]["tool_calls"][0]["function"]["arguments"] == {"slot": 1}
    assert tokenizer.template_kwargs == {
        "tools": request["tools"],
        "chat_template": None,
        "add_generation_prompt": True,
        "tokenize": True,
        "return_tensors": "pt",
        "return_dict": True,
        "tokenizer_kwargs": {"return_token_type_ids": False},
    }
    assert model.generate_kwargs == {
        "input_ids": [[1, 2, 3]],
        "attention_mask": [[1, 1, 1]],
        "max_new_tokens": 17,
        "do_sample": True,
        "use_cache": True,
        "temperature": 0.7,
        "top_p": 0.8,
        "eos_token_id": 7,
        "pad_token_id": 0,
    }
    assert torch_module.seeds == [99]
    assert torch_module.cuda.seeds == [99]
    assert response["model"] == "submitted-model"
    assert response["choices"][0]["finish_reason"] == "tool_calls"
    assert response["choices"][0]["message"]["content"] is None
    assert response["choices"][0]["message"]["tool_calls"][0]["function"] == {
        "name": "play",
        "arguments": '{"card":"ace"}',
    }
    assert response["usage"] == {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}


def test_openai_endpoints_report_model_and_reject_streaming():
    runtime, _, _, _ = _runtime()
    client = TestClient(create_app(runtime))

    models_response = client.get("/v1/models")
    assert models_response.status_code == 200
    assert models_response.json()["data"][0]["id"] == runtime.model_id

    stream_response = client.post(
        "/v1/chat/completions",
        json={"model": runtime.model_id, "messages": [{"role": "user", "content": "hello"}], "stream": True},
    )
    assert stream_response.status_code == 400
    assert stream_response.json()["detail"] == "streaming chat completions are not supported"


def test_default_seed_and_greedy_generation_are_reproducible():
    runtime, model, _, torch_module = _runtime()
    request = {
        "messages": [{"role": "user", "content": "play"}],
        "tools": [{"type": "function", "function": {"name": "play", "parameters": {}}}],
        "temperature": 0,
        "max_tokens": 4,
    }

    first = runtime.chat_completion(request)
    second = runtime.chat_completion(request)

    assert first["id"] == second["id"]
    assert first["choices"] == second["choices"]
    assert torch_module.seeds == [42, 42]
    assert model.generate_kwargs["do_sample"] is False
    assert "temperature" not in model.generate_kwargs
    assert "top_p" not in model.generate_kwargs


def test_cli_accepts_existing_sglang_flags_and_rejects_native_lora():
    args = parse_cli_args(
        [
            "--model-path",
            "/models/olmo",
            "--port",
            "31000",
            "--tensor-parallel-size",
            "1",
            "--dtype",
            "bfloat16",
            "--enable-deterministic-inference",
            "--tool-call-parser",
            "olmo",
            "--attention-backend",
            "triton",
            "--prefill-attention-backend",
            "triton",
            "--decode-attention-backend",
            "triton",
            "--sampling-backend",
            "pytorch",
        ]
    )
    assert args.model_path == "/models/olmo"
    assert args.port == 31000
    assert args.enable_deterministic_inference is True

    with pytest.raises(SystemExit):
        parse_cli_args(["--model-path", "/models/olmo", "--enable-lora", "--lora-paths", "trained=/lora"])
