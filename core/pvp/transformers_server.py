"""Small OpenAI-compatible Transformers server for models SGLang cannot run.

This is intentionally a narrow fallback rather than a general inference
server.  It uses the checkpoint tokenizer's native chat template, serializes
generation so per-request seeding is deterministic, and supports only the
non-streaming chat-completions surface used by environment evaluation.
"""

import argparse
import hashlib
import json
import logging
import random
import sys
import threading
import time
from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from fastapi import Body
from fastapi import FastAPI
from fastapi import HTTPException

from core.pvp.olmo_tool_calls import parse_olmo_tool_calls


logger = logging.getLogger(__name__)

_SUPPORTED_ROLES = {"system", "user", "assistant", "tool", "environment"}
_DEFAULT_MAX_TOKENS = 256


class InvalidChatRequest(ValueError):
    """The request uses an unsupported or malformed OpenAI chat option."""


def _option_name(argument: str) -> str:
    return argument.split("=", maxsplit=1)[0].replace("_", "-").lower()


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Transformers OpenAI-compatible fallback server")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--chat-template")
    parser.add_argument("--tool-call-parser")
    parser.add_argument("--model-impl")
    parser.add_argument("--log-level", default="warning")
    parser.add_argument("--decode-log-interval")
    parser.add_argument("--attention-backend")
    parser.add_argument("--prefill-attention-backend")
    parser.add_argument("--decode-attention-backend")
    parser.add_argument("--sampling-backend")
    parser.add_argument("--enable-deterministic-inference", action="store_true")
    return parser


def parse_cli_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the SGLang-shaped command line used by existing launchers.

    Unknown SGLang tuning flags are ignored because they have no equivalent in
    this backend. Native SGLang LoRA flags are rejected explicitly: serving an
    OLMo Hybrid adapter requires materializing merged full weights first.
    """
    parser = _build_cli_parser()
    raw_args = list(argv) if argv is not None else None
    inspected_args = raw_args if raw_args is not None else sys.argv[1:]
    lora_flags = [argument for argument in inspected_args if argument.startswith("--") and "lora" in _option_name(argument)]
    if lora_flags:
        parser.error(
            "native LoRA serving is unsupported by the Transformers fallback; "
            f"merge the adapter first (received {', '.join(lora_flags)})"
        )

    args, unknown = parser.parse_known_args(raw_args)
    if args.tensor_parallel_size != 1:
        parser.error("the Transformers fallback supports only --tensor-parallel-size 1")
    if unknown:
        logger.warning("Ignoring SGLang-only arguments in Transformers fallback: %s", " ".join(unknown))
    return args


def _resolve_torch_dtype(torch_module: Any, dtype_name: str) -> Any:
    normalized = dtype_name.lower().replace("torch.", "")
    aliases = {
        "auto": "auto",
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
        "fp16": "float16",
        "float16": "float16",
        "half": "float16",
        "fp32": "float32",
        "float32": "float32",
        "float": "float32",
    }
    resolved = aliases.get(normalized)
    if resolved is None:
        raise ValueError(f"Unsupported Transformers fallback dtype: {dtype_name}")
    return resolved if resolved == "auto" else getattr(torch_module, resolved)


def _load_chat_template(template_arg: str | None) -> str | None:
    if not template_arg:
        return None
    template_path = Path(template_arg)
    if template_path.is_file():
        return template_path.read_text()
    # Transformers also accepts an inline template. This keeps the CLI
    # compatible with SGLang deployments that provide one directly.
    return template_arg


def load_runtime(args: argparse.Namespace) -> "TransformersRuntime":
    """Load a causal LM and tokenizer without executing repository code."""
    import torch

    from core.model_loading import load_causal_language_model
    from core.model_loading import load_causal_tokenizer

    dtype = _resolve_torch_dtype(torch, args.dtype)
    tokenizer = load_causal_tokenizer(args.model_path, trust_remote_code=False)
    model = load_causal_language_model(
        args.model_path,
        dtype=dtype,
        trust_remote_code=False,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    if args.enable_deterministic_inference:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    return TransformersRuntime(
        model=model,
        tokenizer=tokenizer,
        torch_module=torch,
        model_id=args.model_path,
        device=device,
        default_seed=args.random_seed,
        chat_template=_load_chat_template(args.chat_template),
    )


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise InvalidChatRequest(f"{field} must be a positive integer")
    return value


def _finite_number(value: Any, field: str, *, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidChatRequest(f"{field} must be a number")
    number = float(value)
    if not minimum <= number <= maximum:
        raise InvalidChatRequest(f"{field} must be between {minimum} and {maximum}")
    return number


def _decode_tool_arguments(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, Mapping):
        return dict(arguments)
    if not isinstance(arguments, str):
        return {}
    try:
        decoded = json.loads(arguments)
    except json.JSONDecodeError:
        return {}
    return dict(decoded) if isinstance(decoded, Mapping) else {}


def _messages_for_tokenizer(raw_messages: Any) -> list[dict[str, Any]]:
    """Convert OpenAI wire history to the OLMo tokenizer template contract."""
    if not isinstance(raw_messages, list) or not raw_messages:
        raise InvalidChatRequest("messages must be a non-empty list")

    messages: list[dict[str, Any]] = []
    for raw_message in raw_messages:
        if not isinstance(raw_message, Mapping):
            raise InvalidChatRequest("each message must be an object")
        role = raw_message.get("role")
        if role not in _SUPPORTED_ROLES:
            raise InvalidChatRequest(f"unsupported message role: {role!r}")
        message = dict(raw_message)
        if role == "assistant" and isinstance(message.get("tool_calls"), list):
            normalized_calls = []
            for raw_call in message["tool_calls"]:
                if not isinstance(raw_call, Mapping):
                    continue
                call = dict(raw_call)
                function = call.get("function")
                if isinstance(function, Mapping):
                    normalized_function = dict(function)
                    normalized_function["arguments"] = _decode_tool_arguments(function.get("arguments"))
                    call["function"] = normalized_function
                normalized_calls.append(call)
            message["tool_calls"] = normalized_calls
        messages.append(message)
    return messages


def _tools_for_tokenizer(raw_tools: Any, tool_choice: Any) -> list[dict[str, Any]] | None:
    if tool_choice == "none":
        return None
    if raw_tools is None:
        return None
    if not isinstance(raw_tools, list):
        raise InvalidChatRequest("tools must be a list")
    tools = []
    for tool in raw_tools:
        if not isinstance(tool, Mapping) or tool.get("type") != "function" or not isinstance(tool.get("function"), Mapping):
            raise InvalidChatRequest("only OpenAI function tools are supported")
        tools.append(dict(tool))
    return tools or None


def _move_inputs_to_device(inputs: Any, device: str) -> dict[str, Any]:
    if hasattr(inputs, "to"):
        inputs = inputs.to(device)
    if isinstance(inputs, Mapping):
        return {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
    value = inputs.to(device) if hasattr(inputs, "to") else inputs
    return {"input_ids": value}


def _sequence_length(input_ids: Any) -> int:
    shape = getattr(input_ids, "shape", None)
    if shape is not None:
        return int(shape[-1])
    first = input_ids[0] if input_ids and isinstance(input_ids[0], (list, tuple)) else input_ids
    return len(first)


def _token_count(token_ids: Any) -> int:
    if hasattr(token_ids, "numel"):
        return int(token_ids.numel())
    return len(token_ids)


def _strip_special_tokens(text: str, tokenizer: Any) -> str:
    for token in getattr(tokenizer, "all_special_tokens", ()):
        if token:
            text = text.replace(token, "")
    return text.strip()


def _truncate_at_stop(text: str, stop: Any) -> str:
    if stop is None:
        return text
    stops = [stop] if isinstance(stop, str) else stop
    if not isinstance(stops, list) or not all(isinstance(item, str) for item in stops):
        raise InvalidChatRequest("stop must be a string or list of strings")
    positions = [position for item in stops if item and (position := text.find(item)) >= 0]
    return text[: min(positions)] if positions else text


def _allowed_tool_names(tools: list[dict[str, Any]] | None) -> set[str]:
    if not tools:
        return set()
    return {
        str(tool["function"].get("name"))
        for tool in tools
        if tool["function"].get("name")
    }


class TransformersRuntime:
    """Serialized deterministic generation over a loaded Transformers model."""

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        torch_module: Any,
        model_id: str,
        device: str,
        default_seed: int,
        chat_template: str | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.torch = torch_module
        self.model_id = model_id
        self.device = device
        self.default_seed = default_seed
        self.chat_template = chat_template
        self.created = int(time.time())
        self._generation_lock = threading.Lock()

    def _seed(self, seed: int) -> None:
        random.seed(seed)
        try:
            import numpy as np

            np.random.seed(seed % (2**32))
        except ImportError:
            pass
        self.torch.manual_seed(seed)
        cuda = getattr(self.torch, "cuda", None)
        if cuda is not None and cuda.is_available():
            cuda.manual_seed_all(seed)

    def chat_completion(self, request: Mapping[str, Any]) -> dict[str, Any]:
        if request.get("stream", False):
            raise InvalidChatRequest("streaming chat completions are not supported")
        if request.get("n", 1) != 1:
            raise InvalidChatRequest("only n=1 is supported")

        messages = _messages_for_tokenizer(request.get("messages"))
        tools = _tools_for_tokenizer(request.get("tools"), request.get("tool_choice"))
        max_tokens = _positive_int(
            request.get("max_completion_tokens", request.get("max_tokens", _DEFAULT_MAX_TOKENS)),
            "max_tokens",
        )
        temperature_value = request.get("temperature")
        temperature = (
            0.0
            if temperature_value is None
            else _finite_number(temperature_value, "temperature", minimum=0.0, maximum=2.0)
        )
        top_p = _finite_number(request.get("top_p", 1.0), "top_p", minimum=0.0, maximum=1.0)
        if top_p == 0:
            raise InvalidChatRequest("top_p must be greater than zero")
        raw_seed = request.get("seed", self.default_seed)
        if raw_seed is None:
            raw_seed = self.default_seed
        if isinstance(raw_seed, bool) or not isinstance(raw_seed, int):
            raise InvalidChatRequest("seed must be an integer")
        seed = raw_seed

        with self._generation_lock:
            self._seed(seed)
            encoded = self.tokenizer.apply_chat_template(
                messages,
                tools=tools,
                chat_template=self.chat_template,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
                tokenizer_kwargs={"return_token_type_ids": False},
            )
            model_inputs = _move_inputs_to_device(encoded, self.device)
            prompt_tokens = _sequence_length(model_inputs["input_ids"])
            generation_args: dict[str, Any] = {
                "max_new_tokens": max_tokens,
                "do_sample": temperature > 0,
                "use_cache": True,
            }
            if temperature > 0:
                generation_args.update(temperature=temperature, top_p=top_p)
            if getattr(self.tokenizer, "eos_token_id", None) is not None:
                generation_args["eos_token_id"] = self.tokenizer.eos_token_id
            pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
            if pad_token_id is not None:
                generation_args["pad_token_id"] = pad_token_id

            inference_mode = getattr(self.torch, "inference_mode", nullcontext)
            with inference_mode():
                output = self.model.generate(**model_inputs, **generation_args)

        sequences = getattr(output, "sequences", output)
        generated_ids = sequences[0][prompt_tokens:]
        completion_tokens = _token_count(generated_ids)
        decode_args = {"clean_up_tokenization_spaces": False}
        raw_text = self.tokenizer.decode(generated_ids, skip_special_tokens=False, **decode_args)
        clean_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True, **decode_args)
        raw_text = _truncate_at_stop(raw_text, request.get("stop"))
        clean_text = _truncate_at_stop(clean_text, request.get("stop"))

        try:
            normal_text, parsed_calls = parse_olmo_tool_calls(raw_text)
        except (SyntaxError, ValueError):
            logger.warning("Could not parse OLMo tool call; returning output as content", exc_info=True)
            normal_text, parsed_calls = clean_text, []

        output_had_tool_calls = bool(parsed_calls)
        allowed_names = _allowed_tool_names(tools)
        parsed_calls = [call for call in parsed_calls if call.name in allowed_names]
        if output_had_tool_calls:
            content = _strip_special_tokens(normal_text, self.tokenizer) or None
        else:
            content = clean_text.strip() or None

        tool_calls = []
        for index, call in enumerate(parsed_calls):
            arguments = json.dumps(call.arguments, ensure_ascii=False, separators=(",", ":"))
            call_hash = hashlib.sha256(f"{seed}\0{raw_text}\0{index}".encode()).hexdigest()[:24]
            tool_calls.append(
                {
                    "id": f"call_{call_hash}",
                    "type": "function",
                    "function": {"name": call.name, "arguments": arguments},
                }
            )

        completion_hash = hashlib.sha256(f"{seed}\0{raw_text}".encode()).hexdigest()[:24]
        response_message: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            response_message["tool_calls"] = tool_calls
        return {
            "id": f"chatcmpl-{completion_hash}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": str(request.get("model") or self.model_id),
            "choices": [
                {
                    "index": 0,
                    "message": response_message,
                    "logprobs": None,
                    "finish_reason": "tool_calls" if tool_calls else "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }


def create_app(runtime: TransformersRuntime) -> FastAPI:
    app = FastAPI(title="Transformers OpenAI fallback", docs_url=None, redoc_url=None)

    @app.get("/v1/models")
    def list_models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": runtime.model_id,
                    "object": "model",
                    "created": runtime.created,
                    "owned_by": "transformers",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(request: dict[str, Any] = Body(...)) -> dict[str, Any]:
        try:
            return runtime.chat_completion(request)
        except InvalidChatRequest as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Transformers fallback generation failed")
            raise HTTPException(status_code=500, detail="model generation failed") from exc

    return app


def main(argv: list[str] | None = None) -> None:
    import uvicorn

    args = parse_cli_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.WARNING))
    runtime = load_runtime(args)
    uvicorn.run(create_app(runtime), host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
