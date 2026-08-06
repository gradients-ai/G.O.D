"""Inference entrypoint with OLMo parsing and an OLMo Hybrid fallback."""

import json
import logging
import os
import runpy
import sys
from collections.abc import Sequence
from contextlib import contextmanager
from typing import Iterator

from core.pvp.olmo_tool_calls import OLMO_TOOL_CALL_END
from core.pvp.olmo_tool_calls import OLMO_TOOL_CALL_START
from core.pvp.olmo_tool_calls import parse_olmo_tool_calls


logger = logging.getLogger(__name__)

_OLMO_HYBRID_MODEL_TYPE = "olmo_hybrid"


def _register_olmo_tool_parser() -> None:
    from sglang.srt.entrypoints.openai.protocol import Tool
    from sglang.srt.environ import envs
    from sglang.srt.function_call.base_format_detector import BaseFormatDetector
    from sglang.srt.function_call.core_types import StreamingParseResult
    from sglang.srt.function_call.core_types import ToolCallItem
    from sglang.srt.function_call.function_call_parser import FunctionCallParser

    class OlmoDetector(BaseFormatDetector):
        def __init__(self):
            super().__init__()
            self.bot_token = OLMO_TOOL_CALL_START
            self.eot_token = OLMO_TOOL_CALL_END

        def has_tool_call(self, text: str) -> bool:
            return OLMO_TOOL_CALL_START in text

        def detect_and_parse(self, text: str, tools: list[Tool]) -> StreamingParseResult:
            try:
                normal_text, parsed_calls = parse_olmo_tool_calls(text)
            except (SyntaxError, ValueError):
                logger.exception("Could not parse OLMo tool call")
                return StreamingParseResult(normal_text=text)

            tool_indices = self._get_tool_indices(tools)
            calls = []
            for parsed in parsed_calls:
                if parsed.name not in tool_indices and not envs.SGLANG_FORWARD_UNKNOWN_TOOLS.get():
                    logger.warning("Model attempted to call undefined function: %s", parsed.name)
                    continue
                calls.append(
                    ToolCallItem(
                        tool_index=tool_indices.get(parsed.name, -1),
                        name=parsed.name,
                        parameters=json.dumps(parsed.arguments, ensure_ascii=False),
                    )
                )
            return StreamingParseResult(normal_text=normal_text, calls=calls)

        def parse_streaming_increment(self, new_text: str, tools: list[Tool]) -> StreamingParseResult:
            self._buffer += new_text
            start = self._buffer.find(OLMO_TOOL_CALL_START)
            if start < 0:
                partial_len = self._ends_with_partial_token(self._buffer, OLMO_TOOL_CALL_START)
                normal_text = self._buffer[:-partial_len] if partial_len else self._buffer
                self._buffer = self._buffer[-partial_len:] if partial_len else ""
                return StreamingParseResult(normal_text=normal_text)

            normal_prefix = self._buffer[:start]
            end = self._buffer.find(OLMO_TOOL_CALL_END, start + len(OLMO_TOOL_CALL_START))
            if end < 0:
                self._buffer = self._buffer[start:]
                return StreamingParseResult(normal_text=normal_prefix)

            block_end = end + len(OLMO_TOOL_CALL_END)
            result = self.detect_and_parse(self._buffer[start:block_end], tools)
            result.normal_text = normal_prefix + result.normal_text
            self._buffer = self._buffer[block_end:]
            return result

        def supports_structural_tag(self) -> bool:
            return False

        def structure_info(self):
            raise NotImplementedError

    FunctionCallParser.ToolCallParserEnum["olmo"] = OlmoDetector


def _option_value(arguments: Sequence[str], option: str) -> str | None:
    for index, argument in enumerate(arguments):
        if argument == option:
            return arguments[index + 1] if index + 1 < len(arguments) else None
        if argument.startswith(f"{option}="):
            return argument.split("=", maxsplit=1)[1]
    return None


def _model_type_for(model_path: str) -> str | None:
    """Read model_type without ever opting into repository-provided code."""
    config_path = os.path.join(model_path, "config.json")
    if os.path.isfile(config_path):
        try:
            with open(config_path) as config_file:
                return str(json.load(config_file).get("model_type", "")).lower() or None
        except (OSError, ValueError, TypeError) as exc:
            logger.warning("Could not inspect local model config %s: %s", config_path, exc)
            return None

    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_path, trust_remote_code=False)
    except Exception as exc:
        logger.warning("Could not inspect model_type for %s; deferring to SGLang: %s", model_path, exc)
        return None
    model_type = getattr(config, "model_type", None)
    return str(model_type).lower() if model_type else None


@contextmanager
def _temporary_argv(arguments: Sequence[str]) -> Iterator[None]:
    original = sys.argv
    sys.argv = [original[0], *arguments]
    try:
        yield
    finally:
        sys.argv = original


def _run_transformers_server(arguments: list[str]) -> None:
    from core.pvp.transformers_server import main as transformers_main

    transformers_main(arguments)


def _run_sglang_server(arguments: list[str]) -> None:
    _register_olmo_tool_parser()
    with _temporary_argv(arguments):
        runpy.run_module("sglang.launch_server", run_name="__main__")


def main(argv: list[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    model_path = _option_value(arguments, "--model-path")
    if model_path and _model_type_for(model_path) == _OLMO_HYBRID_MODEL_TYPE:
        logger.warning("SGLang does not support OLMo Hybrid; using the serialized Transformers fallback")
        _run_transformers_server(arguments)
        return
    _run_sglang_server(arguments)


if __name__ == "__main__":
    main()
