"""Safe parser for the function-call syntax emitted by OLMo 3 tokenizers."""

import ast
from dataclasses import dataclass
from typing import Any


OLMO_TOOL_CALL_START = "<function_calls>"
OLMO_TOOL_CALL_END = "</function_calls>"


@dataclass(frozen=True)
class OlmoToolCall:
    name: str
    arguments: dict[str, Any]


def _literal(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        json_names = {"true": True, "false": False, "null": None}
        if node.id in json_names:
            return json_names[node.id]
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_literal(item) for item in node.elts]
    if isinstance(node, ast.Dict):
        return {_literal(key): _literal(value) for key, value in zip(node.keys, node.values)}
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _literal(node.operand)
        if not isinstance(value, (int, float)):
            raise ValueError("Unary operators are only valid for numeric tool arguments")
        return value if isinstance(node.op, ast.UAdd) else -value
    raise ValueError("Tool call arguments must be literals")


def _parse_call(node: ast.AST) -> OlmoToolCall:
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        raise ValueError("Expected a simple function call")
    if node.args:
        raise ValueError("OLMo tool calls must use named arguments")

    arguments: dict[str, Any] = {}
    for keyword in node.keywords:
        if keyword.arg is None:
            raise ValueError("Expanded keyword arguments are not supported")
        arguments[keyword.arg] = _literal(keyword.value)
    return OlmoToolCall(name=node.func.id, arguments=arguments)


def parse_olmo_tool_calls(text: str) -> tuple[str, list[OlmoToolCall]]:
    """Extract one OLMo ``<function_calls>`` block without executing code."""
    start = text.find(OLMO_TOOL_CALL_START)
    if start < 0:
        return text, []
    content_start = start + len(OLMO_TOOL_CALL_START)
    end = text.find(OLMO_TOOL_CALL_END, content_start)
    if end < 0:
        return text, []

    call_source = text[content_start:end].strip()
    normal_text = (text[:start] + text[end + len(OLMO_TOOL_CALL_END) :]).strip()
    if not call_source:
        return normal_text, []

    module = ast.parse(call_source, mode="exec")
    calls: list[OlmoToolCall] = []
    for statement in module.body:
        if not isinstance(statement, ast.Expr):
            raise ValueError("Only function calls are allowed inside <function_calls>")
        calls.append(_parse_call(statement.value))
    return normal_text, calls
