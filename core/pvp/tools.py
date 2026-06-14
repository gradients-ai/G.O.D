"""OpenAI-compatible tool schemas and dispatch for the PvP memory harness.

Memory tools are generated from the MemoryArea x MemoryOp product, and the
argument schemas come from the Pydantic models in core.models.pvp_models. To
add a memory area or operation, extend those enums (and provide a SlotMemory
instance / method) — the toolset and dispatch expand with no edits here.

game_action terminates the turn and is handled by the turn loop, not by
execute_memory_tool. All dispatch is total: an unknown tool or malformed args
returns an error string, never an exception.
"""

from __future__ import annotations

from copy import deepcopy

from pydantic import BaseModel

from core.models.pvp_models import FunctionSchema
from core.models.pvp_models import GameActionArgs
from core.models.pvp_models import JsonScalar
from core.models.pvp_models import MemoryArea
from core.models.pvp_models import MemoryConfig
from core.models.pvp_models import MemoryOp
from core.models.pvp_models import MemorySlotEdit
from core.models.pvp_models import ToolSchema
from core.pvp.memory import SlotMemory


GAME_ACTION_TOOL_NAME = "game_action"

# Presentation metadata — exactly one entry per enum member (asserted exhaustive).
_AREA_PURPOSE: dict[MemoryArea, str] = {
    MemoryArea.WORKING: "notes for THIS game, reset each game",
    MemoryArea.LONG_TERM: "notes on THIS opponent, persist across games",
}
_OP_PHRASING: dict[MemoryOp, tuple[str, str]] = {
    MemoryOp.REWRITE: ("Overwrite", "replaces the slot's previous content"),
    MemoryOp.APPEND: ("Append to", "oldest text drops if the slot is full"),
}
assert set(_AREA_PURPOSE) == set(MemoryArea), "every MemoryArea needs a purpose"
assert set(_OP_PHRASING) == set(MemoryOp), "every MemoryOp needs phrasing"


def memory_tool_name(area: MemoryArea, op: MemoryOp) -> str:
    return f"{area.value}_{op.value}"


# Reverse map for dispatch, generated from the same product the tools come from.
_TOOL_TO_AREA_OP: dict[str, tuple[MemoryArea, MemoryOp]] = {
    memory_tool_name(area, op): (area, op) for area in MemoryArea for op in MemoryOp
}


def _params_schema(model: type[BaseModel], *, slot_bounds: tuple[int, int] | None = None) -> dict:
    """JSON Schema for a tool's arguments, stripped of Pydantic titles.

    When slot_bounds is given, the integer 'slot' field carries minimum/maximum
    for the valid range. Advisory only (servers don't grammar-enforce tool args
    under tool_choice="auto"); SlotMemory rejects out-of-range slots regardless.
    """
    schema = deepcopy(model.model_json_schema())
    schema.pop("title", None)
    schema.pop("description", None)  # drop the model docstring; the function description covers it
    for prop in schema.get("properties", {}).values():
        prop.pop("title", None)
    if slot_bounds is not None and "slot" in schema.get("properties", {}):
        lo, hi = slot_bounds
        schema["properties"]["slot"]["minimum"] = lo
        schema["properties"]["slot"]["maximum"] = hi
    return schema


def _function_tool(name: str, description: str, parameters: dict) -> ToolSchema:
    return ToolSchema(function=FunctionSchema(name=name, description=description, parameters=parameters))


def build_memory_tools(configs: dict[MemoryArea, MemoryConfig]) -> list[ToolSchema]:
    """Generate memory tool schemas for the configured areas (one per area x op)."""
    out: list[ToolSchema] = []
    for area, cfg in configs.items():
        for op in MemoryOp:
            verb, effect = _OP_PHRASING[op]
            description = f"{verb} a {area.value} slot (slots 1-{cfg.n_slots}; {_AREA_PURPOSE[area]}); {effect}."
            parameters = _params_schema(MemorySlotEdit, slot_bounds=(1, cfg.n_slots))
            out.append(_function_tool(memory_tool_name(area, op), description, parameters))
    return out


def build_game_action_tool(legal_hint: str, legal_actions: list[int] | None = None) -> ToolSchema:
    """Schema for the turn-terminating move tool. legal_hint surfaces current legal ids.

    When legal_actions is given, action_id carries a JSON-Schema enum of the legal
    set. This is advisory: SGLang with tool_choice="auto" does not grammar-enforce
    argument values (verified empirically — malformed/illegal output reaches the
    wire), so the binding guard is the bot's validate-then-forfeit on the result.
    """
    parameters = _params_schema(GameActionArgs)
    if legal_actions is not None and "action_id" in parameters.get("properties", {}):
        parameters["properties"]["action_id"]["enum"] = list(legal_actions)
    return _function_tool(
        GAME_ACTION_TOOL_NAME,
        f"Commit your move and end your turn. {legal_hint}",
        parameters,
    )


def execute_memory_tool(memories: dict[MemoryArea, SlotMemory], name: str, args: dict[str, JsonScalar]) -> str:
    """Apply a memory tool call against its area's SlotMemory and return the result string."""
    routed = _TOOL_TO_AREA_OP.get(name)
    if routed is None:
        return f"error: unknown tool {name!r}"

    area, op = routed
    target = memories.get(area)
    if target is None:
        return f"error: memory area {area.value} not configured"

    raw_slot = args.get("slot")
    if isinstance(raw_slot, bool) or not isinstance(raw_slot, (int, str)):
        return f"error: tool {name} requires an integer 'slot'"
    try:
        slot = int(raw_slot)
    except ValueError:
        return f"error: tool {name} requires an integer 'slot'"

    content = args.get("content", "")
    if not isinstance(content, str):
        content = str(content)

    return getattr(target, op.value)(slot, content)
