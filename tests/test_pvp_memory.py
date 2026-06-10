"""Tests for the fixed-slot memory store and tool dispatch (Phase 1, pure).

These modules have no pyspiel / openai dependency, so they run everywhere.
The store is deliberately total: bad indices and malformed input return an
error string and never raise, so a fumbled memory op can never crash a game.
"""

import pytest

from core.models.pvp_models import MemoryArea
from core.models.pvp_models import MemoryConfig
from core.models.pvp_models import MemoryOp
from core.pvp.memory import SlotMemory
from core.pvp.memory import WhitespaceTokenCounter
from core.pvp import tools


def make_memory(n_slots: int = 4, budget: int = 5) -> SlotMemory:
    return SlotMemory(n_slots=n_slots, slot_token_budget=budget, counter=WhitespaceTokenCounter())


# --- Token counter ---


class TestWhitespaceTokenCounter:
    def test_count_is_word_count(self):
        assert WhitespaceTokenCounter().count("a b c d") == 4
        assert WhitespaceTokenCounter().count("") == 0

    def test_truncate_head_keeps_front(self):
        assert WhitespaceTokenCounter().truncate("a b c d e", 2, "head") == "a b"

    def test_truncate_tail_keeps_end(self):
        assert WhitespaceTokenCounter().truncate("a b c d e", 2, "tail") == "d e"

    def test_truncate_noop_when_under_budget(self):
        assert WhitespaceTokenCounter().truncate("a b", 5, "head") == "a b"


# --- SlotMemory basics ---


class TestSlotMemoryBasics:
    def test_init_creates_empty_numbered_slots(self):
        mem = make_memory(n_slots=3)
        assert set(mem.slots.keys()) == {1, 2, 3}
        assert all(v == "" for v in mem.slots.values())

    def test_rewrite_sets_content(self):
        mem = make_memory()
        mem.rewrite(2, "hello world")
        assert mem.slots[2] == "hello world"
        assert mem.read(2) == "hello world"

    def test_read_empty_slot_shows_placeholder(self):
        assert make_memory().read(1) == "(empty)"

    def test_clear_empties_slot(self):
        mem = make_memory()
        mem.rewrite(1, "stuff")
        mem.clear(1)
        assert mem.slots[1] == ""

    def test_render_lists_all_slots_including_empty(self):
        mem = make_memory(n_slots=2)
        mem.rewrite(1, "plan")
        rendered = mem.render()
        assert "[1]" in rendered and "plan" in rendered
        assert "[2]" in rendered and "(empty)" in rendered

    def test_render_title_is_prepended(self):
        assert make_memory(n_slots=1).render(title="WORKING").startswith("WORKING")


# --- Append vs rewrite semantics ---


class TestAppendAndRewrite:
    def test_append_to_empty_equals_content(self):
        mem = make_memory()
        mem.append(1, "first note")
        assert mem.slots[1] == "first note"

    def test_append_to_nonempty_concatenates(self):
        mem = make_memory(budget=20)
        mem.rewrite(1, "first")
        mem.append(1, "second")
        assert "first" in mem.slots[1] and "second" in mem.slots[1]
        # order preserved: first before second
        assert mem.slots[1].index("first") < mem.slots[1].index("second")

    def test_rewrite_over_budget_truncates_tail_keeps_head(self):
        mem = make_memory(budget=3)
        mem.rewrite(1, "a b c d e")
        assert mem.slots[1] == "a b c"  # kept the front, dropped the tail

    def test_append_over_budget_drops_oldest_keeps_tail(self):
        mem = make_memory(budget=3)
        mem.rewrite(1, "a b c")  # slot full at budget
        mem.append(1, "d e")     # overflow -> drop oldest (front)
        assert mem.slots[1] == "c d e"

    def test_rewrite_returns_truncation_notice(self):
        mem = make_memory(budget=2)
        res = mem.rewrite(1, "a b c d")
        assert "ok" in res and "truncat" in res.lower()

    def test_append_returns_drop_notice(self):
        mem = make_memory(budget=2)
        mem.rewrite(1, "a b")
        res = mem.append(1, "c d")
        assert "ok" in res and ("drop" in res.lower() or "oldest" in res.lower())


# --- Total semantics: bad ops never raise, never mutate ---


class TestBadOpsAreNoOps:
    @pytest.mark.parametrize("bad", [0, -1, 99])
    def test_rewrite_bad_slot_returns_error_no_raise(self, bad):
        mem = make_memory(n_slots=4)
        before = mem.to_dict()
        res = mem.rewrite(bad, "x")
        assert "error" in res.lower()
        assert mem.to_dict() == before

    @pytest.mark.parametrize("bad", [0, -1, 99])
    def test_append_bad_slot_is_noop(self, bad):
        mem = make_memory(n_slots=4)
        before = mem.to_dict()
        assert "error" in mem.append(bad, "x").lower()
        assert mem.to_dict() == before

    def test_read_bad_slot_returns_error(self):
        assert "error" in make_memory().read(99).lower()

    def test_clear_bad_slot_returns_error(self):
        assert "error" in make_memory().clear(99).lower()


# --- Tool schemas (generated from MemoryArea x MemoryOp) ---


def make_configs(n_working: int = 4, n_longterm: int = 8) -> dict[MemoryArea, MemoryConfig]:
    return {
        MemoryArea.WORKING: MemoryConfig(n_slots=n_working, slot_token_budget=128),
        MemoryArea.LONG_TERM: MemoryConfig(n_slots=n_longterm, slot_token_budget=128),
    }


class TestToolSchemas:
    def test_one_tool_per_area_op_product(self):
        configs = make_configs()
        schemas = tools.build_memory_tools(configs)
        # generative: count is areas x ops, names are derived, not hardcoded
        assert len(schemas) == len(configs) * len(MemoryOp)
        names = {s.function.name for s in schemas}
        expected = {tools.memory_tool_name(a, o) for a in configs for o in MemoryOp}
        assert names == expected

    def test_memory_tools_are_openai_function_shape(self):
        for s in tools.build_memory_tools(make_configs(2, 2)):
            assert s.type == "function"
            params = s.function.parameters
            assert params["type"] == "object"
            assert set(params["required"]) == {"slot", "content"}

    def test_slot_field_is_range_constrained_for_grammar(self):
        schemas = tools.build_memory_tools(make_configs(n_working=4, n_longterm=8))
        working = next(s for s in schemas if s.function.name == "working_memory_rewrite")
        slot = working.function.parameters["properties"]["slot"]
        assert slot["minimum"] == 1 and slot["maximum"] == 4
        longterm = next(s for s in schemas if s.function.name == "long_term_memory_append")
        assert longterm.function.parameters["properties"]["slot"]["maximum"] == 8

    def test_pydantic_titles_and_docstring_are_stripped(self):
        params = tools.build_memory_tools(make_configs())[0].function.parameters
        assert "title" not in params
        assert "description" not in params  # model docstring not leaked onto the wire
        assert all("title" not in p for p in params["properties"].values())
        # per-field descriptions are kept (they're useful to the model)
        assert params["properties"]["slot"]["description"]

    def test_to_openai_round_trips_to_wire_shape(self):
        wire = tools.build_game_action_tool("Legal: 1.").to_openai()
        assert wire["type"] == "function"
        assert wire["function"]["name"] == tools.GAME_ACTION_TOOL_NAME

    def test_game_action_tool_shape(self):
        t = tools.build_game_action_tool("Legal: 1, 2, 3.")
        assert t.function.name == tools.GAME_ACTION_TOOL_NAME
        assert t.function.parameters["required"] == ["action_id"]
        assert "Legal: 1, 2, 3." in t.function.description


# --- Tool dispatch (keyed by MemoryArea) ---


class TestExecuteMemoryTool:
    def _mems(self) -> dict[MemoryArea, SlotMemory]:
        return {
            MemoryArea.WORKING: make_memory(n_slots=4, budget=20),
            MemoryArea.LONG_TERM: make_memory(n_slots=8, budget=20),
        }

    def test_routes_working_rewrite(self):
        mems = self._mems()
        res = tools.execute_memory_tool(mems, "working_memory_rewrite", {"slot": 2, "content": "plan"})
        assert "ok" in res
        assert mems[MemoryArea.WORKING].slots[2] == "plan"
        assert mems[MemoryArea.LONG_TERM].slots[2] == ""

    def test_routes_long_term_append(self):
        mems = self._mems()
        tools.execute_memory_tool(mems, "long_term_memory_append", {"slot": 1, "content": "opp bluffs"})
        assert "opp bluffs" in mems[MemoryArea.LONG_TERM].slots[1]
        assert mems[MemoryArea.WORKING].slots[1] == ""

    def test_unknown_tool_returns_error(self):
        assert "error" in tools.execute_memory_tool(self._mems(), "frobnicate", {}).lower()

    def test_unconfigured_area_returns_error_no_raise(self):
        only_working = {MemoryArea.WORKING: make_memory()}
        res = tools.execute_memory_tool(only_working, "long_term_memory_rewrite", {"slot": 1, "content": "x"})
        assert "error" in res.lower()

    def test_missing_slot_arg_returns_error_no_raise(self):
        assert "error" in tools.execute_memory_tool(self._mems(), "working_memory_rewrite", {"content": "x"}).lower()

    def test_non_integer_slot_is_coerced_or_errors(self):
        mems = self._mems()
        tools.execute_memory_tool(mems, "working_memory_rewrite", {"slot": "2", "content": "ok"})
        assert mems[MemoryArea.WORKING].slots[2] == "ok"
        assert "error" in tools.execute_memory_tool(
            mems, "working_memory_rewrite", {"slot": "two", "content": "x"}
        ).lower()

    def test_non_string_content_is_coerced(self):
        mems = self._mems()
        tools.execute_memory_tool(mems, "working_memory_rewrite", {"slot": 1, "content": 42})
        assert mems[MemoryArea.WORKING].slots[1] == "42"

    def test_bool_slot_is_rejected_not_coerced_to_one(self):
        mems = self._mems()
        res = tools.execute_memory_tool(mems, "working_memory_rewrite", {"slot": True, "content": "x"})
        assert "error" in res.lower()
        assert mems[MemoryArea.WORKING].slots[1] == ""
