import pytest

from core.pvp.olmo_tool_calls import OlmoToolCall
from core.pvp.olmo_tool_calls import parse_olmo_tool_calls


def test_parses_multiple_olmo_tool_calls_and_preserves_normal_text():
    text = """Thinking first.
<function_calls>
play_card(card="ace", metadata={"visible": true, "rank": 1})
remember(values=[1, 2], missing=null)
</function_calls>
Done."""

    normal_text, calls = parse_olmo_tool_calls(text)

    assert normal_text == "Thinking first.\n\nDone."
    assert calls == [
        OlmoToolCall(name="play_card", arguments={"card": "ace", "metadata": {"visible": True, "rank": 1}}),
        OlmoToolCall(name="remember", arguments={"values": [1, 2], "missing": None}),
    ]


def test_text_without_complete_function_call_block_is_unchanged():
    text = "<function_calls>play_card(card='ace')"
    assert parse_olmo_tool_calls(text) == (text, [])


@pytest.mark.parametrize(
    "source",
    [
        "dangerous(__import__('os'))",
        "obj.method(value=1)",
        "tool(**{'value': 1})",
        "value = tool(value=1)",
    ],
)
def test_rejects_non_literal_or_non_call_syntax(source):
    with pytest.raises(ValueError):
        parse_olmo_tool_calls(f"<function_calls>{source}</function_calls>")
