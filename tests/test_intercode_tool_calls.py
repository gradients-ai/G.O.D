from core.models.pvp_models import ChatResult
from core.models.pvp_models import ToolCall
from validator.evaluation import eval_intercode as intercode


class FakeEnv:
    def __init__(self):
        self.actions: list[str] = []

    def step(self, action: str):
        self.actions.append(action)
        if action == "submit":
            return "submitted", 0.9, True, {}
        return f"ran {action}", 0.0, False, {"action_executed": True}


class ScriptedChat:
    def __init__(self, *responses: ChatResult):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def __call__(self, client, config, messages, tools=None):
        self.calls.append({"config": config, "messages": messages, "tools": tools})
        assert self.responses, "unexpected extra chat call"
        return self.responses.pop(0)


def _call(name: str, **arguments) -> ToolCall:
    return ToolCall(id="c1", name=name, arguments=arguments)


def test_intercode_tools_are_openai_function_shape():
    tools = intercode.build_intercode_action_tools()
    by_name = {tool.function.name: tool for tool in tools}

    assert set(by_name) == {intercode.INTERCODE_EXECUTE_TOOL_NAME, intercode.INTERCODE_SUBMIT_TOOL_NAME}
    execute_params = by_name[intercode.INTERCODE_EXECUTE_TOOL_NAME].function.parameters
    assert execute_params["required"] == ["command"]
    assert execute_params["properties"]["command"]["type"] == "string"
    assert by_name[intercode.INTERCODE_SUBMIT_TOOL_NAME].function.parameters["properties"] == {}


def test_tool_episode_executes_bash_then_submit(monkeypatch):
    chat = ScriptedChat(
        ChatResult(
            content="I will inspect the target.",
            tool_calls=[_call(intercode.INTERCODE_EXECUTE_TOOL_NAME, command="printf hi")],
        ),
        ChatResult(tool_calls=[_call(intercode.INTERCODE_SUBMIT_TOOL_NAME)]),
    )
    monkeypatch.setattr(intercode, "chat_completion", chat)
    env = FakeEnv()

    reward = intercode._run_tool_episode(
        env,
        query="print hi",
        client=object(),
        model_name="model",
        temperature=0.0,
        max_turns=3,
        max_tokens_per_call=128,
    )

    assert reward == 0.9
    assert env.actions == ["printf hi", "submit"]
    tool_names = {tool.function.name for tool in chat.calls[0]["tools"]}
    assert {intercode.INTERCODE_EXECUTE_TOOL_NAME, intercode.INTERCODE_SUBMIT_TOOL_NAME} <= tool_names
    second_user = chat.calls[1]["messages"][1].content
    assert "Thought 1: I will inspect the target." in second_user
    assert "Observation 1: ran printf hi" in second_user


def test_text_action_is_not_parsed_as_bash(monkeypatch):
    chat = ScriptedChat(ChatResult(content="Action 1: execute[echo should_not_run]"))
    monkeypatch.setattr(intercode, "chat_completion", chat)
    env = FakeEnv()

    reward = intercode._run_tool_episode(
        env,
        query="try old syntax",
        client=object(),
        model_name="model",
        temperature=0.0,
        max_turns=1,
        max_tokens_per_call=128,
    )

    assert reward == 0.9
    assert env.actions == ["submit"]


def test_malformed_execute_tool_does_not_run_command(monkeypatch):
    chat = ScriptedChat(ChatResult(tool_calls=[_call(intercode.INTERCODE_EXECUTE_TOOL_NAME)]))
    monkeypatch.setattr(intercode, "chat_completion", chat)
    env = FakeEnv()

    reward = intercode._run_tool_episode(
        env,
        query="bad tool args",
        client=object(),
        model_name="model",
        temperature=0.0,
        max_turns=1,
        max_tokens_per_call=128,
    )

    assert reward == 0.9
    assert env.actions == ["submit"]


def test_build_sglang_command_adds_tool_parser_for_parser_model_id(monkeypatch):
    monkeypatch.delenv("SGLANG_TOOL_CALL_PARSER", raising=False)

    cmd = intercode._build_sglang_command("/tmp/merged_model", 123, parser_model_id="Qwen/Qwen2.5-Coder-7B")

    assert "--tool-call-parser qwen25" in cmd
