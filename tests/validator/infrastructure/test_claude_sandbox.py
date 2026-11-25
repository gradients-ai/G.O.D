import asyncio
from pathlib import Path

import pytest

from validator.infrastructure.claude_sandbox import agent_env
from validator.infrastructure.claude_sandbox import confine_to_cwd_hooks
from validator.infrastructure.claude_sandbox import tool_call_denial


@pytest.fixture
def root(tmp_path: Path) -> Path:
    repo = tmp_path / "review"
    (repo / "challenger").mkdir(parents=True)
    (repo / "challenger" / "train.py").write_text("print(1)\n")
    (tmp_path / "secret.env").write_text("KEY=1\n")
    (repo / "challenger" / "link.env").symlink_to(tmp_path / "secret.env")
    (repo / "challenger" / "linkdir").symlink_to(tmp_path, target_is_directory=True)
    return repo.resolve()


@pytest.mark.parametrize(
    "tool,tool_input",
    [
        ("Read", {"file_path": "challenger/train.py"}),
        ("Read", {"file_path": "{root}/challenger/train.py"}),
        ("Glob", {"pattern": "**/*.py"}),
        ("Glob", {"pattern": "*.py", "path": "challenger"}),
        ("Grep", {"pattern": "import os", "path": "{root}", "glob": "*.py"}),
        ("Grep", {"pattern": "../../etc/passwd"}),
    ],
)
def test_calls_inside_the_repo_are_allowed(root, tool, tool_input):
    tool_input = {k: v.format(root=root) for k, v in tool_input.items()}
    assert tool_call_denial(root, tool, tool_input) is None


@pytest.mark.parametrize(
    "tool,tool_input",
    [
        ("Read", {"file_path": "/etc/passwd"}),
        ("Read", {"file_path": "../secret.env"}),
        ("Read", {"file_path": "~/.ssh/id_rsa"}),
        ("Read", {"file_path": "challenger/link.env"}),
        ("Read", {"file_path": "challenger/linkdir/secret.env"}),
        ("Glob", {"pattern": "*", "path": ".."}),
        ("Glob", {"pattern": "/root/**"}),
        ("Glob", {"pattern": "../*.env"}),
        ("Grep", {"pattern": "KEY", "path": "/"}),
        ("Grep", {"pattern": "KEY", "glob": "../**"}),
        ("Bash", {"command": "cat /etc/passwd"}),
        ("WebFetch", {"url": "https://example.com"}),
    ],
)
def test_calls_reaching_outside_are_denied(root, tool, tool_input):
    assert tool_call_denial(root, tool, tool_input)


def test_hook_returns_a_deny_decision(root):
    pytest.importorskip("claude_agent_sdk")
    [matcher] = confine_to_cwd_hooks(root)["PreToolUse"]
    [hook] = matcher.hooks

    denied = asyncio.run(hook({"tool_name": "Read", "tool_input": {"file_path": "/etc/passwd"}}, None, {}))
    allowed = asyncio.run(hook({"tool_name": "Read", "tool_input": {"file_path": "challenger/train.py"}}, None, {}))

    assert denied["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert allowed == {}


def test_agent_env_keeps_only_what_the_cli_needs():
    env = agent_env(
        {
            "PATH": "/bin",
            "HOME": "/home/v",
            "ANTHROPIC_API_KEY": "sk-ant",
            "FRONTEND_API_KEY": "f",
            "ADMIN_API_KEY": "a",
            "POSTGRES_PASSWORD": "p",
            "S3_COMPATIBLE_SECRET_KEY": "s",
            "PWD": "/x",
        }
    )
    assert env["PATH"] == "/bin" and env["HOME"] == "/home/v" and env["ANTHROPIC_API_KEY"] == "sk-ant"
    for secret in ("FRONTEND_API_KEY", "ADMIN_API_KEY", "POSTGRES_PASSWORD", "S3_COMPATIBLE_SECRET_KEY"):
        assert env[secret] == ""
    assert "PWD" not in env
