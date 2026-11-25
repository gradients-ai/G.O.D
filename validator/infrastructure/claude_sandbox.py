"""Limits for the read-only Claude agents that review miner repositories.

The agents run the Claude Code CLI with Read/Glob/Grep over a checkout. Two limits apply on top of the tool allowlist:

- `agent_env()`: the CLI process sees only the variables it needs. The SDK merges `ClaudeAgentOptions.env` over the
  validator's own environment, so every other variable is overridden with an empty value.
- `confine_to_cwd_hooks(cwd)`: a PreToolUse hook that denies any tool call whose path resolves (after `~`, `..` and
  symlinks) outside the agent's working directory, and any tool other than Read, Glob and Grep. Hooks run for every
  tool call, including tools already allowed by `allowed_tools`.
"""

import os
from pathlib import Path
from typing import Any

from core.logging import get_logger


logger = get_logger(__name__)

# What the Claude Code CLI needs to start, authenticate and reach the API.
AGENT_ENV_KEEP = frozenset(
    {
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "SHELL",
        "LANG",
        "LC_ALL",
        "TERM",
        "TMPDIR",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_BASE_URL",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "no_proxy",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "NODE_EXTRA_CA_CERTS",
    }
)
# Set by the SDK itself; left alone.
_SDK_OWNED_ENV = frozenset({"CLAUDE_CODE_ENTRYPOINT", "CLAUDE_AGENT_SDK_VERSION", "PWD"})

READ_ONLY_TOOLS = ("Read", "Glob", "Grep")
_PATH_KEYS = ("file_path", "path", "notebook_path")
_GLOB_KEYS = {"Glob": ("pattern",), "Grep": ("glob",)}


def agent_env(environ: dict[str, str] | None = None) -> dict[str, str]:
    """`ClaudeAgentOptions.env` for a review agent: the kept variables as they are, every other one emptied."""
    source = dict(os.environ if environ is None else environ)
    env = {key: "" for key in source if key not in AGENT_ENV_KEEP and key not in _SDK_OWNED_ENV}
    env.update({key: value for key, value in source.items() if key in AGENT_ENV_KEEP})
    return env


def _resolves_inside(root: Path, candidate: str) -> bool:
    path = Path(os.path.expanduser(candidate))
    if not path.is_absolute():
        path = root / path
    resolved = Path(os.path.realpath(path))
    return resolved == root or root in resolved.parents


def _glob_escapes(pattern: str) -> bool:
    return pattern.startswith(("/", "~")) or ".." in Path(pattern).parts


def tool_call_denial(root: Path, tool_name: str, tool_input: dict[str, Any]) -> str | None:
    """Why the call is denied, or None when it stays inside `root`. `root` must already be resolved."""
    if tool_name not in READ_ONLY_TOOLS:
        return f"{tool_name} is not available to this review"
    for key in _PATH_KEYS:
        value = tool_input.get(key)
        if isinstance(value, str) and value and not _resolves_inside(root, value):
            return f"{key} {value!r} is outside the repository under review"
    for key in _GLOB_KEYS.get(tool_name, ()):
        value = tool_input.get(key)
        if isinstance(value, str) and value and _glob_escapes(value):
            return f"{key} {value!r} reaches outside the repository under review"
    return None


def confine_to_cwd_hooks(cwd: Path | str) -> dict[str, list]:
    """`ClaudeAgentOptions.hooks` that keep every tool call inside `cwd`."""
    from claude_agent_sdk import HookMatcher

    root = Path(os.path.realpath(cwd))

    async def deny_outside_cwd(input_data: dict[str, Any], tool_use_id: str | None, context: Any) -> dict[str, Any]:
        reason = tool_call_denial(root, input_data.get("tool_name", ""), input_data.get("tool_input") or {})
        if reason is None:
            return {}
        logger.warning(f"Denied review agent tool call: {reason}")
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": reason,
            }
        }

    return {"PreToolUse": [HookMatcher(matcher=None, hooks=[deny_outside_cwd])]}
