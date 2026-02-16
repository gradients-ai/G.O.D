"""Tests for validator.utils.reward_functions — reward function validation and processing."""

import pytest

from validator.utils.reward_functions import (
    extract_docstring,
    extract_function_name,
    validate_reward_function,
    process_reward_function_code,
    supports_extra_data,
    restricted_execution,
)


class TestSupportsExtraData:
    """Tests for supports_extra_data."""

    def test_function_with_extra_data(self):
        def func(completions, extra_data=None):
            pass
        assert supports_extra_data(func) is True

    def test_function_without_extra_data(self):
        def func(completions):
            pass
        assert supports_extra_data(func) is False

    def test_function_with_kwargs(self):
        def func(completions, **kwargs):
            pass
        assert supports_extra_data(func) is False


class TestValidateRewardFunction:
    """Tests for validate_reward_function."""

    def test_valid_simple_function(self):
        code = """
def reward_length(completions, **kwargs):
    return [len(c) for c in completions]
"""
        is_valid, error, func = validate_reward_function(code)
        assert is_valid is True
        assert error == ""
        assert func is not None
        assert func(["hello", "world"]) == [5, 5]

    def test_invalid_returns_non_list(self):
        code = """
def reward_bad(completions, **kwargs):
    return 42
"""
        is_valid, error, func = validate_reward_function(code)
        assert is_valid is False
        assert "list" in error.lower()

    def test_invalid_returns_wrong_length(self):
        code = """
def reward_wrong_len(completions, **kwargs):
    return [1.0]
"""
        is_valid, error, func = validate_reward_function(code)
        assert is_valid is False

    def test_invalid_returns_non_numeric(self):
        code = """
def reward_strings(completions, **kwargs):
    return ["good" for c in completions]
"""
        is_valid, error, func = validate_reward_function(code)
        assert is_valid is False

    def test_syntax_error(self):
        code = "def broken(:"
        is_valid, error, func = validate_reward_function(code)
        assert is_valid is False
        assert func is None

    def test_runtime_error(self):
        code = """
def reward_crash(completions, **kwargs):
    raise ValueError("boom")
"""
        is_valid, error, func = validate_reward_function(code)
        assert is_valid is False
        assert "boom" in error


class TestExtractFunctionName:
    """Tests for extract_function_name."""

    def test_simple_function(self):
        assert extract_function_name("def my_reward(completions):") == "my_reward"

    def test_function_with_decorators(self):
        code = "@decorator\ndef decorated_func(x, y):\n    pass"
        assert extract_function_name(code) == "decorated_func"

    def test_no_function(self):
        assert extract_function_name("x = 42") == "unknown_function"

    def test_multiple_functions_returns_first(self):
        code = "def first():\n    pass\ndef second():\n    pass"
        assert extract_function_name(code) == "first"


class TestExtractDocstring:
    """Tests for extract_docstring."""

    def test_double_quote_docstring(self):
        code = '"""This is a docstring."""'
        assert extract_docstring(code) == "This is a docstring."

    def test_single_quote_docstring(self):
        code = "'''Single quotes docstring.'''"
        assert extract_docstring(code) == "Single quotes docstring."

    def test_multiline_docstring(self):
        code = '"""Line one.\n    Line two.\n    Line three."""'
        result = extract_docstring(code)
        assert "Line one." in result
        assert "Line three." in result

    def test_no_docstring(self):
        code = "def func():\n    return 42"
        assert extract_docstring(code) == "No description available"


class TestRestrictedExecution:
    """Tests for restricted_execution."""

    def test_simple_print(self):
        output, error = restricted_execution("print('hello')", "")
        assert "hello" in output

    def test_with_input(self):
        code = "x = input()\nprint(x)"
        output, error = restricted_execution(code, "test_input")
        assert "test_input" in output

    def test_builtin_functions_available(self):
        code = "print(sum([1, 2, 3]))"
        output, error = restricted_execution(code, "")
        assert "6" in output

    def test_dangerous_code_blocked(self):
        code = "import os\nos.system('echo hacked')"
        output, error = restricted_execution(code, "")
        assert error != ""


class TestProcessRewardFunctionCode:
    """Tests for process_reward_function_code."""

    def test_fixes_function_signature(self):
        code = """
def my_reward(texts):
    return [1.0 for t in texts]
"""
        result = process_reward_function_code(code)
        assert "completions" in result
        assert "**kwargs" in result or "kwargs" in result

    def test_preserves_function_body(self):
        code = """
def reward_func(completions, **kwargs):
    scores = [len(c) / 100.0 for c in completions]
    return scores
"""
        result = process_reward_function_code(code)
        assert "scores" in result
        assert "len(c)" in result or "len" in result

    def test_handles_invalid_code_gracefully(self):
        code = "this is not valid python{{{}"
        result = process_reward_function_code(code)
        # Should return original code on failure
        assert result == code
