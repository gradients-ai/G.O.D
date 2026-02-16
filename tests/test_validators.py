"""Tests for core.validators module — input validation utilities."""

import pytest

from core.validators import InputValidators


class TestYesNoValidator:
    """Tests for InputValidators.yes_no."""

    @pytest.mark.parametrize("value", ["y", "Y", "yes", "Yes", "YES", "n", "N", "no", "No", "NO"])
    def test_valid_inputs(self, value):
        assert InputValidators.yes_no(value) is True

    def test_empty_string_is_valid(self):
        """Empty string is valid (allows default behavior)."""
        assert InputValidators.yes_no("") is True

    @pytest.mark.parametrize("value", ["maybe", "yep", "nah", "1", "0"])
    def test_invalid_inputs(self, value):
        assert InputValidators.yes_no(value) is False


class TestNonEmptyValidator:
    """Tests for InputValidators.non_empty."""

    def test_non_empty_string(self):
        assert InputValidators.non_empty("hello") is True

    def test_empty_string(self):
        assert InputValidators.non_empty("") is False

    def test_whitespace_only(self):
        assert InputValidators.non_empty("   ") is False

    def test_string_with_content_and_spaces(self):
        assert InputValidators.non_empty("  hello  ") is True


class TestNumberValidator:
    """Tests for InputValidators.number."""

    @pytest.mark.parametrize("value", ["0", "1", "42", "999999"])
    def test_valid_numbers(self, value):
        assert InputValidators.number(value) is True

    @pytest.mark.parametrize("value", ["", "abc", "3.14", "-1", "1e5"])
    def test_invalid_numbers(self, value):
        assert InputValidators.number(value) is False


class TestFloatNumberValidator:
    """Tests for InputValidators.float_number."""

    @pytest.mark.parametrize("value", ["0", "1", "3.14", "-2.5", "1e5", "0.001"])
    def test_valid_floats(self, value):
        assert InputValidators.float_number(value) is True

    @pytest.mark.parametrize("value", ["", "abc", "1.2.3", "inf_text"])
    def test_invalid_floats(self, value):
        assert InputValidators.float_number(value) is False


class TestWebsocketUrlValidator:
    """Tests for InputValidators.websocket_url."""

    def test_wss_url(self):
        assert InputValidators.websocket_url("wss://example.com/ws") is True

    def test_ws_url(self):
        assert InputValidators.websocket_url("ws://localhost:8080") is True

    def test_none_is_valid(self):
        assert InputValidators.websocket_url(None) is True

    def test_empty_string_is_valid(self):
        assert InputValidators.websocket_url("") is True

    def test_http_url_invalid(self):
        assert InputValidators.websocket_url("http://example.com") is False

    def test_plain_string_invalid(self):
        assert InputValidators.websocket_url("not-a-url") is False


class TestHttpUrlValidator:
    """Tests for InputValidators.http_url."""

    def test_http_url(self):
        assert InputValidators.http_url("http://example.com") is True

    def test_https_url(self):
        assert InputValidators.http_url("https://example.com/path") is True

    def test_ws_url_invalid(self):
        assert InputValidators.http_url("ws://example.com") is False

    def test_empty_string_invalid(self):
        assert InputValidators.http_url("") is False

    def test_no_host_invalid(self):
        assert InputValidators.http_url("https://") is False
