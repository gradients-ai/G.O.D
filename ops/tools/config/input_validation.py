import re
from typing import Callable
from typing import TypeVar


T = TypeVar("T")


def validate_input(prompt: str, validator: Callable[[str], bool], default: str | None = None) -> str:
    """Validate user input with a given validator function."""
    while True:
        value = input(prompt)
        if not value and default is not None:
            return default
        if validator(value):
            return value
        print("Invalid input. Please try again.")


class InputValidators:
    @staticmethod
    def yes_no(value: str) -> bool:
        return value.lower() in ["y", "n", "yes", "no"] or not value

    @staticmethod
    def non_empty(value: str) -> bool:
        return bool(value.strip())

    @staticmethod
    def number(value: str) -> bool:
        return value.isdigit()

    @staticmethod
    def float_number(value: str) -> bool:
        try:
            float(value)
            return True
        except ValueError:
            return False

    @staticmethod
    def websocket_url(value: str | None) -> bool:
        if not value:
            return True
        return bool(re.match(r"^wss?://", value))

    @staticmethod
    def http_url(value: str) -> bool:
        return bool(re.match(r"^https?://.+", value))
