from typing import Protocol
from typing import runtime_checkable


@runtime_checkable
class KeypairLike(Protocol):
    """Minimal signing interface used by chain-free task preparation."""

    @property
    def ss58_address(self) -> str: ...

    def sign(self, data: bytes | str) -> bytes: ...
