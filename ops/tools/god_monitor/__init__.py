"""G.O.D Tournament Monitor — a read-only CLI for watching tournaments live."""

__all__ = ["main"]


def main() -> None:
    from ops.tools.god_monitor.cli import main as _main

    _main()
