"""The trainer's Vector log handler must never raise, recurse or spam when the Vector sink is absent."""

import logging
import socket

import pytest
import requests

from trainer import telemetry
from trainer.telemetry import VectorHandler


def _record(msg: str = "hello", args: tuple = ()) -> logging.LogRecord:
    return logging.LogRecord("trainer", logging.INFO, __file__, 1, msg, args, None)


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch) -> FakeClock:
    clock = FakeClock()
    monkeypatch.setattr(telemetry.time, "monotonic", clock.monotonic)
    return clock


class FakePost:
    def __init__(self) -> None:
        self.calls = 0
        self.fail = True

    def __call__(self, url, json, timeout):
        self.calls += 1
        if self.fail:
            raise requests.exceptions.ConnectionError("Connection refused")


@pytest.fixture
def post(monkeypatch) -> FakePost:
    post = FakePost()
    monkeypatch.setattr(telemetry.requests, "post", post)
    return post


def _closed_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_absent_sink_never_raises_nor_calls_handle_error(monkeypatch, capsys):
    handler = VectorHandler(f"http://127.0.0.1:{_closed_port()}")
    monkeypatch.setattr(handler, "handleError", lambda record: pytest.fail("handleError called"))
    logger = logging.getLogger("test-telemetry-real-socket")
    logger.propagate = False
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    try:
        for i in range(5):
            logger.info("message %d", i)
    finally:
        logger.removeHandler(handler)

    err = capsys.readouterr().err
    assert "Traceback" not in err
    assert err.count("unreachable") == 1


def test_outage_backs_off_then_recovers(clock, post, capsys):
    handler = VectorHandler("http://sink", retry_seconds=60)

    handler.emit(_record())
    handler.emit(_record())
    clock.now += 59
    handler.emit(_record())
    assert post.calls == 1  # one failed connect, then records are dropped without touching the network

    clock.now += 2
    handler.emit(_record())
    assert post.calls == 2  # retried after the window, still down: still one stderr line

    post.fail = False
    clock.now += 61
    handler.emit(_record())
    handler.emit(_record())
    assert post.calls == 4

    err = capsys.readouterr().err
    assert err.count("unreachable") == 1
    assert err.count("reachable again") == 1


def test_malformed_record_is_dropped_without_marking_the_sink_down(clock, post, capsys):
    post.fail = False
    handler = VectorHandler("http://sink")

    handler.emit(_record("%d", ("not-a-number",)))
    handler.emit(_record())

    assert post.calls == 1
    assert capsys.readouterr().err == ""


def test_non_json_extras_are_stringified(clock, monkeypatch):
    sent = []
    monkeypatch.setattr(telemetry.requests, "post", lambda url, json, timeout: sent.append(json))
    record = _record()
    record.task_id = "t1"
    record.path = object()

    VectorHandler("http://sink").emit(record)

    assert sent[0]["message"] == "hello"
    assert sent[0]["task_id"] == "t1"
    assert isinstance(sent[0]["path"], str)
