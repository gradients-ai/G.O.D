"""Refunds: only a cancelled tournament, in one transaction that holds a row lock on the tournament."""

import pytest

from validator.db.sql import transfers
from validator.db.sql.transfers import RefundRefused
from validator.db.sql.transfers import refund_tournament_participants


class FakeConnection:
    def __init__(self, status: str | None, net: dict[str, int], balance_update: str = "UPDATE 1"):
        self.status = status
        self.net = net
        self.balance_update = balance_update
        self.statements: list[str] = []
        self.in_transaction = False
        self.committed = False
        self.inserted: list[tuple] = []
        self.credited: list[tuple] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False

    def transaction(self):
        conn = self

        class Tx:
            async def __aenter__(self):
                conn.in_transaction = True

            async def __aexit__(self, exc_type, *_):
                conn.committed = exc_type is None
                if exc_type is not None:
                    conn.inserted.clear()
                    conn.credited.clear()
                return False

        return Tx()

    async def fetchrow(self, query, *args):
        assert self.in_transaction
        self.statements.append(query)
        return None if self.status is None else {"status": self.status}

    async def fetch(self, query, *args):
        assert self.in_transaction
        self.statements.append(query)
        return [{"coldkey": k, "net_rao": v} for k, v in self.net.items()]

    async def execute(self, query, *args):
        assert self.in_transaction
        self.statements.append(query)
        if "INSERT INTO balance_events" in query:
            self.inserted.append(args)
            return "INSERT 0 1"
        self.credited.append(args)
        return self.balance_update


class FakeDB:
    def __init__(self, connection: FakeConnection):
        self._connection = connection

    async def connection(self):
        return self._connection


@pytest.mark.parametrize("status", ["pending", "active", "completed"])
async def test_refuses_a_tournament_that_is_not_cancelled(status):
    conn = FakeConnection(status, {"ck": -100})
    with pytest.raises(RefundRefused, match=status):
        await refund_tournament_participants(FakeDB(conn), "t1")
    assert conn.inserted == [] and conn.credited == []


async def test_refuses_an_unknown_tournament():
    conn = FakeConnection(None, {"ck": -100})
    with pytest.raises(RefundRefused, match="not found"):
        await refund_tournament_participants(FakeDB(conn), "t1")


async def test_locks_the_tournament_row_first_and_refunds_the_net_amounts():
    conn = FakeConnection("cancelled", {"paid": -100, "refunded": 0})

    assert await refund_tournament_participants(FakeDB(conn), "t1") == 1

    assert "FOR UPDATE" in conn.statements[0]
    assert [args[:3] for args in conn.inserted] == [("t1", "paid", 100)]
    assert conn.credited == [("paid", 100)]
    assert conn.committed


async def test_a_missing_balance_row_rolls_back_everything():
    conn = FakeConnection("cancelled", {"a": -100, "b": -50}, balance_update="UPDATE 0")
    with pytest.raises(RuntimeError):
        await refund_tournament_participants(FakeDB(conn), "t1")
    assert not conn.committed
    assert conn.inserted == [] and conn.credited == []


def test_refund_sql_sums_every_event_not_a_page_of_them():
    import inspect

    source = inspect.getsource(transfers.refund_tournament_participants)
    assert "LIMIT" not in source and "SUM(amount_rao)" in source
