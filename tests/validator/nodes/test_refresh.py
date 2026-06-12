from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from validator.nodes import refresh


@pytest.mark.asyncio
async def test_fetch_nodes_from_substrate_uses_temporary_substrate(monkeypatch):
    temp_substrate = Mock()
    nodes = [Mock()]
    calls = []

    def fake_get_substrate(_config, url):
        calls.append(("get_substrate", _config, url))
        return temp_substrate

    def fake_get_nodes_for_uid(substrate, netuid):
        calls.append(("get_nodes_for_uid", substrate, netuid))
        return nodes

    monkeypatch.setattr(refresh.interface, "get_substrate", fake_get_substrate)
    monkeypatch.setattr(refresh.fetch_nodes, "_get_nodes_for_uid", fake_get_nodes_for_uid)

    config = SimpleNamespace(substrate=SimpleNamespace(url="ws://substrate.test"), netuid=42)

    result = await refresh._fetch_nodes_from_substrate(config)

    assert result == nodes
    assert calls == [
        ("get_substrate", None, "ws://substrate.test"),
        ("get_nodes_for_uid", temp_substrate, 42),
    ]
    temp_substrate.close.assert_called_once_with()
