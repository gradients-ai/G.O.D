import json

from core.pvp import sglang_server


def test_olmo_hybrid_config_dispatches_to_transformers_fallback(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "olmo_hybrid"}))
    calls = []
    monkeypatch.setattr(sglang_server, "_run_transformers_server", lambda arguments: calls.append(("transformers", arguments)))
    monkeypatch.setattr(sglang_server, "_run_sglang_server", lambda arguments: calls.append(("sglang", arguments)))

    arguments = ["--model-path", str(tmp_path), "--port", "30000"]
    sglang_server.main(arguments)

    assert calls == [("transformers", arguments)]


def test_non_hybrid_config_still_launches_sglang(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "olmo3"}))
    calls = []
    monkeypatch.setattr(sglang_server, "_run_transformers_server", lambda arguments: calls.append(("transformers", arguments)))
    monkeypatch.setattr(sglang_server, "_run_sglang_server", lambda arguments: calls.append(("sglang", arguments)))

    arguments = [f"--model-path={tmp_path}", "--port", "30000"]
    sglang_server.main(arguments)

    assert calls == [("sglang", arguments)]

