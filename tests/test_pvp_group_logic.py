"""Tests for PvP group evaluation setup logic: LoRA detection,
multi-LoRA CLI arg building, and full-weight exclusion.
"""

from unittest.mock import patch

from core.models.pvp_models import PvPGroupModelSpec


# --- 4a: _detect_lora_names ---


class TestDetectLoraNames:
    def test_all_lora(self):
        models = [
            PvPGroupModelSpec(repo="org/lora-a", hotkey="hk_a"),
            PvPGroupModelSpec(repo="org/lora-b", hotkey="hk_b"),
            PvPGroupModelSpec(repo="org/lora-c", hotkey="hk_c"),
        ]
        with (
            patch("validator.evaluation.pvp.group._repo_exists", return_value=True),
            patch("validator.evaluation.pvp.group.check_for_lora", return_value=True),
        ):
            from validator.evaluation.pvp.group import _detect_lora_names
            names, missing = _detect_lora_names(models)

        assert missing == []
        assert names["org/lora-a"] == "lora_0"
        assert names["org/lora-b"] == "lora_1"
        assert names["org/lora-c"] == "lora_2"

    def test_mixed_lora_and_full_weight(self):
        models = [
            PvPGroupModelSpec(repo="org/lora-a", hotkey="hk_a"),
            PvPGroupModelSpec(repo="org/full-weight", hotkey="hk_b"),
            PvPGroupModelSpec(repo="org/lora-c", hotkey="hk_c"),
        ]
        side_effects = [True, False, True]
        with (
            patch("validator.evaluation.pvp.group._repo_exists", return_value=True),
            patch("validator.evaluation.pvp.group.check_for_lora", side_effect=side_effects),
        ):
            from validator.evaluation.pvp.group import _detect_lora_names
            names, missing = _detect_lora_names(models)

        assert missing == []
        assert names["org/lora-a"] == "lora_0"
        assert names["org/full-weight"] == ""
        assert names["org/lora-c"] == "lora_2"

    def test_all_full_weight(self):
        models = [
            PvPGroupModelSpec(repo="org/full-a", hotkey="hk_a"),
            PvPGroupModelSpec(repo="org/full-b", hotkey="hk_b"),
        ]
        with (
            patch("validator.evaluation.pvp.group._repo_exists", return_value=True),
            patch("validator.evaluation.pvp.group.check_for_lora", return_value=False),
        ):
            from validator.evaluation.pvp.group import _detect_lora_names
            names, missing = _detect_lora_names(models)

        assert missing == []
        assert all(v == "" for v in names.values())


# --- 4b: _build_multi_lora_args ---


class TestBuildMultiLoraArgs:
    def test_three_lora_adapters(self):
        from validator.evaluation.pvp.group import _build_multi_lora_args
        lora_names = {
            "org/repo-a": "lora_0",
            "org/repo-b": "lora_1",
            "org/repo-c": "lora_2",
        }
        result = _build_multi_lora_args(lora_names)

        assert "--enable-lora" in result
        assert "--lora-backend triton" in result
        assert "lora_0=org/repo-a" in result
        assert "lora_1=org/repo-b" in result
        assert "lora_2=org/repo-c" in result

    def test_full_weight_entries_excluded(self):
        from validator.evaluation.pvp.group import _build_multi_lora_args
        lora_names = {
            "org/repo-a": "lora_0",
            "org/full-weight": "",  # full weight, should be excluded
        }
        result = _build_multi_lora_args(lora_names)

        assert "lora_0=org/repo-a" in result
        assert "full-weight" not in result

    def test_no_lora_returns_empty(self):
        from validator.evaluation.pvp.group import _build_multi_lora_args
        lora_names = {
            "org/full-a": "",
            "org/full-b": "",
        }
        result = _build_multi_lora_args(lora_names)
        assert result == ""


# --- 4c: Full-weight exclusion in group eval ---


class TestFullWeightExclusion:
    def test_full_weight_models_flagged(self):
        """When some models are full-weight, they should be in full_weight_fallbacks."""
        from validator.evaluation.pvp.group import _detect_lora_names

        models = [
            PvPGroupModelSpec(repo="org/lora-a", hotkey="hk_a"),
            PvPGroupModelSpec(repo="org/full-b", hotkey="hk_b"),
            PvPGroupModelSpec(repo="org/lora-c", hotkey="hk_c"),
        ]
        with (
            patch("validator.evaluation.pvp.group._repo_exists", return_value=True),
            patch("validator.evaluation.pvp.group.check_for_lora", side_effect=[True, False, True]),
        ):
            names, missing = _detect_lora_names(models)

        assert missing == []
        full_weight = [spec for spec in models if not names[spec.repo]]
        lora_specs = [spec for spec in models if names[spec.repo]]

        assert len(full_weight) == 1
        assert full_weight[0].hotkey == "hk_b"
        assert len(lora_specs) == 2
