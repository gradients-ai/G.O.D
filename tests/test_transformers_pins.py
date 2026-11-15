"""The validator and the quant evaluator image run the same transformers.

Compression prep tokenises the test set and labels its chat turns in the validator (the `gpu` extra); the quant
evaluator image loads the same tokenizer and scores those token ids. A different transformers on either side can
render a chat template or add special tokens differently, so the two pins are held equal here.
"""

import re
from importlib import metadata
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
QUANT_EVALUATOR_REQUIREMENTS = REPO_ROOT / "ops" / "docker" / "requirements" / "quant-evaluator.txt"


def _pyproject_transformers_pins() -> list[str]:
    return re.findall(r'"transformers==([^"]+)"', PYPROJECT.read_text())


def _quant_evaluator_transformers_pins() -> list[str]:
    return re.findall(r"^transformers==(\S+)\s*$", QUANT_EVALUATOR_REQUIREMENTS.read_text(), flags=re.MULTILINE)


def test_the_validator_pins_the_quant_evaluator_images_transformers():
    validator_pins = _pyproject_transformers_pins()
    evaluator_pins = _quant_evaluator_transformers_pins()

    assert len(validator_pins) == 1 and len(evaluator_pins) == 1
    assert validator_pins == evaluator_pins


def test_the_installed_transformers_is_the_pinned_one():
    """The prep/eval parity tests only mean something on the pinned version."""
    try:
        installed = metadata.version("transformers")
    except metadata.PackageNotFoundError:
        pytest.skip("transformers is not installed (the `gpu` extra)")

    assert [installed] == _pyproject_transformers_pins()
