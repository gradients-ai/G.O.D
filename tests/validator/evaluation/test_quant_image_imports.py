import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
# The scorer's own code imports none of these. The training stack behind validator.evaluation.common
# and fiber (core.logging, whose pydantic pin conflicts with vLLM's) are not in the quant image;
# torch and transformers are, but only vLLM may pull them in.
FORBIDDEN_IMPORTS = (
    "torch",
    "transformers",
    "axolotl",
    "peft",
    "accelerate",
    "datasets",
    "trl",
    "fiber",
    "core.logging",
    "validator.evaluation.common",
)

IMPORT_SCRIPT = f"""
import sys
from tests.validator.evaluation import fake_vllm

sys.modules.update(fake_vllm.fake_modules(heuristics_config=lambda *args, **kwargs: {{}}))
import validator.evaluation.evaluators.quant
import validator.evaluation.evaluators.single_quant
print(sorted(name for name in {FORBIDDEN_IMPORTS!r} if name in sys.modules))
"""


def test_quant_scorer_imports_without_validator_stack():
    # A fresh interpreter, so modules other tests imported cannot hide an import.
    result = subprocess.run([sys.executable, "-c", IMPORT_SCRIPT], cwd=REPO_ROOT, capture_output=True, text=True, check=True)

    assert result.stdout.strip() == "[]"
