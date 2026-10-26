"""The validator's code no longer depends on axolotl (static checks, no imports needed).

A static scan rather than an import probe: the golden venv has axolotl installed (to have captured the fixtures), so an
accidental import would succeed there. Docstrings and comments in the eval data path may still name axolotl - they record
which behaviour was replicated and why - but no code may import it, reference it, or load it by name.
"""

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
VALIDATOR = REPO_ROOT / "validator"
EVALUATION = VALIDATOR / "evaluation"
# An LLM prompt describing what miner repos may legitimately depend on; prose, not a dependency.
PROSE_ONLY = {VALIDATOR / "infrastructure" / "repo_dedup.py"}


def _python_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)


def _docstring_nodes(tree: ast.AST) -> set[int]:
    nodes = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                nodes.add(id(first.value))
    return nodes


def _axolotl_code_references(path: Path) -> list[str]:
    """Imports of, names containing, and non-docstring strings mentioning axolotl."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    docstrings = _docstring_nodes(tree)
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [f"line {node.lineno}: import {alias.name}" for alias in node.names if "axolotl" in alias.name]
        elif isinstance(node, ast.ImportFrom) and node.module and "axolotl" in node.module:
            found.append(f"line {node.lineno}: from {node.module} import ...")
        elif isinstance(node, ast.Name) and "axolotl" in node.id.lower():
            found.append(f"line {node.lineno}: name {node.id}")
        elif isinstance(node, ast.Attribute) and "axolotl" in node.attr.lower():
            found.append(f"line {node.lineno}: attribute {node.attr}")
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and "axolotl" in node.value.lower()
            and id(node) not in docstrings
        ):
            found.append(f"line {node.lineno}: string {node.value[:60]!r}")
    return found


def test_no_validator_module_imports_axolotl():
    offenders = {}
    for path in _python_files(VALIDATOR):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            if any(name == "axolotl" or name.startswith("axolotl.") for name in names):
                offenders.setdefault(str(path.relative_to(REPO_ROOT)), []).append(node.lineno)
    assert not offenders


def test_eval_code_never_references_axolotl_outside_docs():
    """Covers lazy loads too (importlib.import_module("axolotl..."), a config path string)."""
    offenders = {
        str(path.relative_to(REPO_ROOT)): refs
        for path in _python_files(EVALUATION) + [VALIDATOR / "infrastructure" / "service_constants.py"]
        if (refs := _axolotl_code_references(path))
    }
    assert not offenders


def test_only_known_prose_mentions_axolotl_outside_the_eval_package():
    mentioning = {
        path
        for path in _python_files(VALIDATOR)
        if EVALUATION not in path.parents and "axolotl" in path.read_text(encoding="utf-8").lower()
    }
    assert mentioning <= PROSE_ONLY


def test_axolotl_eval_config_is_gone():
    assert not (VALIDATOR / "assets" / "test_axolotl.yml").exists()
    service_constants = (VALIDATOR / "infrastructure" / "service_constants.py").read_text(encoding="utf-8")
    assert "VALI_CONFIG_PATH" not in service_constants
    for path in _python_files(VALIDATOR):
        source = path.read_text(encoding="utf-8")
        assert "DictDefault" not in source, path
        assert "test_axolotl.yml" not in source, path
