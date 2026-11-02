"""Materialize third-party dependencies for challenger Claude code review.

Discovers install specs from the live build path (Dockerfiles, requirements,
pyproject), downloads non-allowlisted artifacts, unpacks them safely into
`_dependencies/`, and writes machine findings under `_manifests/`.

This is intentionally package-agnostic: any non-allowlisted dependency on the
build/data path is exposed to Claude the same way first-party code is.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tarfile
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any

from core.logging import get_logger


logger = get_logger(__name__)

# Common trainer stack — still recorded in the manifest, but not unpacked (too large /
# already well-known). Everything else on the install path is materialized for review.
DEFAULT_ALLOWLIST: frozenset[str] = frozenset(
    {
        "torch",
        "torchvision",
        "torchaudio",
        "transformers",
        "datasets",
        "accelerate",
        "peft",
        "trl",
        "deepspeed",
        "vllm",
        "flash-attn",
        "flash_attn",
        "axolotl",
        "numpy",
        "scipy",
        "pandas",
        "scikit-learn",
        "sklearn",
        "pillow",
        "opencv-python",
        "opencv-python-headless",
        "huggingface-hub",
        "huggingface_hub",
        "hf-transfer",
        "hf_transfer",
        "tokenizers",
        "safetensors",
        "sentencepiece",
        "protobuf",
        "requests",
        "urllib3",
        "aiohttp",
        "httpx",
        "tenacity",
        "tiktoken",
        "packaging",
        "setuptools",
        "wheel",
        "ninja",
        "pip",
        "uv",
        "pydantic",
        "pyyaml",
        "yaml",
        "tqdm",
        "rich",
        "click",
        "typer",
        "filelock",
        "fsspec",
        "jinja2",
        "networkx",
        "sympy",
        "einops",
        "bitsandbytes",
        "xformers",
        "triton",
        "wandb",
        "tensorboard",
        "sentence-transformers",
        "evaluate",
        "trl",
        "lm-eval",
        "lm_eval",
        # Subnet / trainer plumbing commonly vendored by miners
        "fiber",
        "docker",
        "toml",
        "debugpy",
        "minio",
        "mlflow",
        "mpi4py",
        "astor",
        "restrictedpython",
        "liger-kernel",
        "liger_kernel",
    }
)

# Dockerfile / shell keywords that must never be treated as package names.
_DOCKER_KEYWORDS = frozenset(
    {
        "add",
        "arg",
        "cmd",
        "copy",
        "entrypoint",
        "env",
        "expose",
        "from",
        "healthcheck",
        "label",
        "maintainer",
        "onbuild",
        "run",
        "shell",
        "stopsignal",
        "user",
        "volume",
        "workdir",
        "true",
        "false",
        "as",
        "then",
        "fi",
        "done",
        "esac",
        "source",
        "deactivate",
        "activate",
    }
)

_PIP_INSTALL_RE = re.compile(
    r"""(?x)
    (?:uv\s+pip\s+install|pip3?\s+install|python\s+-m\s+pip\s+install)
    (?P<flags>(?:\s+(?:--[a-zA-Z0-9\-]+(?:=[^\s]+)?|-[a-zA-Z]))*)
    \s+
    (?P<body>.+?)
    (?=\s*(?:&&|\\?$|;|\n))
    """,
    re.IGNORECASE | re.MULTILINE,
)

_REQ_TOKEN_RE = re.compile(
    r"""(?x)
    (?P<git>git\+https?://[^\s;]+)
    |
    (?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)
    (?:\[(?P<extras>[^\]]+)\])?
    (?P<spec>(?:==|!=|<=|>=|~=|<|>)[^;\s]+)?
    """,
)

_COMMIT_SHA_RE = re.compile(r"^[0-9a-f]{7,40}$", re.IGNORECASE)

# Suspicious package-data / injection heuristics (generic — not package-specific).
_DATA_NAME_HINTS = re.compile(
    r"(test|eval|benchmark|label|answer|gold|bias|holdout|leak)",
    re.IGNORECASE,
)
_CODE_HINTS = re.compile(
    r"(add_bias|bias_test|hash_mapping|fingerprint|package_data|files\(__file__\)|"
    r"importlib\.resources|pkgutil\.get_data|resources\.files)",
    re.IGNORECASE,
)


@dataclass
class RequirementSpec:
    raw: str
    name: str | None
    extras: list[str] = field(default_factory=list)
    version_spec: str | None = None
    git_url: str | None = None
    source_file: str = ""
    pinned: bool = False


@dataclass
class ResolvedPackage:
    name: str
    version: str | None
    status: str  # allowlisted | extracted | unresolved | git_extracted | skipped
    pinned: bool = False
    artifact_url: str | None = None
    sha256: str | None = None
    upload_time: str | None = None
    extract_path: str | None = None
    source_file: str = ""
    detail: str = ""
    file_count: int = 0
    unpacked_bytes: int = 0


@dataclass
class Finding:
    severity: str
    kind: str
    path: str
    detail: str


@dataclass
class DependencyMaterialization:
    packages: list[ResolvedPackage] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)

    @property
    def needs_review(self) -> bool:
        return bool(self.unresolved)


def _normalize_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _parse_requirement_token(token: str, source_file: str) -> RequirementSpec | None:
    token = token.strip().strip("\\\"'")
    if not token or token.startswith("#") or token.startswith("-"):
        # Skip flags like -r, --system, --no-build-isolation (handled by caller)
        return None
    if token.startswith("-r") or token.startswith("--requirement"):
        return None
    # Dockerfile instructions accidentally pulled into a token stream
    if token.split("[", 1)[0].strip().lower() in _DOCKER_KEYWORDS:
        return None

    git_match = re.match(r"git\+(\S+)", token)
    if git_match or token.startswith("https://github.com/") and ".git" in token:
        raw = token if token.startswith("git+") else f"git+{token}"
        url = raw[4:] if raw.startswith("git+") else raw
        # egg / package name
        name = None
        if "#egg=" in url:
            name = url.split("#egg=", 1)[1].split("&")[0]
            url = url.split("#", 1)[0]
        elif "@" in urllib.parse.urlparse(url).path:
            # github.com/org/repo@ref
            pass
        repo_name = urllib.parse.urlparse(url.split("@")[0]).path.rstrip("/").split("/")[-1]
        if repo_name.endswith(".git"):
            repo_name = repo_name[:-4]
        name = name or repo_name
        ref = None
        if "@" in url.split("#")[0]:
            ref = url.split("@", 1)[1].split("#")[0]
        pinned = bool(ref and _COMMIT_SHA_RE.match(ref))
        return RequirementSpec(
            raw=token,
            name=name,
            git_url=raw if raw.startswith("git+") else f"git+{url}",
            source_file=source_file,
            pinned=pinned,
            version_spec=ref,
        )

    # poetry/fiber style: "fiber @ git+https://..."
    at_git = re.match(r"([A-Za-z0-9_.-]+)\s*@\s*(git\+\S+)", token)
    if at_git:
        inner = _parse_requirement_token(at_git.group(2), source_file)
        if inner:
            inner.name = at_git.group(1)
            inner.raw = token
            return inner

    m = _REQ_TOKEN_RE.match(token)
    if not m or not m.group("name"):
        return None
    if m.group("git"):
        return _parse_requirement_token(m.group("git"), source_file)
    name = m.group("name")
    if name.lower() in _DOCKER_KEYWORDS:
        return None
    # PyPI names are case-insensitive but almost never ALLCAPS Dockerfile verbs
    if name.isupper() and len(name) > 2:
        return None
    extras = [e.strip() for e in (m.group("extras") or "").split(",") if e.strip()]
    spec = m.group("spec")
    pinned = bool(spec and spec.startswith("=="))
    return RequirementSpec(
        raw=token,
        name=name,
        extras=extras,
        version_spec=spec,
        source_file=source_file,
        pinned=pinned,
    )


def _tokenize_install_body(body: str) -> list[str]:
    # Strip line continuations and comments.
    cleaned = body.replace("\\\n", " ")
    cleaned = re.sub(r"\s+#.*", "", cleaned)
    parts: list[str] = []
    buf = ""
    quote = None
    for ch in cleaned:
        if quote:
            buf += ch
            if ch == quote:
                quote = None
            continue
        if ch in {'"', "'"}:
            quote = ch
            buf += ch
            continue
        if ch.isspace():
            if buf:
                parts.append(buf)
                buf = ""
            continue
        buf += ch
    if buf:
        parts.append(buf)
    # Drop install flags
    out: list[str] = []
    skip_next = False
    for part in parts:
        if skip_next:
            skip_next = False
            continue
        if part.startswith("-"):
            if part in {"-r", "--requirement", "-c", "--constraint", "-e", "--editable"}:
                skip_next = True
            continue
        out.append(part.strip("\\\"'"))
    return out


def discover_requirements(challenger_root: Path) -> list[RequirementSpec]:
    """Parse Dockerfiles / requirements / pyproject for third-party install specs."""
    found: list[RequirementSpec] = []
    seen: set[str] = set()

    def add(spec: RequirementSpec | None) -> None:
        if spec is None or not spec.name:
            return
        key = f"{_normalize_name(spec.name)}|{spec.version_spec}|{spec.git_url}|{spec.source_file}"
        if key in seen:
            return
        seen.add(key)
        found.append(spec)

    patterns = [
        "**/Dockerfile*",
        "**/*dockerfile*",
        "**/requirements*.txt",
        "**/training_requirements.txt",
        "**/pyproject.toml",
    ]
    files: list[Path] = []
    for pattern in patterns:
        files.extend(challenger_root.glob(pattern))

    for path in sorted(set(files)):
        if not path.is_file():
            continue
        rel = path.relative_to(challenger_root).as_posix()
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue

        if path.name == "pyproject.toml":
            # Minimal TOML scrape for dependencies = [ "..."] arrays
            for block in re.findall(
                r"(?:dependencies|requires)\s*=\s*\[(.*?)\]",
                text,
                flags=re.DOTALL | re.IGNORECASE,
            ):
                for item in re.findall(r'["\']([^"\']+)["\']', block):
                    add(_parse_requirement_token(item, rel))
            continue

        if path.suffix == ".txt" or "requirements" in path.name.lower():
            for line in text.splitlines():
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("-"):
                    continue
                add(_parse_requirement_token(line, rel))
            continue

        # Dockerfile / shell install lines
        for match in _PIP_INSTALL_RE.finditer(text):
            for token in _tokenize_install_body(match.group("body")):
                add(_parse_requirement_token(token, rel))

        # Also catch quoted forms: pip install "fiber @ git+..."
        for match in re.finditer(
            r"""(?:uv\s+pip\s+install|pip3?\s+install)[^\n]*["']([^"']+)["']""",
            text,
            flags=re.IGNORECASE,
        ):
            add(_parse_requirement_token(match.group(1), rel))

    return found


def _http_json(url: str, timeout: float = 30.0) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": "gradients-challenger-dep-review"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _download(url: str, dest: Path, timeout: float = 120.0) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "gradients-challenger-dep-review"})
    h = hashlib.sha256()
    with urllib.request.urlopen(req, timeout=timeout) as resp, dest.open("wb") as out:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
            out.write(chunk)
    return h.hexdigest()


def _safe_members_zip(zf: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    members = []
    for info in zf.infolist():
        name = info.filename.replace("\\", "/")
        if name.startswith("/") or name.startswith("../") or "/../" in f"/{name}/":
            continue
        members.append(info)
    return members


def _safe_members_tar(tf: tarfile.TarFile) -> list[tarfile.TarInfo]:
    members = []
    for info in tf.getmembers():
        name = info.name.replace("\\", "/")
        if name.startswith("/") or name.startswith("../") or "/../" in f"/{name}/":
            continue
        if info.issym() or info.islnk():
            continue
        members.append(info)
    return members


def _should_extract_member(name: str, size: int, data_budget: dict[str, int], limits: dict[str, int]) -> bool:
    lower = name.lower()
    if lower.endswith((".py", ".pyi", ".typed")):
        return True
    if "metadata" in lower or lower.endswith((".toml", ".cfg", ".ini")):
        return True
    if lower.endswith((".md", ".rst", ".txt", ".json", ".yaml", ".yml", ".csv", ".tsv")):
        # Cap large packaged datasets: always inventory; extract small files + samples.
        if size <= limits["max_data_file_bytes"]:
            if data_budget["bytes"] + size <= limits["max_data_bytes_total"]:
                data_budget["bytes"] += size
                data_budget["files"] += 1
                return True
            # Still extract a few oversized-but-interesting names for Claude to open
            if data_budget["files"] < limits["max_data_files"] and _DATA_NAME_HINTS.search(name):
                data_budget["files"] += 1
                return True
        elif data_budget["files"] < min(8, limits["max_data_files"]) and _DATA_NAME_HINTS.search(name):
            # Extract truncated? We extract full file only if under hard cap
            if size <= limits["max_single_extract_bytes"]:
                data_budget["files"] += 1
                data_budget["bytes"] += size
                return True
        return False
    return False


def _extract_wheel_or_sdist(
    artifact: Path,
    dest_dir: Path,
    limits: dict[str, int],
) -> tuple[int, int, list[dict[str, Any]]]:
    """Unpack source/config/small data; write full inventory. Returns file_count, bytes, inventory."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    inventory: list[dict[str, Any]] = []
    data_budget = {"bytes": 0, "files": 0}
    extracted_files = 0
    extracted_bytes = 0

    def record(name: str, size: int, extracted: bool) -> None:
        inventory.append({"path": name, "size": size, "extracted": extracted})

    if zipfile.is_zipfile(artifact):
        with zipfile.ZipFile(artifact) as zf:
            members = _safe_members_zip(zf)
            if len(members) > limits["max_member_count"]:
                raise RuntimeError(f"archive has too many members ({len(members)})")
            total = sum(m.file_size for m in members)
            if total > limits["max_unpacked_bytes"]:
                raise RuntimeError(f"archive unpack size {total} exceeds limit")
            for info in members:
                name = info.filename.replace("\\", "/")
                if name.endswith("/"):
                    continue
                extract = _should_extract_member(name, info.file_size, data_budget, limits)
                record(name, info.file_size, extract)
                if not extract:
                    continue
                target = dest_dir / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, target.open("wb") as out:
                    data = src.read()
                    out.write(data)
                extracted_files += 1
                extracted_bytes += len(data)
    elif tarfile.is_tarfile(artifact):
        with tarfile.open(artifact) as tf:
            members = [m for m in _safe_members_tar(tf) if m.isfile()]
            if len(members) > limits["max_member_count"]:
                raise RuntimeError(f"archive has too many members ({len(members)})")
            total = sum(m.size for m in members)
            if total > limits["max_unpacked_bytes"]:
                raise RuntimeError(f"archive unpack size {total} exceeds limit")
            for info in members:
                name = info.name.replace("\\", "/")
                extract = _should_extract_member(name, info.size, data_budget, limits)
                record(name, info.size, extract)
                if not extract:
                    continue
                target = dest_dir / name
                target.parent.mkdir(parents=True, exist_ok=True)
                src = tf.extractfile(info)
                if src is None:
                    continue
                with src, target.open("wb") as out:
                    data = src.read()
                    out.write(data)
                extracted_files += 1
                extracted_bytes += len(data)
    else:
        raise RuntimeError(f"unsupported artifact format: {artifact.name}")

    (dest_dir / "PACKAGE_FILE_INVENTORY.json").write_text(
        json.dumps(
            {
                "artifact": artifact.name,
                "member_count": len(inventory),
                "extracted_files": extracted_files,
                "extracted_bytes": extracted_bytes,
                "files": inventory,
            },
            indent=2,
        )
    )
    return extracted_files, extracted_bytes, inventory


def _select_pypi_artifact(project: dict[str, Any], version: str | None) -> tuple[str, dict[str, Any]]:
    releases = project.get("releases") or {}
    if version and version in releases:
        files = releases[version]
        ver = version
    else:
        ver = (project.get("info") or {}).get("version")
        files = releases.get(ver) or []
        # Fallback: urls on info
        if not files:
            files = (project.get("urls") or [])
    if not files:
        raise RuntimeError(f"no files for version {ver}")
    # Prefer pure py3 wheel, then any wheel, then sdist
    pure = [f for f in files if str(f.get("packagetype")) == "bdist_wheel" and "py3-none-any" in f.get("filename", "")]
    wheels = [f for f in files if str(f.get("packagetype")) == "bdist_wheel"]
    sdists = [f for f in files if str(f.get("packagetype")) == "sdist"]
    choice = (pure or wheels or sdists)[0]
    return ver, choice


def _scan_extracted_package(extract_root: Path, package_name: str) -> list[Finding]:
    findings: list[Finding] = []
    inventory_path = extract_root / "PACKAGE_FILE_INVENTORY.json"
    if inventory_path.exists():
        inventory = json.loads(inventory_path.read_text())
        dataish = [
            f
            for f in inventory.get("files", [])
            if _DATA_NAME_HINTS.search(f.get("path", ""))
            and str(f.get("path", "")).lower().endswith((".json", ".csv", ".tsv", ".parquet", ".arrow", ".txt"))
        ]
        if len(dataish) >= 5:
            findings.append(
                Finding(
                    severity="high",
                    kind="packaged_data_files",
                    path=f"_dependencies/python/{extract_root.name}/",
                    detail=(
                        f"Package {package_name} ships {len(dataish)} packaged files whose names look like "
                        f"test/eval/benchmark/bias data (see PACKAGE_FILE_INVENTORY.json)."
                    ),
                )
            )

    for path in extract_root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(extract_root).as_posix()
        if path.suffix == ".py":
            try:
                text = path.read_text(errors="ignore")
            except OSError:
                continue
            for i, line in enumerate(text.splitlines(), 1):
                if _CODE_HINTS.search(line):
                    findings.append(
                        Finding(
                            severity="high",
                            kind="dependency_code_pattern",
                            path=f"_dependencies/python/{extract_root.name}/{rel}:{i}",
                            detail=line.strip()[:240],
                        )
                    )
        elif path.suffix.lower() in {".json", ".csv", ".tsv"} and _DATA_NAME_HINTS.search(path.name):
            findings.append(
                Finding(
                    severity="medium",
                    kind="packaged_data_sample",
                    path=f"_dependencies/python/{extract_root.name}/{rel}",
                    detail=f"Packaged data file present ({path.stat().st_size} bytes)",
                )
            )
    return findings


def _scan_challenger_refs(challenger_root: Path, package_names: set[str]) -> list[Finding]:
    findings: list[Finding] = []
    # Only match real import/install references to non-allowlisted packages (normalized).
    norms = sorted({_normalize_name(n) for n in package_names}, key=len, reverse=True)
    import_res = [
        re.compile(rf"(?i)^\s*(?:from|import)\s+{_normalize_name(n).replace('-', '[-_]')}(?:[.\s]|$)")
        for n in norms
    ] if norms else []
    install_re = re.compile(
        r"(?i)(?:uv\s+pip\s+install|pip3?\s+install).*\b("
        + "|".join(re.escape(n) for n in norms)
        + r")\b"
    ) if norms else None

    for path in challenger_root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix not in {".py", ".sh", ".txt", ".toml", ".yml", ".yaml"} and "dockerfile" not in path.name.lower():
            continue
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        rel = path.relative_to(challenger_root).as_posix()
        for i, line in enumerate(text.splitlines(), 1):
            if _CODE_HINTS.search(line):
                findings.append(
                    Finding(
                        severity="critical",
                        kind="challenger_data_path_pattern",
                        path=f"challenger/{rel}:{i}",
                        detail=line.strip()[:240],
                    )
                )
                continue
            if any(r.search(line) for r in import_res) or (install_re and install_re.search(line)):
                findings.append(
                    Finding(
                        severity="medium",
                        kind="challenger_dependency_reference",
                        path=f"challenger/{rel}:{i}",
                        detail=line.strip()[:240],
                    )
                )
    return findings


def _local_artifact_dirs() -> list[Path]:
    """Optional offline/mirror dirs of wheels/sdists (colon-separated CHALLENGER_DEP_ARTIFACT_CACHE)."""
    raw = os.environ.get("CHALLENGER_DEP_ARTIFACT_CACHE", "").strip()
    if not raw:
        return []
    return [Path(part) for part in raw.split(":") if part.strip()]


def _find_local_artifact(name: str, version: str | None) -> Path | None:
    norm = _normalize_name(name).replace("-", "_")
    norm_dash = _normalize_name(name)
    candidates: list[Path] = []
    for directory in _local_artifact_dirs():
        if not directory.is_dir():
            continue
        for path in directory.glob("*"):
            if not path.is_file():
                continue
            lower = path.name.lower()
            if not (lower.endswith(".whl") or lower.endswith(".tar.gz") or lower.endswith(".zip")):
                continue
            if norm not in lower and norm_dash not in lower:
                continue
            if version and version.lower() not in lower:
                continue
            candidates.append(path)
    if not candidates:
        return None
    # Prefer exact version wheel if present
    candidates.sort(key=lambda p: (0 if p.suffix == ".whl" else 1, len(p.name)))
    return candidates[0]


def _extract_resolved_artifact(
    *,
    name: str,
    ver: str | None,
    artifact: Path,
    dest_root: Path,
    limits: dict[str, int],
    pinned: bool,
    source_file: str,
    artifact_url: str | None,
    upload_time: str | None,
    sha256: str | None,
) -> ResolvedPackage:
    extract_dir = dest_root / "python" / f"{_normalize_name(name)}-{ver or 'unknown'}"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    file_count, unpacked, _inv = _extract_wheel_or_sdist(artifact, extract_dir, limits)
    return ResolvedPackage(
        name=name,
        version=ver,
        status="extracted",
        pinned=pinned,
        artifact_url=artifact_url,
        sha256=sha256,
        upload_time=upload_time,
        extract_path=f"_dependencies/python/{extract_dir.name}",
        source_file=source_file,
        file_count=file_count,
        unpacked_bytes=unpacked,
    )


def _resolve_pypi_package(
    spec: RequirementSpec,
    dest_root: Path,
    cache_dir: Path,
    limits: dict[str, int],
) -> ResolvedPackage:
    assert spec.name
    name = spec.name
    version = None
    if spec.version_spec and spec.version_spec.startswith("=="):
        version = spec.version_spec[2:]

    local = _find_local_artifact(name, version)
    if local is not None:
        try:
            digest = hashlib.sha256(local.read_bytes()).hexdigest()
            # Infer version from filename name-version-...
            ver = version
            if not ver:
                m = re.match(
                    rf"{re.escape(_normalize_name(name).replace('-', '[-_]'))}-([0-9][^-\s]+)",
                    local.name,
                    flags=re.IGNORECASE,
                )
                ver = m.group(1) if m else "local"
            return _extract_resolved_artifact(
                name=name,
                ver=ver,
                artifact=local,
                dest_root=dest_root,
                limits=limits,
                pinned=spec.pinned,
                source_file=spec.source_file,
                artifact_url=f"file://{local}",
                upload_time=None,
                sha256=digest,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Local artifact extract failed for {name} from {local}: {exc}")

    try:
        if version:
            project = _http_json(
                f"https://pypi.org/pypi/{urllib.parse.quote(name)}/{urllib.parse.quote(version)}/json"
            )
            info = project.get("info") or {}
            ver = info.get("version") or version
            urls = project.get("urls") or []
            if not urls:
                raise RuntimeError("no urls in version response")
            pure = [
                f
                for f in urls
                if f.get("packagetype") == "bdist_wheel" and "py3-none-any" in f.get("filename", "")
            ]
            wheels = [f for f in urls if f.get("packagetype") == "bdist_wheel"]
            sdists = [f for f in urls if f.get("packagetype") == "sdist"]
            choice = (pure or wheels or sdists)[0]
        else:
            project = _http_json(f"https://pypi.org/pypi/{urllib.parse.quote(name)}/json")
            ver, choice = _select_pypi_artifact(project, None)
    except Exception as exc:  # noqa: BLE001
        return ResolvedPackage(
            name=name,
            version=version,
            status="unresolved",
            pinned=spec.pinned,
            source_file=spec.source_file,
            detail=f"pypi lookup failed: {exc}",
        )

    url = choice.get("url")
    if not url:
        return ResolvedPackage(
            name=name,
            version=ver,
            status="unresolved",
            pinned=spec.pinned,
            source_file=spec.source_file,
            detail="artifact missing url",
        )

    filename = choice.get("filename") or Path(urllib.parse.urlparse(url).path).name
    artifact = cache_dir / filename
    try:
        digest = _download(url, artifact)
        expected = (choice.get("digests") or {}).get("sha256")
        if expected and expected != digest:
            return ResolvedPackage(
                name=name,
                version=ver,
                status="unresolved",
                pinned=spec.pinned,
                artifact_url=url,
                sha256=digest,
                upload_time=choice.get("upload_time_iso_8601") or choice.get("upload_time"),
                source_file=spec.source_file,
                detail=f"sha256 mismatch (expected {expected})",
            )
        return _extract_resolved_artifact(
            name=name,
            ver=ver,
            artifact=artifact,
            dest_root=dest_root,
            limits=limits,
            pinned=spec.pinned,
            source_file=spec.source_file,
            artifact_url=url,
            upload_time=choice.get("upload_time_iso_8601") or choice.get("upload_time"),
            sha256=digest,
        )
    except Exception as exc:  # noqa: BLE001
        return ResolvedPackage(
            name=name,
            version=ver if "ver" in locals() else version,
            status="unresolved",
            pinned=spec.pinned,
            artifact_url=url,
            source_file=spec.source_file,
            detail=f"download/extract failed: {exc}",
        )


def materialize_dependencies(
    challenger_root: Path,
    review_root: Path,
    *,
    allowlist: frozenset[str] | None = None,
    limits: dict[str, int] | None = None,
) -> DependencyMaterialization:
    """Discover, download, unpack, and scan dependencies into ``review_root``."""
    allow = {_normalize_name(x) for x in (allowlist or DEFAULT_ALLOWLIST)}
    limits = {
        "max_member_count": 20000,
        "max_unpacked_bytes": 800_000_000,
        "max_data_file_bytes": 2_000_000,
        "max_data_bytes_total": 40_000_000,
        "max_data_files": 40,
        "max_single_extract_bytes": 8_000_000,
        **(limits or {}),
    }

    deps_root = review_root / "_dependencies"
    manifests = review_root / "_manifests"
    deps_root.mkdir(parents=True, exist_ok=True)
    manifests.mkdir(parents=True, exist_ok=True)

    specs = discover_requirements(challenger_root)
    result = DependencyMaterialization()
    cache_dir = Path(tempfile.mkdtemp(prefix="dep-cache-"))
    extracted_roots: list[tuple[str, Path]] = []

    try:
        for spec in specs:
            assert spec.name
            norm = _normalize_name(spec.name)
            if norm in allow:
                result.packages.append(
                    ResolvedPackage(
                        name=spec.name,
                        version=(spec.version_spec or "").lstrip("=<>!~"),
                        status="allowlisted",
                        pinned=spec.pinned,
                        artifact_url=spec.git_url,
                        source_file=spec.source_file,
                        detail="skipped unpack (allowlisted common trainer dependency)",
                    )
                )
                continue

            if spec.git_url:
                # Non-allowlisted git deps must be commit-pinned to be reviewable later;
                # until clone support lands, fail closed so Claude cannot mark them clean.
                detail = (
                    "git dependency is not commit-pinned"
                    if not spec.pinned
                    else "git dependency clone not yet materialized for review"
                )
                result.packages.append(
                    ResolvedPackage(
                        name=spec.name,
                        version=spec.version_spec,
                        status="unresolved",
                        pinned=spec.pinned,
                        artifact_url=spec.git_url,
                        source_file=spec.source_file,
                        detail=detail,
                    )
                )
                result.unresolved.append(f"{spec.name} ({detail})")
                continue

            resolved = _resolve_pypi_package(spec, deps_root, cache_dir, limits)
            result.packages.append(resolved)
            if resolved.status == "extracted" and resolved.extract_path:
                extract_dir = review_root / resolved.extract_path
                # extract_path is relative like _dependencies/python/name-ver
                if not extract_dir.exists():
                    # Fallback absolute under deps_root
                    extract_dir = deps_root / "python" / f"{_normalize_name(resolved.name)}-{resolved.version}"
                extracted_roots.append((resolved.name, extract_dir))
            elif resolved.status == "unresolved":
                result.unresolved.append(f"{resolved.name}: {resolved.detail}")
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)

    # Deterministic findings
    non_allowlisted = {
        _normalize_name(p.name) for p in result.packages if p.status != "allowlisted"
    }
    result.findings.extend(_scan_challenger_refs(challenger_root, non_allowlisted))
    for pkg_name, extract_dir in extracted_roots:
        if extract_dir.exists():
            result.findings.extend(_scan_extracted_package(extract_dir, pkg_name))

    # Write manifests for Claude
    (manifests / "resolved-python.json").write_text(
        json.dumps({"packages": [asdict(p) for p in result.packages]}, indent=2)
    )
    (manifests / "findings.json").write_text(
        json.dumps({"findings": [asdict(f) for f in result.findings]}, indent=2)
    )
    (manifests / "unresolved.json").write_text(
        json.dumps({"unresolved": result.unresolved}, indent=2)
    )
    logger.info(
        f"Dependency materialization: {len(result.packages)} packages, "
        f"{sum(1 for p in result.packages if p.status == 'extracted')} extracted, "
        f"{len(result.unresolved)} unresolved, {len(result.findings)} findings"
    )
    return result
