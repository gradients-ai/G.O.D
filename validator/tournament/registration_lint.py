"""Registration lint of a quant entrant's build: the immutable-dependency rule.

The image is built per job from the pinned commit, so anything the build pulls by a mutable reference can change
after registration and carry work done offline. The rule requires:

- hash-pinned pip: every `pip install` / `uv pip install` takes its packages from requirements files whose every
  entry carries `--hash=`, or from a local path in the build context, or from a URL with a `#sha256=` fragment.
  Inline specs (`pip install foo==1.2`) cannot carry a hash and fail. `uv sync` passes only with `--locked` or
  `--frozen` (uv.lock carries the hashes);
- SHA-pinned git: every `git+...` requirement ends `@<40-hex>`, and a `git clone` must be followed in the same RUN
  (or build script) by a checkout/reset to a 40-hex SHA;
- no fetch of miner-controlled URLs: `curl`, `wget` and `ADD <url>` fail unless the URL is under
  `QUANT_TRUSTED_FETCH_URL_PREFIXES`, the ADD carries `--checksum=sha256:`, or the same RUN (build script) runs
  `sha256sum -c`;
- base images (`FROM`, `COPY --from=<image>`) are digest-pinned or under `QUANT_TRUSTED_BASE_IMAGE_PREFIXES`,
  because a miner-owned image tag is a mutable dependency like any other.

Scope: the text Dockerfile the trainer builds (`TEXT_DOCKERFILE_PATHS`, first match) and the shell scripts its RUN
instructions execute (`bash x.sh`, `./x.sh`), resolved into the repo through the Dockerfile's COPY/ADD and WORKDIR.
A requirements file that cannot be resolved into the build context cannot be checked and fails. apt packages come
from distro mirrors, not miner-controlled, and are not checked. Python scripts run at build time and anything the
container fetches at runtime are out of reach of a static lint and stay with manual review.

This is a heuristic shell reader, not a shell: it splits commands on `&& || ; |`, substitutes ARG/ENV defaults and
looks at each command's first word, so `apt-get install wget` is not a wget.

Files over `QUANT_LARGE_FILE_REVIEW_BYTES` are listed in the report for manual review and never fail it.

With `QUANT_REGISTRATION_LINT_ENFORCE` False the lint is warn-only: a failing entrant is enrolled and its
violations are logged. When it is True, a failing entrant is excluded.
"""

import json
import posixpath
import re
import shlex
from enum import Enum
from pathlib import Path

from pydantic import BaseModel
from pydantic import Field

from core.build_context import LargeFile
from core.build_context import describe_large_files
from core.build_context import find_large_files
from core.constants.docker import TEXT_DOCKERFILE_PATHS
from core.constants.quant import QUANT_LARGE_FILE_REVIEW_BYTES
from core.logging import get_logger
from validator.tournament import constants as t_cst


logger = get_logger(__name__)

_SHA40 = re.compile(r"[0-9a-f]{40}")
_SHA256_FRAGMENT = re.compile(r"[#&]sha256=[0-9a-fA-F]{64}\b")
_URL = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_VARIABLE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)")
_HEREDOC = re.compile(r"<<-?\s*[\"']?([A-Za-z_][A-Za-z0-9_]*)[\"']?")
_GIT_SHA_PIN = re.compile(
    r"\bgit\s+(?:-C\s+\S+\s+)?(?:checkout|switch\s+--detach|reset\s+--hard)\s+(?:-\S+\s+)*[0-9a-f]{40}\b"
)
_SHA256_CHECK = re.compile(r"\bsha256sum\b[^\n;&|]*\s(?:-c|--check)\b")
_PIP_NAME = re.compile(r"^pip(\d+(\.\d+)?)?$")
_PYTHON_NAME = re.compile(r"^python(\d+(\.\d+)?)?$")
_SHELL_NAMES = {"sh", "bash", "dash", "zsh", "source", "."}
_PREFIX_WORDS = {
    "{", "}", "(", ")", "!", "then", "do", "else", "elif", "if", "while", "until", "time", "exec", "sudo", "env", "nohup",
}
_OPERATORS = {"&&", "||", ";", "|", "&", ";;", "(", ")", "|&"}
_LOCAL_ARCHIVE_SUFFIXES = (".whl", ".tar.gz", ".zip", ".tgz")
_REQUIREMENT_FLAGS = ("-r", "--requirement", "--requirements")
_CONSTRAINT_FLAGS = ("-c", "--constraint", "--constraints", "--override", "--overrides")
_EDITABLE_FLAGS = ("-e", "--editable")
# pip and `uv pip install` options that take a separate value; without this list the value reads as a package.
_VALUE_FLAGS = {
    "-i", "--index-url", "--extra-index-url", "--index", "--default-index", "-f", "--find-links", "-t", "--target",
    "--prefix", "--root", "--platform", "--python-version", "--implementation", "--abi", "--only-binary",
    "--no-binary", "--progress-bar", "--trusted-host", "--cache-dir", "--src", "--upgrade-strategy",
    "--global-option", "--config-settings", "--config-setting", "-C", "--log", "--timeout", "--retries", "--proxy",
    "--cert", "--client-cert", "--python", "-p", "--torch-backend", "--index-strategy", "--python-platform",
    "--resolution", "--prerelease", "--keyring-provider", "--exclude-newer", "--link-mode", "--upgrade-package",
    "-P", "--reinstall-package", "--no-build-isolation-package", "--group", "--extra", "--report", "--root-user-action",
    "--exists-action", "--use-feature", "--use-deprecated", "--hash",
}
_UV_SYNC_LOCK_FLAGS = {"--locked", "--frozen"}
_MAX_REASONS = 5


class LintRule(str, Enum):
    NO_DOCKERFILE = "no_dockerfile"
    UNHASHED_PIP = "unhashed_pip"
    UNPINNED_GIT = "unpinned_git"
    REMOTE_FETCH = "remote_fetch"
    UNPINNED_BASE_IMAGE = "unpinned_base_image"


class LintViolation(BaseModel):
    rule: LintRule
    file: str  # repo-relative
    line: int
    detail: str

    def describe(self) -> str:
        return f"{self.file}:{self.line} [{self.rule.value}] {self.detail}"


class RegistrationLintReport(BaseModel):
    dockerfile: str | None
    build_scripts: list[str] = Field(default_factory=list)
    violations: list[LintViolation] = Field(default_factory=list)
    large_files: list[LargeFile] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.violations

    def rules_failed(self) -> list[LintRule]:
        return sorted({violation.rule for violation in self.violations})

    def rejection_reason(self) -> str:
        shown = "; ".join(violation.describe() for violation in self.violations[:_MAX_REASONS])
        more = len(self.violations) - _MAX_REASONS
        return f"{len(self.violations)} build-lint violation(s): {shown}" + (f"; and {more} more" if more > 0 else "")


class RegistrationLintFailed(ValueError):
    """The entrant's build pulls a mutable dependency; registration excludes it while enforcement is on."""

    def __init__(self, report: RegistrationLintReport):
        super().__init__(report.rejection_reason())
        self.report = report


class _Instruction(BaseModel):
    keyword: str
    args: str
    line: int


class _CopyMapping(BaseModel):
    source: Path  # absolute path inside the repo
    dest: str  # absolute container path
    source_is_dir: bool


class _Context(BaseModel):
    """What a RUN line can see: the repo, the files COPY'd so far, WORKDIR and the variables in scope."""

    repo_dir: Path
    mappings: list[_CopyMapping] = Field(default_factory=list)
    workdir: str = "/"
    variables: dict[str, str] = Field(default_factory=dict)


class _Linter:
    def __init__(self, repo_dir: Path):
        self.repo_dir = repo_dir.resolve()
        self.violations: list[LintViolation] = []
        self.build_scripts: list[str] = []
        self._seen_requirements: set[Path] = set()
        self._seen_scripts: set[Path] = set()

    def rel(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.repo_dir).as_posix()
        except ValueError:
            return path.as_posix()

    def add(self, rule: LintRule, file: str, line: int, detail: str) -> None:
        self.violations.append(LintViolation(rule=rule, file=file, line=line, detail=detail))

    def in_repo(self, path: Path) -> Path | None:
        resolved = path.resolve()
        if resolved != self.repo_dir and self.repo_dir not in resolved.parents:
            return None
        return resolved if resolved.exists() else None

    # --- Dockerfile ------------------------------------------------------------------------------------------

    def lint_dockerfile(self, dockerfile: Path) -> None:
        file = self.rel(dockerfile)
        context = _Context(repo_dir=self.repo_dir)
        stages: set[str] = set()
        for instruction in _dockerfile_instructions(dockerfile.read_text(errors="replace")):
            keyword = instruction.keyword
            args = _substitute(instruction.args, context.variables)
            if keyword in ("ARG", "ENV"):
                context.variables.update(_declared_variables(keyword, args))
            elif keyword == "FROM":
                self._lint_from(args, stages, file, instruction.line)
                context = _Context(repo_dir=self.repo_dir, variables=context.variables)
            elif keyword == "WORKDIR":
                context.workdir = posixpath.normpath(posixpath.join(context.workdir, args.strip()))
            elif keyword in ("COPY", "ADD"):
                self._lint_copy(keyword, args, context, stages, file, instruction.line)
            elif keyword == "RUN":
                command = _exec_form(args) or args
                self.lint_shell(command, context, file, instruction.line, verified_text=command)

    def _lint_from(self, args: str, stages: set[str], file: str, line: int) -> None:
        words = [word for word in args.split() if not word.startswith("--")]
        if not words:
            return
        image = words[0]
        if len(words) >= 3 and words[1].lower() == "as":
            stages.add(words[2].lower())
        self._lint_image(image, stages, file, line)

    def _lint_image(self, image: str, stages: set[str], file: str, line: int) -> None:
        if image.lower() in stages or image == "scratch" or "@sha256:" in image:
            return
        if "/" not in image.split(":")[0] or image.startswith(t_cst.QUANT_TRUSTED_BASE_IMAGE_PREFIXES):
            return
        self.add(
            LintRule.UNPINNED_BASE_IMAGE, file, line,
            f"image {image} is neither digest-pinned (@sha256:) nor from a trusted publisher",
        )

    def _lint_copy(self, keyword: str, args: str, context: _Context, stages: set[str], file: str, line: int) -> None:
        words = _exec_form_words(args) or args.split()
        flags = [word for word in words if word.startswith("--")]
        paths = [word for word in words if not word.startswith("--")]
        from_flag = next((flag.split("=", 1)[1] for flag in flags if flag.startswith("--from=")), None)
        if from_flag is not None:
            if not from_flag.isdigit():
                self._lint_image(from_flag, stages, file, line)
            return
        if len(paths) < 2:
            return
        sources, dest = paths[:-1], paths[-1]
        if keyword == "ADD":
            has_checksum = any(flag.startswith("--checksum=sha256:") for flag in flags)
            for source in sources:
                if _URL.match(source) and not has_checksum and not _git_url_pinned(source):
                    self._lint_url(source, file, line, verified=False, verb="ADD")
            sources = [source for source in sources if not _URL.match(source)]
        dest = posixpath.normpath(posixpath.join(context.workdir, dest)) + ("/" if dest.endswith("/") else "")
        expanded = [match for source in sources for match in self._expand(source)]
        into_dir = dest.endswith("/") or len(expanded) > 1
        for source in expanded:
            target = dest.rstrip("/") or "/"
            if source.is_dir():
                context.mappings.append(_CopyMapping(source=source, dest=target, source_is_dir=True))
            else:
                # `COPY a.txt /app` lands at /app/a.txt when /app is already a directory, which only the image
                # knows; map both so either spelling resolves.
                file_dests = [posixpath.join(target, source.name)] + ([] if into_dir else [target])
                for file_dest in file_dests:
                    context.mappings.append(_CopyMapping(source=source, dest=file_dest, source_is_dir=False))

    def _expand(self, source: str) -> list[Path]:
        pattern = source.lstrip("/")
        if any(char in pattern for char in "*?["):
            return sorted(path for path in self.repo_dir.glob(pattern) if self.in_repo(path))
        found = self.in_repo(self.repo_dir / pattern) if pattern not in ("", ".") else self.repo_dir
        return [found] if found else []

    # --- shell -----------------------------------------------------------------------------------------------

    def lint_shell(self, text: str, context: _Context, file: str, line: int, verified_text: str) -> None:
        """Lint shell text; verified_text is the unit (one RUN, one script) a checksum or SHA checkout may sit in."""
        verified = bool(_SHA256_CHECK.search(verified_text))
        git_pinned = bool(_GIT_SHA_PIN.search(verified_text))
        for words in _simple_commands(text):
            self._lint_command(words, context, file, line, verified, git_pinned)

    def _lint_command(
        self, words: list[str], context: _Context, file: str, line: int, verified: bool, git_pinned: bool
    ) -> None:
        words = _strip_prefix(words)
        if not words:
            return
        name = posixpath.basename(words[0])
        args = words[1:]
        if _PIP_NAME.match(name):
            self._lint_pip(args, context, file, line)
        elif _PYTHON_NAME.match(name) and len(args) >= 2 and args[0] == "-m":
            if args[1] == "pip":
                self._lint_pip(args[2:], context, file, line)
            elif args[1] == "uv":
                self._lint_uv(args[2:], context, file, line)
        elif name == "uv":
            self._lint_uv(args, context, file, line)
        elif name in ("curl", "wget"):
            for url in (arg for arg in args if _URL.match(arg)):
                self._lint_url(url, file, line, verified=verified, verb=name)
        elif name == "git" and "clone" in args:
            urls = [arg for arg in args if _URL.match(arg) or arg.startswith("git@")]
            if urls and not git_pinned:
                self.add(
                    LintRule.UNPINNED_GIT, file, line,
                    f"git clone {urls[0]} is not checked out at a 40-hex commit SHA in the same step",
                )
        script = _script_target(name, words[0], args)
        if script:
            self._lint_script(script, context, file, line)

    def _lint_url(self, url: str, file: str, line: int, verified: bool, verb: str) -> None:
        if verified or url.startswith(t_cst.QUANT_TRUSTED_FETCH_URL_PREFIXES):
            return
        self.add(
            LintRule.REMOTE_FETCH, file, line,
            f"{verb} {url}: remote fetch without a checksum from a host that is not on the trusted list",
        )

    def _lint_pip(self, args: list[str], context: _Context, file: str, line: int) -> None:
        if not args or args[0] != "install":
            return
        self._lint_install_args(args[1:], context, file, line)

    def _lint_uv(self, args: list[str], context: _Context, file: str, line: int) -> None:
        if not args:
            return
        if args[0] == "pip" and len(args) >= 2 and args[1] == "install":
            self._lint_install_args(args[2:], context, file, line)
        elif args[0] == "pip" and len(args) >= 2 and args[1] == "sync":
            positional = _positional(args[2:])
            self._lint_install_args([word for path in positional for word in ("-r", path)], context, file, line)
        elif args[0] == "sync" and not (_UV_SYNC_LOCK_FLAGS & set(args)):
            self.add(LintRule.UNHASHED_PIP, file, line, "uv sync without --locked or --frozen can re-resolve uv.lock")
        elif args[0] in ("add", "tool") or (args[0] == "run" and "--with" in args):
            self.add(LintRule.UNHASHED_PIP, file, line, f"uv {args[0]} installs unhashed packages")

    def _lint_install_args(self, args: list[str], context: _Context, file: str, line: int) -> None:
        inline: list[str] = []
        index = 0
        while index < len(args):
            word = args[index]
            flag, has_value, value = word.partition("=")
            if word in _REQUIREMENT_FLAGS:
                if index + 1 < len(args):
                    self._lint_requirements_ref(args[index + 1], context, file, line)
                index += 2
            elif _is_requirement_flag(word):
                path = value if word.startswith("--") else word.removeprefix("-r")
                self._lint_requirements_ref(path, context, file, line)
                index += 1
            elif word in _EDITABLE_FLAGS:
                if index + 1 < len(args):
                    self._lint_spec(args[index + 1], file, line, inline)
                index += 2
            elif word.startswith("-"):
                takes_value = (word in _VALUE_FLAGS or word in _CONSTRAINT_FLAGS) and not has_value
                index += 2 if takes_value else 1
            else:
                self._lint_spec(word, file, line, inline)
                index += 1
        if inline:
            self.add(
                LintRule.UNHASHED_PIP, file, line,
                f"inline package spec(s) cannot carry a hash: {', '.join(inline[:_MAX_REASONS])}"
                + (f" and {len(inline) - _MAX_REASONS} more" if len(inline) > _MAX_REASONS else ""),
            )

    def _lint_spec(self, spec: str, file: str, line: int, inline: list[str]) -> None:
        """One requirement: a git ref must be a SHA, a remote file needs #sha256=, a local path passes."""
        if "git+" in spec:
            if not _git_requirement_pinned(spec):
                self.add(LintRule.UNPINNED_GIT, file, line, f"{spec} is not pinned to a 40-hex commit SHA")
        elif _URL.match(spec) and not spec.startswith("file:"):
            if not _SHA256_FRAGMENT.search(spec):
                inline.append(spec)
        elif not _is_local_path(spec):
            inline.append(spec)

    def _lint_requirements_ref(self, path: str, context: _Context, file: str, line: int) -> None:
        resolved = self._resolve(path, context)
        if resolved is None:
            self.add(
                LintRule.UNHASHED_PIP, file, line,
                f"requirements file {path} is not in the build context, so its hashes cannot be checked",
            )
            return
        self._lint_requirements_file(resolved)

    def _lint_requirements_file(self, path: Path) -> None:
        if path in self._seen_requirements:
            return
        self._seen_requirements.add(path)
        file = self.rel(path)
        unhashed: list[str] = []
        first_line = 0
        for number, entry in _requirement_entries(path.read_text(errors="replace")):
            words = entry.split()
            head = words[0]
            if _is_requirement_flag(head):
                if head in _REQUIREMENT_FLAGS:
                    nested = words[1] if len(words) > 1 else ""
                else:
                    nested = head.split("=", 1)[1] if head.startswith("--") else head.removeprefix("-r")
                target = self.in_repo(path.parent / nested)
                if target is None:
                    self.add(LintRule.UNHASHED_PIP, file, number, f"nested requirements file {nested} is not in the repo")
                else:
                    self._lint_requirements_file(target)
                continue
            if head in _EDITABLE_FLAGS:
                spec = words[1] if len(words) > 1 else ""
                if "git+" in spec and not _git_requirement_pinned(spec):
                    self.add(LintRule.UNPINNED_GIT, file, number, f"{spec} is not pinned to a 40-hex commit SHA")
                continue
            if head.startswith("-"):
                continue  # global options: index urls, --require-hashes, constraints
            if "git+" in entry:
                if not _git_requirement_pinned(entry.split(" --hash")[0].strip()):
                    self.add(LintRule.UNPINNED_GIT, file, number, f"{head} is not pinned to a 40-hex commit SHA")
                continue
            if _is_local_path(head) or "--hash" in entry or _SHA256_FRAGMENT.search(entry):
                continue
            unhashed.append(re.split(r"[<>=!~;\[ @]", head, maxsplit=1)[0] or head)
            first_line = first_line or number
        if unhashed:
            self.add(
                LintRule.UNHASHED_PIP, file, first_line,
                f"{len(unhashed)} requirement(s) without --hash (e.g. {', '.join(unhashed[:_MAX_REASONS])})",
            )

    def _lint_script(self, path: str, context: _Context, file: str, line: int) -> None:
        resolved = self._resolve(path, context)
        if resolved is None or resolved in self._seen_scripts or not resolved.is_file():
            return
        self._seen_scripts.add(resolved)
        script_file = self.rel(resolved)
        self.build_scripts.append(script_file)
        text = resolved.read_text(errors="replace")
        for number, logical in _logical_lines(text):
            self.lint_shell(logical, context, script_file, number, verified_text=text)

    def _resolve(self, path: str, context: _Context) -> Path | None:
        """Map a container path (absolute, or relative to WORKDIR) back into the repo via the COPYs so far."""
        container_path = posixpath.normpath(posixpath.join(context.workdir, path))
        for mapping in reversed(context.mappings):
            if mapping.source_is_dir:
                if container_path == mapping.dest or container_path.startswith(mapping.dest.rstrip("/") + "/"):
                    relative = container_path[len(mapping.dest):].lstrip("/")
                    found = self.in_repo(mapping.source / relative)
                    if found:
                        return found
            elif container_path == mapping.dest:
                return mapping.source
        return None


def lint_repo(repo_dir: Path) -> RegistrationLintReport:
    """Lint the text Dockerfile at repo_dir and the build scripts it runs; list files over the review size."""
    large_files = find_large_files(repo_dir, QUANT_LARGE_FILE_REVIEW_BYTES)
    dockerfile = next((repo_dir / path for path in TEXT_DOCKERFILE_PATHS if (repo_dir / path).is_file()), None)
    if dockerfile is None:
        violation = LintViolation(
            rule=LintRule.NO_DOCKERFILE, file=TEXT_DOCKERFILE_PATHS[0], line=0,
            detail=f"no text Dockerfile at any of {', '.join(TEXT_DOCKERFILE_PATHS)}",
        )
        return RegistrationLintReport(dockerfile=None, violations=[violation], large_files=large_files)
    linter = _Linter(repo_dir)
    linter.lint_dockerfile(dockerfile)
    return RegistrationLintReport(
        dockerfile=linter.rel(dockerfile),
        build_scripts=linter.build_scripts,
        violations=linter.violations,
        large_files=large_files,
    )


def check_registration_lint(repo_dir: Path) -> RegistrationLintReport:
    """Lint a registration clone and log the outcome; raise RegistrationLintFailed when enforcing and it fails."""
    report = lint_repo(repo_dir)
    if report.large_files:
        logger.warning(
            f"Build context has {len(report.large_files)} file(s) over {QUANT_LARGE_FILE_REVIEW_BYTES} bytes, "
            f"flagged for manual review (weights-derived payloads are banned): {describe_large_files(report.large_files)}"
        )
    if report.passed:
        logger.info(f"Build lint passed for {report.dockerfile} and {len(report.build_scripts)} build script(s)")
        return report
    if t_cst.QUANT_REGISTRATION_LINT_ENFORCE:
        raise RegistrationLintFailed(report)
    logger.warning(f"Build lint (warn-only, entrant kept): {report.rejection_reason()}")
    return report


# --- parsing helpers -----------------------------------------------------------------------------------------


def _dockerfile_instructions(text: str) -> list[_Instruction]:
    """Join continuation lines, drop comments, and fold a RUN heredoc body into its instruction."""
    instructions: list[_Instruction] = []
    lines = text.splitlines()
    buffer: list[str] = []
    start = 0
    index = 0
    while index < len(lines):
        number, stripped = index + 1, lines[index].strip()
        index += 1
        if not stripped or stripped.startswith("#"):
            continue
        if not buffer:
            start = number
        if stripped.endswith("\\"):
            buffer.append(stripped[:-1])
            continue
        buffer.append(stripped)
        joined = " ".join(buffer)
        buffer = []
        keyword, _, args = joined.partition(" ")
        heredoc = _HEREDOC.search(args)
        if heredoc and keyword.upper() == "RUN":
            body: list[str] = []
            while index < len(lines) and lines[index].strip() != heredoc.group(1):
                body.append(lines[index])
                index += 1
            index += 1
            args = _HEREDOC.sub("", args) + "\n" + "\n".join(body)
        instructions.append(_Instruction(keyword=keyword.upper(), args=args.strip(), line=start))
    if buffer:
        keyword, _, args = " ".join(buffer).partition(" ")
        instructions.append(_Instruction(keyword=keyword.upper(), args=args.strip(), line=start))
    return instructions


def _exec_form_words(args: str) -> list[str] | None:
    if not args.lstrip().startswith("["):
        return None
    try:
        parsed = json.loads(args)
    except ValueError:
        return None
    return [str(word) for word in parsed] if isinstance(parsed, list) else None


def _exec_form(args: str) -> str | None:
    words = _exec_form_words(args)
    return shlex.join(words) if words else None


def _declared_variables(keyword: str, args: str) -> dict[str, str]:
    declared: dict[str, str] = {}
    words = _split_words(args)
    if keyword == "ENV" and len(words) >= 2 and "=" not in words[0]:
        return {words[0]: " ".join(words[1:])}  # legacy `ENV KEY value`
    for word in words:
        name, has_value, value = word.partition("=")
        if has_value:
            declared[name] = value
    return declared


def _substitute(text: str, variables: dict[str, str]) -> str:
    def replace(match: re.Match) -> str:
        name = match.group(1) or match.group(2)
        return variables.get(name, match.group(0))

    return _VARIABLE.sub(replace, text)


def _split_words(text: str) -> list[str]:
    try:
        return shlex.split(text)
    except ValueError:
        return text.split()


def _logical_lines(text: str) -> list[tuple[int, str]]:
    """Shell lines with backslash continuations joined, numbered by their first physical line."""
    logical: list[tuple[int, str]] = []
    buffer: list[str] = []
    start = 0
    for number, raw in enumerate(text.splitlines(), 1):
        if not buffer:
            start = number
        if raw.rstrip().endswith("\\"):
            buffer.append(raw.rstrip()[:-1])
            continue
        buffer.append(raw)
        logical.append((start, " ".join(buffer)))
        buffer = []
    if buffer:
        logical.append((start, " ".join(buffer)))
    return logical


def _tokens(line: str) -> list[str]:
    lexer = shlex.shlex(line, posix=True, punctuation_chars=";&|()")
    lexer.whitespace_split = True
    lexer.commenters = ""
    try:
        return list(lexer)
    except ValueError:
        return line.split()


def _simple_commands(text: str) -> list[list[str]]:
    commands: list[list[str]] = []
    for _, line in _logical_lines(text):
        current: list[str] = []
        for token in _tokens(line):
            if token.startswith("#") and not current:
                break
            if token in _OPERATORS:
                if current:
                    commands.append(current)
                current = []
            else:
                current.append(token)
        if current:
            commands.append(current)
    return commands


def _strip_prefix(words: list[str]) -> list[str]:
    index = 0
    while index < len(words) and (words[index] in _PREFIX_WORDS or _ASSIGNMENT.match(words[index])):
        index += 1
    return words[index:]


def _script_target(name: str, argv0: str, args: list[str]) -> str | None:
    """The .sh file a command executes: `./x.sh`, `/path/x.sh`, or `bash [-flags] x.sh`."""
    if argv0.endswith(".sh"):
        return argv0
    if name in _SHELL_NAMES:
        first = next((arg for arg in args if not arg.startswith("-")), None)
        if first and first.endswith(".sh"):
            return first
    return None


def _positional(args: list[str]) -> list[str]:
    positional: list[str] = []
    index = 0
    while index < len(args):
        word = args[index]
        if word.startswith("-"):
            index += 2 if word in _VALUE_FLAGS and "=" not in word else 1
            continue
        positional.append(word)
        index += 1
    return positional


def _requirement_entries(text: str) -> list[tuple[int, str]]:
    entries: list[tuple[int, str]] = []
    for number, line in _logical_lines(text):
        stripped = re.sub(r"(^|\s)#.*$", "", line).strip()
        if stripped:
            entries.append((number, stripped))
    return entries


def _is_requirement_flag(word: str) -> bool:
    if word in _REQUIREMENT_FLAGS:
        return True
    if word.startswith("--"):
        return word.split("=", 1)[0] in _REQUIREMENT_FLAGS and "=" in word
    return word.startswith("-r") and len(word) > 2


def _git_requirement_pinned(spec: str) -> bool:
    url = spec[spec.index("git+"):].split("#", 1)[0].strip().strip("\"'")
    _, at, ref = url.rpartition("@")
    return bool(at) and bool(_SHA40.fullmatch(ref))


def _git_url_pinned(url: str) -> bool:
    _, hashmark, ref = url.partition("#")
    return bool(hashmark) and bool(_SHA40.fullmatch(ref.split(":", 1)[0]))


def _is_local_path(spec: str) -> bool:
    spec = spec.strip("\"'")
    if _URL.match(spec):
        return spec.startswith("file:")
    return spec.startswith((".", "/", "~", "$")) or spec.endswith(_LOCAL_ARCHIVE_SUFFIXES)
