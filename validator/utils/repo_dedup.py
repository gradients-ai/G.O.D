"""Functional de-duplication of tournament submissions (anti-spam).

Spammers submit the same training repo many times under different identities to beat
the randomness of the bracket. This module detects functionally-equivalent submissions:

- T0 (exact commit):      identical resolved HEAD commit -> definite copy.
- T1 (normalized content): identical source after stripping whitespace/comments/ordering.
- T2 (Claude judgement):   pairwise functional-equivalence verdict via the Anthropic API.

T0/T1 are deterministic and run pre-training in R1 (auto-eliminate). T2 runs at the R1->R2
transition behind a human-review gate. Boss (open-source baseline) is always protected.
"""

import asyncio
import hashlib
import itertools
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from core.models.tournament_models import DedupCluster
from core.models.tournament_models import DedupResult
from core.models.tournament_models import DedupTier
from core.models.tournament_models import DupRelationship
from core.models.tournament_models import PairVerdict
from core.models.tournament_models import PreparedRepo
from core.models.tournament_models import RepoRef
from core.utils import sanitize_git_text
from validator.core import constants as cst
from validator.utils.logging import get_logger
from validator.utils.repo_diff_report import _clone_repo
from validator.utils.repo_diff_report import _collect_files


logger = get_logger(__name__)

# Redacted from any model-authored text we publish (defense in depth; the source clone-token in
# .git is already removed before the agent runs).
_GH_TOKEN_RE = re.compile(r"\b(?:gh[posru]_[A-Za-z0-9]{16,}|github_pat_[A-Za-z0-9_]{20,})\b")


def _sanitize_reason(text: str, tokens: set[str]) -> str:
    """Strip any credential that could ride along in the model's published free-text reason."""
    if not text:
        return text
    cleaned = sanitize_git_text(text, *tokens)
    return _GH_TOKEN_RE.sub("***", cleaned)

CONFIG_PATH = Path(__file__).with_name("repo_dedup_config.json")


@lru_cache(maxsize=1)
def _load_config() -> dict[str, Any]:
    with CONFIG_PATH.open() as handle:
        return json.load(handle)


# --------------------------------------------------------------------------- #
# Normalization + diffing
# --------------------------------------------------------------------------- #
def _read_text(path: Path) -> str:
    try:
        return path.read_text(errors="replace")
    except OSError:
        return ""


def _normalize_text(content: str) -> str:
    """Strip whitespace, blank lines and whole-line comments; collapse internal runs.

    Catches reformatting / comment / ordering-of-blank-lines disguises cheaply. Subtler
    rewrites are left to T2.
    """
    lines: list[str] = []
    for raw in content.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        lines.append(" ".join(stripped.split()))
    return "\n".join(lines)


def _normalized_digest(root: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    for rel in sorted(_collect_files(root)):
        normalized = _normalize_text(_read_text(root / rel))
        total += len(normalized)
        digest.update(rel.encode())
        digest.update(b"\0")
        digest.update(normalized.encode())
        digest.update(b"\0")
    return digest.hexdigest(), total


def _file_summary(repo_a: Path, repo_b: Path) -> str:
    """Deterministic file-level summary (names only) to focus the agent's reading.

    Files are compared by exact content; identical files are collapsed, the rest are listed
    as differing / only-in-one. The agent reads the actual contents itself."""
    files_a = _collect_files(repo_a)
    files_b = _collect_files(repo_b)

    identical: list[str] = []
    differing: list[str] = []
    only_a: list[str] = []
    only_b: list[str] = []
    for rel in sorted(files_a | files_b):
        in_a, in_b = rel in files_a, rel in files_b
        if in_a and in_b:
            (identical if _read_text(repo_a / rel) == _read_text(repo_b / rel) else differing).append(rel)
        elif in_a:
            only_a.append(rel)
        else:
            only_b.append(rel)

    def _section(title: str, items: list[str]) -> str:
        if not items:
            return f"{title}: none"
        return f"{title} ({len(items)}):\n" + "\n".join(f"  - {r}" for r in items)

    return "\n".join(
        [
            _section("Identical in both (shared baseline)", identical),
            _section("Present in both but DIFFERENT", differing),
            _section("Only in A", only_a),
            _section("Only in B", only_b),
        ]
    )


# --------------------------------------------------------------------------- #
# Clustering
# --------------------------------------------------------------------------- #
def _cluster(hotkeys: list[str], dup_pairs: list[tuple[str, str]]) -> list[list[str]]:
    parent = {h: h for h in hotkeys}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in dup_pairs:
        if a in parent and b in parent:
            parent[find(a)] = find(b)

    groups: dict[str, list[str]] = {}
    for h in hotkeys:
        groups.setdefault(find(h), []).append(h)
    return sorted((sorted(v) for v in groups.values() if len(v) > 1), key=lambda g: g[0])


def _hash_group_representatives(hotkeys: list[str], hash_pairs: list[tuple[str, str]], boss_hotkey: str | None) -> list[str]:
    """One representative per hash-duplicate group (boss preferred, else lexically first)."""
    parent = {h: h for h in hotkeys}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in hash_pairs:
        parent[find(a)] = find(b)

    groups: dict[str, list[str]] = {}
    for h in hotkeys:
        groups.setdefault(find(h), []).append(h)

    reps = [boss_hotkey if boss_hotkey and boss_hotkey in members else sorted(members)[0] for members in groups.values()]
    return sorted(reps)


def _flag_from_clusters(clusters: list[DedupCluster], boss_hotkey: str | None) -> set[str]:
    """Boss-protected flagging: drop every member of a cluster, except keep the boss."""
    flagged: set[str] = set()
    for cluster in clusters:
        if boss_hotkey and boss_hotkey in cluster.members:
            flagged.update(m for m in cluster.members if m != boss_hotkey)
        else:
            flagged.update(cluster.members)
    return flagged


def _hash_dup_pairs(prepared: dict[str, PreparedRepo]) -> list[PairVerdict]:
    """Deterministic T0/T1 duplicate verdicts (identical commit / identical normalized source)."""
    ok = sorted((p for p in prepared.values() if p.clone_ok), key=lambda p: p.hotkey)
    verdicts: list[PairVerdict] = []
    for a, b in itertools.combinations(ok, 2):
        if a.head_commit and a.head_commit == b.head_commit:
            tier, reason = DedupTier.T0, "Identical commit hash."
        elif a.normalized_digest and a.normalized_digest == b.normalized_digest:
            tier, reason = DedupTier.T1, "Identical normalized content."
        else:
            continue
        verdicts.append(
            PairVerdict(
                hotkey_a=a.hotkey,
                hotkey_b=b.hotkey,
                tier=tier,
                relationship=DupRelationship.DUPLICATE,
                confidence=1.0,
                reason=reason,
            )
        )
    return verdicts


def _cluster_basis(members: list[str], prepared: dict[str, PreparedRepo]) -> tuple[DedupTier, str]:
    if len({prepared[m].head_commit for m in members}) == 1:
        return DedupTier.T0, "Identical commit hash."
    if len({prepared[m].normalized_digest for m in members}) == 1:
        return DedupTier.T1, "Identical source after stripping whitespace/comments/ordering."
    return DedupTier.T2, "Judged functionally equivalent by Claude (cosmetic/evasive deltas only)."


# --------------------------------------------------------------------------- #
# Cloning / preparation
# --------------------------------------------------------------------------- #
async def _prepare_repos(repos: list[RepoRef], temp_root: Path) -> dict[str, PreparedRepo]:
    prepared: dict[str, PreparedRepo] = {}
    for ref in repos:
        dest = temp_root / ref.hotkey
        try:
            head = await asyncio.to_thread(_clone_repo, ref.repo_url, dest, ref.commit_hash, ref.github_token)
            digest, total = await asyncio.to_thread(_normalized_digest, dest)
            # Drop .git before the T2 agent can read it: .git/config holds the source token and
            # history holds author identity, and the agent's reason is published. head is already captured.
            await asyncio.to_thread(shutil.rmtree, dest / ".git", ignore_errors=True)
            prepared[ref.hotkey] = PreparedRepo(
                hotkey=ref.hotkey,
                repo_url=ref.repo_url,
                head_commit=head,
                normalized_digest=digest,
                content_chars=total,
                path=str(dest),
                clone_ok=True,
            )
        except Exception as exc:  # noqa: BLE001 - infra failure must not punish the miner
            logger.warning(f"dedup: could not prepare repo for {ref.hotkey}: {exc}")
            prepared[ref.hotkey] = PreparedRepo(hotkey=ref.hotkey, repo_url=ref.repo_url, clone_ok=False)
    return prepared


# --------------------------------------------------------------------------- #
# Runtime-contract source for the agent (verify claimed differentiators)
# --------------------------------------------------------------------------- #
def _god_repo_root() -> Path:
    """Repo root of the validator's running source (validator/utils/repo_dedup.py -> root)."""
    return Path(__file__).resolve().parents[2]


def _snapshot_god_source(dest: Path) -> bool:
    """Check out the validator's exact running source (git HEAD, TRACKED files only — local
    .vali.env/.trainer.env and other secrets are untracked and excluded) so the agent can verify
    whether a repo's claimed differentiators are real training-time inputs. False on any failure."""
    root = _god_repo_root()
    tar_path = dest.with_suffix(".tar")
    try:
        dest.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "-C", str(root), "archive", "--format=tar", "-o", str(tar_path), "HEAD"],
            check=True,
            capture_output=True,
            timeout=120,
        )
        with tarfile.open(tar_path) as tar:
            # Keep only regular files and directories. Dropping symlinks/hardlinks/devices means the
            # snapshot can never contain an absolute-path symlink (the repo tracks build-artifact
            # links like core/outputs/axolotl-artifacts) that would both abort the strict filter and
            # let the read-only agent escape the snapshot back onto the validator filesystem.
            members = [m for m in tar.getmembers() if m.isreg() or m.isdir()]
            try:
                tar.extractall(dest, members=members, filter="data")  # py>=3.12
            except TypeError:
                tar.extractall(dest, members=members)
        return True
    except Exception as exc:  # noqa: BLE001 - verification is best-effort; never break the gate
        logger.warning(f"dedup: could not snapshot G.O.D source for agent verification: {exc}")
        return False
    finally:
        try:
            tar_path.unlink(missing_ok=True)
        except OSError:
            pass


def _build_runtime_context(has_source: bool) -> str:
    """Instruction block telling the agent how to verify differentiators against the real contract."""
    if not has_source:
        return (
            "(The validator source snapshot is unavailable this run. Still apply the runtime-contract "
            "rules from your instructions: treat any divergence that only triggers under conditions which "
            "do NOT hold during tournament training — unset env vars, absent baseline-stats fields, "
            "reward-function inputs that are not provided, hostnames, missing files, dates, untouched CLI "
            "flags — as dead code, and the submissions as duplicate.)"
        )
    return (
        "AUTHORITATIVE SOURCE FOR VERIFICATION. The validator's exact running code (git HEAD, no secrets) "
        "is checked out read-only at ./_god_source. Use Read/Glob/Grep on it to establish what is ACTUALLY "
        "provided to training code, then verify every input the two repos rely on to differentiate themselves:\n"
        "- Env vars set at training time and the CLI args passed to the training entrypoint: see "
        "./_god_source/trainer/image_manager.py (run_trainer_container_text / run_trainer_container_image / "
        "build_wandb_env). Every other environment variable is UNSET at training time.\n"
        "- Baseline stats: read the BaselineStats model and where it is populated (Grep 'BaselineStats' and "
        "'baseline_stats' under ./_god_source). Baseline-stats VALUES differ per task, so do not assume a "
        "specific value — confirm a field a repo branches on is a real, actually-computed member of the model, "
        "not an invented or never-populated key.\n"
        "- Reward functions and their inputs (GRPO): defined in code and delivered via the --reward-functions "
        "CLI arg. Read the templates and implementations (Grep 'reward_templates', 'RewardFunction', "
        "'reward_functions' under ./_god_source; e.g. core/reward_templates.py, "
        "validator/utils/affine_reward_functions.py, core/models/utility_models.py, and the --reward-functions "
        "handling in trainer/image_manager.py). Reward functions receive `completions`, `extra_data` and "
        "**kwargs — confirm any field a repo's reward logic branches on is actually populated, not assumed.\n"
        "- Environment-task inputs/config: defined in code under ./_god_source/core.\n"
        "- Third-party library params/tools (e.g. axolotl, transformers, trl, peft): a repo may fake a "
        "difference by passing a config key or calling a 'tool'/feature that does not exist in the pinned "
        "library version, or that the library silently ignores (a no-op). Ground the version from each repo's "
        "pinned dependencies (requirements*.txt / pyproject / Dockerfile in each of the two submission "
        "directories), and "
        "when in doubt CHECK THE LIBRARY'S ACTUAL CODE where it is available to confirm the parameter exists "
        "AND changes training behaviour. A non-existent or silently-ignored param is a fake differentiator.\n"
        "For ANY input Repository A or B uses to differentiate behaviour, confirm against ./_god_source that it "
        "is genuinely delivered/populated at training time. If you CANNOT confirm it is a real training-time "
        "input, treat that branch as dead and the submissions as duplicate, and name the unverified input in "
        "your reason."
    )


# --------------------------------------------------------------------------- #
# Claude pairwise judgement (T2)
# --------------------------------------------------------------------------- #
def _import_claude_sdk():
    try:
        from claude_agent_sdk import ClaudeAgentOptions
        from claude_agent_sdk import ResultMessage
        from claude_agent_sdk import query
    except ImportError as exc:
        raise RuntimeError("claude-agent-sdk is required for pairwise dedup") from exc
    return ClaudeAgentOptions, ResultMessage, query


def _parse_verdict(text: str) -> tuple[DupRelationship, float, str]:
    """Extract the strict JSON verdict object from the model's reply."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"no JSON object in verdict reply: {text[:200]!r}")
    obj = json.loads(text[start : end + 1])
    raw = str(obj["relationship"]).strip()
    try:
        relationship = DupRelationship(raw)
    except ValueError as exc:
        raise ValueError(f"invalid relationship {raw!r}") from exc
    return relationship, float(obj.get("confidence", 0.0)), str(obj.get("reason", ""))


async def _judge_pair(cwd: Path, dir_a: str, dir_b: str, file_summary: str, runtime_context: str = "") -> PairVerdict:
    """Read-only agentic judgement over both cloned repos checked out under ``cwd``.

    The miners' private repo URLs are deliberately NOT passed to the model — it sees only
    the neutral on-disk dirs and is instructed to reason in terms of "Repository A/B" — so
    the published reasoning never leaks the original private repo/org names.

    ``runtime_context`` points the agent at ./_god_source (the validator's running code) so it can
    verify whether each repo's claimed differentiators are real training-time inputs."""
    ClaudeAgentOptions, ResultMessage, query = _import_claude_sdk()
    config = _load_config()
    prompt = config["user_prompt_template"].format(
        dir_a=dir_a, dir_b=dir_b, file_summary=file_summary, runtime_context=runtime_context
    )
    options = ClaudeAgentOptions(
        cwd=str(cwd),
        model=cst.TOURN_DEDUP_CLAUDE_MODEL,
        max_turns=cst.TOURN_DEDUP_CLAUDE_MAX_TURNS,
        max_budget_usd=cst.TOURN_DEDUP_CLAUDE_MAX_BUDGET_USD,
        permission_mode="dontAsk",
        allowed_tools=["Read", "Glob", "Grep"],
        disallowed_tools=["Write", "Edit", "Bash"],
        setting_sources=[],
        system_prompt=config["system_prompt"],
    )

    last_error: Exception | None = None
    for attempt in range(2):
        result_text = ""
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, ResultMessage):
                result_text = message.result or ""
        try:
            relationship, confidence, reason = _parse_verdict(result_text)
            return PairVerdict(
                hotkey_a=dir_a,
                hotkey_b=dir_b,
                tier=DedupTier.T2,
                relationship=relationship,
                confidence=confidence,
                reason=reason,
            )
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            last_error = exc
            logger.warning(f"dedup: unparseable verdict (attempt {attempt + 1}): {exc}")
    raise RuntimeError(f"Claude returned no parseable verdict for {dir_a} vs {dir_b}: {last_error}")


# --------------------------------------------------------------------------- #
# Public entry points
# --------------------------------------------------------------------------- #
async def find_hash_duplicates(repos: list[RepoRef], boss_hotkey: str | None = None) -> DedupResult:
    """Deterministic T0+T1 hash de-dup (no Claude). Used pre-training in R1."""
    temp_root = Path(tempfile.mkdtemp(prefix="dedup-hash-"))
    try:
        prepared = await _prepare_repos(repos, temp_root)
        ok_hotkeys = [p.hotkey for p in prepared.values() if p.clone_ok]
        verdicts = _hash_dup_pairs(prepared)
        pairs = [(v.hotkey_a, v.hotkey_b) for v in verdicts]

        clusters = []
        for members in _cluster(ok_hotkeys, pairs):
            basis, reason = _cluster_basis(members, prepared)
            clusters.append(DedupCluster(members=members, basis=basis, reason=reason))

        flagged = _flag_from_clusters(clusters, boss_hotkey)
        return DedupResult(
            cohort=[r.hotkey for r in repos],
            clusters=clusters,
            pair_verdicts=verdicts,
            flagged_hotkeys=sorted(flagged),
            unclonable_hotkeys=sorted(h for h, p in prepared.items() if not p.clone_ok),
        )
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


async def run_pairwise_dedup(repos: list[RepoRef], boss_hotkey: str | None = None) -> DedupResult:
    """Full T0+T1+T2 de-dup with Claude pairwise judgement. Used at the R1->R2 gate."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is not set; cannot run pairwise dedup")

    temp_root = Path(tempfile.mkdtemp(prefix="dedup-pair-"))
    try:
        prepared = await _prepare_repos(repos, temp_root)
        ok_hotkeys = sorted(p.hotkey for p in prepared.values() if p.clone_ok)
        verdicts: list[PairVerdict] = _hash_dup_pairs(prepared)
        hash_pairs = [(v.hotkey_a, v.hotkey_b) for v in verdicts]
        hash_set = set(hash_pairs)
        repo_tokens = {r.github_token for r in repos if r.github_token}

        # Give the agent the validator's running source so it can verify whether claimed
        # differentiators are real training-time inputs (vs evasive dead code). Best-effort.
        has_source = await asyncio.to_thread(_snapshot_god_source, temp_root / "_god_source")
        runtime_context = _build_runtime_context(has_source)

        dup_pairs: list[tuple[str, str]] = list(hash_pairs)
        evasion: set[str] = set()

        # Collapse hash-duplicate clusters to one representative each so T2 only compares
        # representatives. hash_pairs already link every member to its rep, so a duplicate
        # verdict between two reps merges their whole clusters in the final clustering.
        representatives = _hash_group_representatives(ok_hotkeys, hash_pairs, boss_hotkey)
        pairs_to_judge = [(a, b) for a, b in itertools.combinations(representatives, 2) if (a, b) not in hash_set]
        logger.info(
            f"dedup T2: {len(representatives)} representatives after hash-collapse "
            f"({len(hash_pairs)} hash-dup pairs), {len(pairs_to_judge)} pairs to judge with Claude"
        )
        for idx, (a, b) in enumerate(pairs_to_judge, 1):
            pa, pb = prepared[a], prepared[b]
            logger.info(f"dedup T2 pair {idx}/{len(pairs_to_judge)}: {a[:8]} vs {b[:8]} — judging…")
            started = time.monotonic()
            file_summary = await asyncio.to_thread(_file_summary, Path(str(pa.path)), Path(str(pb.path)))
            verdict = await _judge_pair(temp_root, a, b, file_summary, runtime_context)
            verdict.reason = _sanitize_reason(verdict.reason, repo_tokens)
            verdicts.append(verdict)
            logger.info(
                f"dedup T2 pair {idx}/{len(pairs_to_judge)}: {a[:8]} vs {b[:8]} → "
                f"{verdict.relationship.value} (conf {verdict.confidence:.2f}) in {time.monotonic() - started:.0f}s"
            )
            if verdict.relationship == DupRelationship.DUPLICATE:
                dup_pairs.append((a, b))
            elif verdict.relationship == DupRelationship.DROP_EVASION:
                evasion.add(a if pa.content_chars >= pb.content_chars else b)

        clusters = []
        for members in _cluster(ok_hotkeys, dup_pairs):
            basis, reason = _cluster_basis(members, prepared)
            clusters.append(DedupCluster(members=members, basis=basis, reason=reason))

        flagged = _flag_from_clusters(clusters, boss_hotkey)
        flagged.update(evasion)
        if boss_hotkey:
            flagged.discard(boss_hotkey)

        return DedupResult(
            cohort=[r.hotkey for r in repos],
            clusters=clusters,
            pair_verdicts=verdicts,
            flagged_hotkeys=sorted(flagged),
            evasion_hotkeys=sorted(evasion),
            unclonable_hotkeys=sorted(h for h, p in prepared.items() if not p.clone_ok),
        )
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def render_report(
    result: DedupResult,
    tournament_id: str,
    round_id: str,
    boss_hotkey: str | None,
    include_distinct_verdicts: bool = False,
) -> str:
    """Human-readable markdown report of a dedup result (uploaded for review).

    The uploaded copy is publicly linked (Discord + /auditing/dedup), so by default DISTINCT
    pair verdicts are omitted: their reasons describe how non-flagged miners' repos differ,
    i.e. their training innovations. Full verdicts stay in the DB review row."""
    lines = [
        f"# Tournament dedup review — {tournament_id}",
        f"Guarded round: `{round_id}`",
        f"Cohort size: {len(result.cohort)}  |  Flagged for removal: {len(result.flagged_hotkeys)}",
        "",
        "## Clusters (functional duplicates)",
    ]
    if result.clusters:
        for i, cluster in enumerate(result.clusters, 1):
            kept = f" (boss `{boss_hotkey}` kept)" if boss_hotkey and boss_hotkey in cluster.members else ""
            lines.append(f"\n### Cluster {i} — basis {cluster.basis.value}{kept}")
            lines.append(f"{cluster.reason}")
            for m in cluster.members:
                tag = " — KEPT (boss)" if m == boss_hotkey else " — DROP"
                lines.append(f"- `{m}`{tag}")
    else:
        lines.append("\nNone.")

    if result.evasion_hotkeys:
        lines.append("\n## Evasion (padding/obfuscation) — DROP")
        lines.extend(f"- `{h}`" for h in result.evasion_hotkeys)
    if result.unclonable_hotkeys:
        lines.append("\n## Could not clone (not flagged)")
        lines.extend(f"- `{h}`" for h in result.unclonable_hotkeys)

    if include_distinct_verdicts:
        shown = result.pair_verdicts
        lines.append("\n## All pairwise verdicts")
    else:
        shown = [v for v in result.pair_verdicts if v.relationship != DupRelationship.DISTINCT]
        omitted = len(result.pair_verdicts) - len(shown)
        lines.append("\n## Flagged pairwise verdicts")
        if omitted:
            lines.append(f"({omitted} distinct pair(s) omitted from the published report)")
    for v in shown:
        lines.append(
            f"- [{v.tier.value}] `{v.hotkey_a}` vs `{v.hotkey_b}`: "
            f"**{v.relationship.value}** ({v.confidence:.2f}) — {v.reason}"
        )

    lines.append("\n## Recommended eliminations")
    lines.extend(f"- `{h}`" for h in result.flagged_hotkeys) if result.flagged_hotkeys else lines.append("- none")
    return "\n".join(lines) + "\n"
