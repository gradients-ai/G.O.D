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
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from validator.core import constants as cst
from validator.utils.logging import get_logger
from validator.utils.repo_diff_report import _clone_repo
from validator.utils.repo_diff_report import _collect_files


logger = get_logger(__name__)

CONFIG_PATH = Path(__file__).with_name("repo_dedup_config.json")


@lru_cache(maxsize=1)
def _load_config() -> dict[str, Any]:
    with CONFIG_PATH.open() as handle:
        return json.load(handle)


class RepoRef(BaseModel):
    hotkey: str
    repo_url: str
    commit_hash: str | None = None
    github_token: str | None = None


class PreparedRepo(BaseModel):
    hotkey: str
    repo_url: str
    head_commit: str | None = None
    normalized_digest: str | None = None
    content_chars: int = 0
    path: str | None = None
    clone_ok: bool = False


class PairVerdict(BaseModel):
    hotkey_a: str
    hotkey_b: str
    tier: str  # "T0" | "T1" | "T2"
    relationship: str  # "duplicate" | "distinct" | "drop_evasion"
    confidence: float
    reason: str


class DedupCluster(BaseModel):
    members: list[str]
    basis: str  # "T0" | "T1" | "T2"
    reason: str


class DedupResult(BaseModel):
    cohort: list[str]
    clusters: list[DedupCluster] = []
    pair_verdicts: list[PairVerdict] = []
    flagged_hotkeys: list[str] = []  # recommended eliminations (boss excluded)
    evasion_hotkeys: list[str] = []
    unclonable_hotkeys: list[str] = []


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


def _hash_dup_pairs(prepared: dict[str, PreparedRepo]) -> tuple[list[tuple[str, str]], dict[tuple[str, str], str]]:
    ok = sorted((p for p in prepared.values() if p.clone_ok), key=lambda p: p.hotkey)
    pairs: list[tuple[str, str]] = []
    tier: dict[tuple[str, str], str] = {}
    for a, b in itertools.combinations(ok, 2):
        key = (a.hotkey, b.hotkey)
        if a.head_commit and a.head_commit == b.head_commit:
            pairs.append(key)
            tier[key] = "T0"
        elif a.normalized_digest and a.normalized_digest == b.normalized_digest:
            pairs.append(key)
            tier[key] = "T1"
    return pairs, tier


def _cluster_basis(members: list[str], prepared: dict[str, PreparedRepo]) -> tuple[str, str]:
    if len({prepared[m].head_commit for m in members}) == 1:
        return "T0", "Identical commit hash."
    if len({prepared[m].normalized_digest for m in members}) == 1:
        return "T1", "Identical source after stripping whitespace/comments/ordering."
    return "T2", "Judged functionally equivalent by Claude (cosmetic/evasive deltas only)."


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
# Claude pairwise judgement (T2)
# --------------------------------------------------------------------------- #
_VALID_RELATIONSHIPS = {"duplicate", "distinct", "drop_evasion"}


def _import_claude_sdk():
    try:
        from claude_agent_sdk import ClaudeAgentOptions
        from claude_agent_sdk import ResultMessage
        from claude_agent_sdk import query
    except ImportError as exc:
        raise RuntimeError("claude-agent-sdk is required for pairwise dedup") from exc
    return ClaudeAgentOptions, ResultMessage, query


def _parse_verdict(text: str) -> tuple[str, float, str]:
    """Extract the strict JSON verdict object from the model's reply."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"no JSON object in verdict reply: {text[:200]!r}")
    obj = json.loads(text[start : end + 1])
    relationship = str(obj["relationship"]).strip()
    if relationship not in _VALID_RELATIONSHIPS:
        raise ValueError(f"invalid relationship {relationship!r}")
    return relationship, float(obj.get("confidence", 0.0)), str(obj.get("reason", ""))


async def _judge_pair(cwd: Path, dir_a: str, dir_b: str, file_summary: str) -> PairVerdict:
    """Read-only agentic judgement over both cloned repos checked out under ``cwd``.

    The miners' private repo URLs are deliberately NOT passed to the model — it sees only
    the neutral on-disk dirs and is instructed to reason in terms of "Repository A/B" — so
    the published reasoning never leaks the original private repo/org names."""
    ClaudeAgentOptions, ResultMessage, query = _import_claude_sdk()
    config = _load_config()
    prompt = config["user_prompt_template"].format(dir_a=dir_a, dir_b=dir_b, file_summary=file_summary)
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
            return PairVerdict(hotkey_a="", hotkey_b="", tier="T2", relationship=relationship, confidence=confidence, reason=reason)
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
        pairs, tier = _hash_dup_pairs(prepared)

        verdicts = [
            PairVerdict(
                hotkey_a=a,
                hotkey_b=b,
                tier=tier[(a, b)],
                relationship="duplicate",
                confidence=1.0,
                reason="Identical commit hash." if tier[(a, b)] == "T0" else "Identical normalized content.",
            )
            for (a, b) in pairs
        ]
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
        hash_pairs, tier = _hash_dup_pairs(prepared)
        hash_set = set(hash_pairs)

        dup_pairs: list[tuple[str, str]] = list(hash_pairs)
        evasion: set[str] = set()
        verdicts: list[PairVerdict] = [
            PairVerdict(
                hotkey_a=a,
                hotkey_b=b,
                tier=tier[(a, b)],
                relationship="duplicate",
                confidence=1.0,
                reason="Identical commit hash." if tier[(a, b)] == "T0" else "Identical normalized content.",
            )
            for (a, b) in hash_pairs
        ]

        # Collapse hash-duplicate clusters to one representative each so T2 only compares
        # representatives. hash_pairs already link every member to its rep, so a duplicate
        # verdict between two reps merges their whole clusters in the final clustering.
        representatives = _hash_group_representatives(ok_hotkeys, hash_pairs, boss_hotkey)
        for a, b in itertools.combinations(representatives, 2):
            if (a, b) in hash_set:
                continue
            pa, pb = prepared[a], prepared[b]
            file_summary = await asyncio.to_thread(_file_summary, Path(str(pa.path)), Path(str(pb.path)))
            verdict = await _judge_pair(temp_root, a, b, file_summary)
            verdict.hotkey_a, verdict.hotkey_b = a, b
            verdicts.append(verdict)
            if verdict.relationship == "duplicate":
                dup_pairs.append((a, b))
            elif verdict.relationship == "drop_evasion":
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


def render_report(result: DedupResult, tournament_id: str, round_id: str, boss_hotkey: str | None) -> str:
    """Human-readable markdown report of a dedup result (uploaded for review)."""
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
            lines.append(f"\n### Cluster {i} — basis {cluster.basis}{kept}")
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

    lines.append("\n## All pairwise verdicts")
    for v in result.pair_verdicts:
        lines.append(f"- [{v.tier}] `{v.hotkey_a}` vs `{v.hotkey_b}`: **{v.relationship}** ({v.confidence:.2f}) — {v.reason}")

    lines.append("\n## Recommended eliminations")
    lines.extend(f"- `{h}`" for h in result.flagged_hotkeys) if result.flagged_hotkeys else lines.append("- none")
    return "\n".join(lines) + "\n"
