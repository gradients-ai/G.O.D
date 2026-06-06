"""R1/R2 submission de-duplication gate orchestration (anti-spam).

R1 (pre-training): detect exact/normalized hash duplicates and auto-eliminate (boss kept).
R2 (R1->R2 advance): run the Claude pairwise check, write a review row, ping Discord and HALT
the tournament until a human approves in the DB; on approval, publish the offending repos to
the public gradients-opensource org, eliminate them, and proceed.
"""

import shutil
import tempfile
import time
from pathlib import Path

from core.models.tournament_models import DedupClusterRecord
from core.models.tournament_models import DedupPairVerdict
from core.models.tournament_models import DedupResult
from core.models.tournament_models import DedupReviewStatus
from core.models.tournament_models import GateDecision
from core.models.tournament_models import RepoRef
from core.models.tournament_models import PublishedRepo
from core.models.tournament_models import TournamentData
from core.models.tournament_models import TournamentDedupReview
from core.models.tournament_models import TournamentRoundData
from core.models.tournament_models import generate_round_id
from validator.core import constants as cst
from validator.core.config import Config
from validator.db.database import PSQLDB
from validator.db.sql.dedup import get_dedup_review
from validator.db.sql.dedup import insert_dedup_review
from validator.db.sql.dedup import mark_dedup_review_resolved
from validator.db.sql.tournaments import eliminate_tournament_participants
from validator.db.sql.tournaments import get_tournament_participants
from validator.tournament.repo_uploader import upload_flagged_duplicate_repository
from validator.tournament.utils import notify_tournament_dedup_autoremoved
from validator.tournament.utils import notify_tournament_dedup_error
from validator.tournament.utils import notify_tournament_dedup_resolved
from validator.tournament.utils import notify_tournament_dedup_review
from validator.utils.logging import get_logger
from validator.utils.repo_dedup import find_hash_duplicates
from validator.utils.repo_dedup import render_report
from validator.utils.repo_dedup import run_pairwise_dedup
from validator.utils.util import upload_file_to_minio


logger = get_logger(__name__)

# Round IDs whose T2 gate raised this process; held without re-running (T2 is expensive).
# Cleared on restart (retries once) or bypassed via a DB skip row.
_GATE_FAILED: set[str] = set()


def _repo_refs(participants, hotkeys: set[str]) -> list[RepoRef]:
    return [
        RepoRef(
            hotkey=p.hotkey,
            repo_url=p.training_repo,
            commit_hash=p.training_commit_hash,
            github_token=p.github_token,
        )
        for p in participants
        if p.hotkey in hotkeys and p.training_repo
    ]


def _to_records(result: DedupResult) -> tuple[list[DedupClusterRecord], list[DedupPairVerdict]]:
    clusters = [DedupClusterRecord(members=c.members, basis=c.basis, reason=c.reason) for c in result.clusters]
    verdicts = [
        DedupPairVerdict(
            hotkey_a=v.hotkey_a, hotkey_b=v.hotkey_b, tier=v.tier, relationship=v.relationship, confidence=v.confidence, reason=v.reason
        )
        for v in result.pair_verdicts
    ]
    return clusters, verdicts


async def _upload_report(result: DedupResult, tournament: TournamentData, round_id: str) -> str | None:
    if not cst.BUCKET_NAME:
        logger.warning("S3_BUCKET_NAME not set; skipping dedup report upload")
        return None
    report = render_report(result, tournament.tournament_id, round_id, cst.EMISSION_BURN_HOTKEY)
    temp_dir = Path(tempfile.mkdtemp(prefix="dedup-report-"))
    try:
        path = temp_dir / "dedup_report.md"
        path.write_text(report)
        object_name = (
            f"tournament-dedup-reports/{tournament.tournament_type.value}/"
            f"{tournament.tournament_id}-{round_id}-{int(time.time())}.md"
        )
        return await upload_file_to_minio(str(path), cst.BUCKET_NAME, object_name)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# --------------------------------------------------------------------------- #
# R1: deterministic hash de-dup (detection only; caller eliminates after round exists)
# --------------------------------------------------------------------------- #
async def detect_r1_hash_duplicates(tournament: TournamentData, candidate_hotkeys: list[str], psql_db: PSQLDB) -> DedupResult:
    participants = await get_tournament_participants(tournament.tournament_id, psql_db)
    refs = _repo_refs(participants, set(candidate_hotkeys))
    if len(refs) < 2:
        return DedupResult(cohort=[r.hotkey for r in refs])
    logger.info(f"R1 hash dedup: checking {len(refs)} repos for tournament {tournament.tournament_id}")
    return await find_hash_duplicates(refs, boss_hotkey=cst.EMISSION_BURN_HOTKEY)


async def apply_r1_eliminations(
    tournament: TournamentData, round_id: str, result: DedupResult, config: Config, psql_db: PSQLDB
) -> None:
    flagged = sorted(set(result.flagged_hotkeys))
    if not flagged:
        return
    await eliminate_tournament_participants(tournament.tournament_id, round_id, flagged, psql_db)
    logger.info(f"R1 hash dedup auto-eliminated {len(flagged)} duplicate(s): {flagged}")
    if config.discord_url:
        await notify_tournament_dedup_autoremoved(
            tournament.tournament_id, tournament.tournament_type.value, result.clusters, flagged, config.discord_url
        )


# --------------------------------------------------------------------------- #
# R2: Claude pairwise gate with human approval
# --------------------------------------------------------------------------- #
async def evaluate_r2_dedup_gate(
    tournament: TournamentData, completed_round: TournamentRoundData, winners: list[str], config: Config, psql_db: PSQLDB
) -> GateDecision:
    next_round_id = generate_round_id(tournament.tournament_id, completed_round.round_number + 1)
    existing = await get_dedup_review(next_round_id, psql_db)

    if existing is None:
        if next_round_id in _GATE_FAILED:
            # Failed earlier this process — hold without re-running the expensive T2 check.
            logger.warning(f"Dedup gate {next_round_id}: held after earlier failure — restart or DB-skip to retry")
            return GateDecision(halt=True)
        try:
            return await _run_and_record_gate(tournament, next_round_id, winners, config, psql_db)
        except Exception as exc:
            # Gate failed before writing a review row (clone/API/parse). Don't advance past an
            # unevaluated gate, and don't re-run it every cycle — halt, remember, ping once.
            _GATE_FAILED.add(next_round_id)
            logger.error(f"Dedup gate {next_round_id} failed to evaluate — halting tournament: {exc}", exc_info=True)
            if config.discord_url:
                await notify_tournament_dedup_error(
                    tournament.tournament_id, tournament.tournament_type.value, next_round_id, str(exc), config.discord_url
                )
            return GateDecision(halt=True)

    if existing.status == DedupReviewStatus.PENDING_REVIEW:
        logger.info(f"Dedup gate {next_round_id}: still pending manual review — holding tournament")
        return GateDecision(halt=True)

    if existing.status == DedupReviewStatus.SKIPPED:
        return GateDecision(halt=False)

    # APPROVED
    eliminate = {h for h in existing.approved_eliminations if h != cst.EMISSION_BURN_HOTKEY}
    if existing.resolved_at is not None:
        return GateDecision(halt=False, eliminate=eliminate)
    await _apply_approved_gate(tournament, completed_round, existing, eliminate, config, psql_db)
    return GateDecision(halt=False, eliminate=eliminate)


async def _run_and_record_gate(
    tournament: TournamentData, next_round_id: str, winners: list[str], config: Config, psql_db: PSQLDB
) -> GateDecision:
    participants = await get_tournament_participants(tournament.tournament_id, psql_db)
    refs = _repo_refs(participants, set(winners))
    if len(refs) < 2:
        return GateDecision(halt=False)

    logger.info(f"Dedup gate {next_round_id}: running Claude pairwise check on {len(refs)} R2 entrants")
    result = await run_pairwise_dedup(refs, boss_hotkey=cst.EMISSION_BURN_HOTKEY)
    clusters, verdicts = _to_records(result)

    if not result.flagged_hotkeys:
        review = TournamentDedupReview(
            round_id=next_round_id,
            tournament_id=tournament.tournament_id,
            tournament_type=tournament.tournament_type.value,
            status=DedupReviewStatus.SKIPPED,
            cohort=result.cohort,
            clusters=clusters,
            pair_verdicts=verdicts,
        )
        await insert_dedup_review(review, psql_db)
        logger.info(f"Dedup gate {next_round_id}: no duplicates flagged — proceeding")
        return GateDecision(halt=False)

    report_url = await _upload_report(result, tournament, next_round_id)
    review = TournamentDedupReview(
        round_id=next_round_id,
        tournament_id=tournament.tournament_id,
        tournament_type=tournament.tournament_type.value,
        status=DedupReviewStatus.PENDING_REVIEW,
        cohort=result.cohort,
        clusters=clusters,
        pair_verdicts=verdicts,
        flagged_hotkeys=result.flagged_hotkeys,
        approved_eliminations=result.flagged_hotkeys,
        report_url=report_url,
    )
    await insert_dedup_review(review, psql_db)
    logger.warning(
        f"Dedup gate {next_round_id}: {len(result.flagged_hotkeys)} flagged — HALTING tournament pending manual review"
    )
    if config.discord_url:
        await notify_tournament_dedup_review(
            tournament.tournament_id,
            tournament.tournament_type.value,
            next_round_id,
            clusters,
            result.flagged_hotkeys,
            report_url,
            config.discord_url,
        )
    return GateDecision(halt=True)


async def _apply_approved_gate(
    tournament: TournamentData,
    completed_round: TournamentRoundData,
    review: TournamentDedupReview,
    eliminate: set[str],
    config: Config,
    psql_db: PSQLDB,
) -> None:
    participants = {p.hotkey: p for p in await get_tournament_participants(tournament.tournament_id, psql_db)}

    published: list[PublishedRepo] = []
    for hotkey in sorted(eliminate):
        participant = participants.get(hotkey)
        if not participant or not participant.training_repo:
            continue
        url = await upload_flagged_duplicate_repository(
            tournament.tournament_id,
            tournament.tournament_type.value,
            hotkey,
            participant.training_repo,
            participant.training_commit_hash,
            config,
            participant.github_token,
        )
        if url:
            published.append(PublishedRepo(hotkey=hotkey, public_repo_url=url, commit_hash=participant.training_commit_hash))

    if eliminate:
        # eliminated_in_round_id FKs tournament_rounds; use the completed R1 round (R2 not created yet)
        await eliminate_tournament_participants(tournament.tournament_id, completed_round.round_id, sorted(eliminate), psql_db)

    await mark_dedup_review_resolved(review.round_id, published, review.report_url, psql_db)
    logger.info(f"Dedup gate {review.round_id}: applied — eliminated {len(eliminate)}, published {len(published)} repos")
    if config.discord_url:
        await notify_tournament_dedup_resolved(
            tournament.tournament_id, tournament.tournament_type.value, sorted(eliminate), published, config.discord_url
        )
