"""The link between a quant tournament and the text tournament its tasks come from.

A quant tournament takes every task from one text tournament: each quant task is a new task of its own (own id, own training
rows, own GPU reservations, own status) that compresses the model the text boss trained on one of the text tournament's seed
tasks. The two tournaments never share a task row, a result or a status. The link is stored, not derived from names:
  * quant_tournament_sources: which text tournament feeds which quant tournament (bind_quant_source_tournament);
  * quant_task_sources: per quant task, its slot, the text seed task it compresses and, once released, the exact model repo.

This module binds, resolves the text seed of a slot and records a new quant task's source so that a slot never ends up with
two live tasks and a quant task never exists without its link.
"""

from datetime import datetime
from datetime import timedelta
from datetime import timezone
from uuid import UUID

from core.logging import get_logger
from validator.app.config import Config
from validator.db.sql import quant_sources as sources_sql
from validator.db.sql import tasks as task_sql
from validator.db.sql import tournament_seeds as seeds_sql
from validator.db.sql import tournaments as tournament_sql
from validator.tournament import constants as t_cst
from validator.tournament.models import QUANT_SLOT_SPECS
from validator.tournament.models import TEXT_SEED_ROLES
from validator.tournament.models import QuantSourceSlot
from validator.tournament.models import QuantTaskSource
from validator.tournament.models import QuantTournamentSource
from validator.tournament.models import SeedRole
from validator.tournament.models import SeedStatus
from validator.tournament.models import TournamentData
from validator.tournament.models import TournamentSeedTask
from validator.tournament.models import TournamentStatus
from validator.tournament.models import TournamentType
from validator.tournament.notifications import send_to_discord


logger = get_logger(__name__)

# Quant tournaments waiting for a text tournament, and when this process first saw each waiting (the tournaments table has no
# "became active" timestamp, so a restart restarts the wait). Waiting is logged once per process; the alert is sent once.
_WAITING_SINCE: dict[str, datetime] = {}
_WAIT_ALERTED: set[str] = set()

_LIVE_SEED_STATUSES = frozenset({SeedStatus.TRAINING, SeedStatus.TRAINED, SeedStatus.READY})
_RUNNING_TEXT_STATUSES = (TournamentStatus.ACTIVE, TournamentStatus.PENDING)


async def _has_boss_seeds(tournament: TournamentData, completed: bool, config: Config) -> bool:
    """Whether a text tournament has its four boss seed roles. An active one needs each role live or `failed` (a role whose
    replacement is pending will be live again, so the quant tournament waits for it); a completed one has final seed state, so
    a row of any status per role is enough. A tournament without seed rows started before seeding and is never bound."""
    latest = {seed.role: seed for seed in await seeds_sql.get_latest_seed_tasks(tournament.tournament_id, config.psql_db)}
    if not TEXT_SEED_ROLES <= latest.keys():
        return False
    return completed or all(latest[role].status != SeedStatus.REPLACED for role in TEXT_SEED_ROLES)


async def _alert_if_waiting_too_long(quant_tournament_id: str, config: Config) -> None:
    """Log once that the quant tournament is waiting, and send one Discord alert once it has waited
    QUANT_SOURCE_WAIT_ALERT_SECONDS. The wait is measured from the first time this process saw it waiting."""
    now = datetime.now(timezone.utc)
    if quant_tournament_id not in _WAITING_SINCE:
        _WAITING_SINCE[quant_tournament_id] = now
        logger.info(
            f"Quant tournament {quant_tournament_id} is waiting for a text tournament with its boss tasks "
            f"(all of {sorted(role.value for role in TEXT_SEED_ROLES)}) that feeds no quant tournament"
        )
    waited = now - _WAITING_SINCE[quant_tournament_id]
    if quant_tournament_id in _WAIT_ALERTED or waited < timedelta(seconds=t_cst.QUANT_SOURCE_WAIT_ALERT_SECONDS):
        return
    _WAIT_ALERTED.add(quant_tournament_id)
    message = (
        f"Quant tournament {quant_tournament_id} has waited {int(waited.total_seconds() // 60)} minutes for a text tournament "
        f"to take its tasks from: none that feeds no quant tournament yet has all four boss seed roles. It creates nothing "
        f"until one does."
    )
    logger.error(message)
    if not config.discord_url:
        return
    try:
        await send_to_discord(webhook=config.discord_url, message=message)
    except Exception as e:
        logger.error(f"Failed to send Discord notification: {e}")


async def bind_quant_source_tournament(quant_tournament_id: str, config: Config) -> QuantTournamentSource | None:
    """The text tournament a quant tournament takes its tasks from.

    Returns the stored binding when there is one. Otherwise picks the most recent text tournament that has its four boss seed
    roles and feeds no quant tournament yet, ACTIVE ones first, then those COMPLETED within
    QUANT_SOURCE_COMPLETED_WINDOW_SECONDS. An active tournament qualifies when each role is live or `failed` (a failed role
    never blocks: its quant slot takes the fallback); a completed one when each role has a row, its seed state being final.
    A text tournament without seed rows is never bound. Stores the binding and returns what the database holds (so two passes
    or two processes agree). None when there is no such text tournament: the quant tournament waits, nothing is created and
    nothing is halted. The wait is logged once per process and alerted once after QUANT_SOURCE_WAIT_ALERT_SECONDS, measured
    from the first time this process saw the quant tournament waiting (the tournaments table keeps no activation time).
    """
    psql_db = config.psql_db
    bound = await sources_sql.get_quant_tournament_source(quant_tournament_id, psql_db)
    if bound is not None:
        return bound
    taken = await sources_sql.get_bound_source_tournament_ids(psql_db)
    active = await tournament_sql.get_tournaments_with_status(TournamentStatus.ACTIVE, psql_db)
    since = datetime.now(timezone.utc) - timedelta(seconds=t_cst.QUANT_SOURCE_COMPLETED_WINDOW_SECONDS)
    completed = await tournament_sql.get_completed_tournaments_since(TournamentType.TEXT, since, psql_db)
    for tournament, is_completed in [(t, False) for t in active] + [(t, True) for t in completed]:
        if tournament.tournament_type != TournamentType.TEXT or tournament.tournament_id in taken:
            continue
        if not await _has_boss_seeds(tournament, is_completed, config):
            continue
        stored = await sources_sql.bind_quant_tournament(quant_tournament_id, tournament.tournament_id, psql_db)
        if stored is not None:
            logger.info(f"Quant tournament {quant_tournament_id} is bound to text tournament {stored.source_tournament_id}")
            _WAITING_SINCE.pop(quant_tournament_id, None)
            return stored
        # The text tournament was taken between the read and the insert; try the next one.
        taken.add(tournament.tournament_id)
    await _alert_if_waiting_too_long(quant_tournament_id, config)
    return None


async def final_text_seed(source_tournament_id: str, text_role: SeedRole, config: Config) -> TournamentSeedTask | None:
    """The text seed row a quant task of the role is chained on: the live row, else the role's latest row when it is final
    (status `failed`, or the text tournament is over, whatever the row's status). None only while a replacement is pending in a
    running text tournament, or when the role has no row."""
    latest = next(
        (seed for seed in await seeds_sql.get_latest_seed_tasks(source_tournament_id, config.psql_db) if seed.role == text_role),
        None,
    )
    if latest is None or latest.status in _LIVE_SEED_STATUSES or latest.status == SeedStatus.FAILED:
        return latest
    tournament = await tournament_sql.get_tournament(source_tournament_id, config.psql_db)
    if tournament is not None and tournament.status not in _RUNNING_TEXT_STATUSES:
        return latest
    return None


async def slot_text_seed(binding: QuantTournamentSource, slot: QuantSourceSlot, config: Config) -> TournamentSeedTask | None:
    """The text seed row a slot of the bound text tournament compresses (see final_text_seed): a role that is out of replacements
    still gives its failed row, so the quant task is created on it and takes the fallback model."""
    return await final_text_seed(binding.source_tournament_id, QUANT_SLOT_SPECS[slot].text_role, config)


def new_task_source(
    quant_task_id: UUID, binding: QuantTournamentSource, slot: QuantSourceSlot, text_seed: TournamentSeedTask
) -> QuantTaskSource:
    """The source row of a freshly created, still held quant task."""
    return QuantTaskSource(
        quant_task_id=quant_task_id,
        quant_tournament_id=binding.quant_tournament_id,
        slot=slot,
        source_tournament_id=text_seed.tournament_id,
        source_role=text_seed.role,
        source_task_id=text_seed.task_id,
    )


async def discard_quant_task(task_id: UUID | str, config: Config) -> None:
    """Delete a just-created quant task that must not exist (a duplicate for its slot, or one that could not be linked)."""
    try:
        await task_sql.delete_task(UUID(str(task_id)), config.psql_db)
    except Exception:
        logger.error(f"Could not delete quant task {task_id}; remove it by hand", exc_info=True)


async def record_new_quant_task(
    source: QuantTaskSource, config: Config, replacing_task_id: UUID | None = None
) -> QuantTaskSource:
    """Record the source of a quant task that was just created and return the live source row of its slot.

    The slot already having a live task (a concurrent pass got there first) is not an error: the just-created task is
    deleted and the existing row is returned, so the caller sees that quant_task_id differs from its own. Any other failure
    deletes the task and raises, so a quant task never exists without its link.
    """
    try:
        result = await sources_sql.insert_quant_task_source(source, config.psql_db, replacing_task_id)
    except Exception:
        await discard_quant_task(source.quant_task_id, config)
        raise
    if not result.inserted:
        logger.warning(
            f"Quant tournament {source.quant_tournament_id} slot {source.slot.value} already has task "
            f"{result.source.quant_task_id}; deleting the duplicate {source.quant_task_id}"
        )
        await discard_quant_task(source.quant_task_id, config)
    return result.source
