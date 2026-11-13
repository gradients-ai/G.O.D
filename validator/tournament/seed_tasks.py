"""Seed tasks: everything the boss trains, created and trained upfront when a tournament's first round is created.

See validator/tournament/README.md. A text tournament has four seeds (TEXT_SEED_CREATION_ORDER): the instruct task of rounds
1-3 and the continuous-SFT task of the boss round. A quant tournament has six (SEED_CREATION_ORDER): the freeform and the fixed
compression of the text boss's round-2, round-3 and continuous models. A quant seed is a new chained quant task on a seed of
the BOUND text tournament (see quant_sources.py), held until the text boss's model has landed, then trained by the quant boss
with the quant tournament's pinned code. Each seed is an ordinary task assigned to the boss's node (EMISSION_BURN_HOTKEY) and
trained with the boss participant's pinned repo + commit by the normal pipeline (prep -> model prep -> LOOKING_FOR_NODES ->
orchestrator). The input-seed roles (input_small, input_large, boss_instruct, boss_chained_*) are not created; rows of them still
load and are replaced.
Seeds are not tournament_tasks rows: the boss round attaches its compression seeds later. This module owns creation, status,
replacement and the halt (only an input-seed role halts a tournament; a text or quant role out of replacements is
`failed`, alerts once and the tournament goes on without it).

Entry points:
  * live_seed_tasks(tournament_id, db) -> dict[SeedRole, TournamentSeedTask]: the current row of every role.
  * seeds_sql.boss_round_seed_ready_to_finalise(task_id, db): the hold on every path that finalises a seed (its rows are
    evaluated as they train, the boss's at seed time).
  * seed_halt_reason(tournament_id, db): a non-empty reason while an input-seed role is out of replacements.
"""

import random
from typing import AsyncGenerator
from uuid import UUID

from fiber.chain.models import Node
from huggingface_hub.errors import RepositoryNotFoundError

from core.constants.quant import QUANT_MODEL_SIZE_B
from core.logging import get_logger
from core.logging import redact_url_queries
from core.models.task_models import TaskStatus
from core.models.task_models import TaskType
from validator.app.config import Config
from validator.db.database import PSQLDB
from validator.db.sql import quant_sources as quant_sources_sql
from validator.db.sql import tasks as task_sql
from validator.db.sql import tournament_seeds as seeds_sql
from validator.db.sql import tournaments as tournament_sql
from validator.db.sql.continuous_sft import get_continuous_sft_task_chunk
from validator.db.sql.nodes import get_node_by_hotkey
from validator.scoring.constants import EMISSION_BURN_HOTKEY
from validator.tasks.datasets.models import Dataset
from validator.tasks.models import ChatRawTask
from validator.tasks.models import InstructTextRawTask
from validator.tasks.models import RawTask
from validator.tasks.quant_input_repo import full_model_rejection
from validator.tasks.quant_input_repo import is_adapter_repo
from validator.tasks.quant_shapes import filter_quant_compatible
from validator.tasks.synthetics.scheduler import _get_instruct_text_datasets
from validator.tasks.synthetics.scheduler import _get_text_models
from validator.tasks.synthetics.scheduler import create_chained_continuous_quant_task
from validator.tasks.synthetics.scheduler import create_chained_quant_task
from validator.tasks.synthetics.scheduler import create_continuous_sft_task
from validator.tasks.synthetics.scheduler import create_synthetic_instruct_text_task
from validator.tasks.synthetics.scheduler import draw_quant_format
from validator.tasks.synthetics.scheduler import repoint_chained_quant_task
from validator.tournament import chaining
from validator.tournament import constants as t_cst
from validator.tournament import quant_sources
from validator.tournament.models import ANY_BOSS_COMPRESSION_SEED_ROLES
from validator.tournament.models import ANY_BOSS_ROUND_SEED_ROLES
from validator.tournament.models import BOSS_ROUND_SEED_ROLES
from validator.tournament.models import INPUT_SEED_ROLES
from validator.tournament.models import QUANT_BOSS_SLOT_BY_ROLE
from validator.tournament.models import QUANT_SLOT_SPECS
from validator.tournament.models import TEXT_ROUND_SEED_ROLES
from validator.tournament.models import TEXT_SEED_ROLES
from validator.tournament.models import SeedRole
from validator.tournament.models import SeedStatus
from validator.tournament.models import TournamentSeedTask
from validator.tournament.models import TournamentStatus
from validator.tournament.models import TournamentType
from validator.tournament.models import TrainingRepoInfo
from validator.tournament.models import TrainingStatus
from validator.tournament.notifications import send_to_discord
from validator.tournament.participants import get_latest_commit_hash_from_github
from validator.tournament.task_creator import swap_seed_task_registration


logger = get_logger(__name__)

# A quant tournament's seeds: the freeform and fixed compression of the text boss's round-2, round-3 and continuous models.
SEED_CREATION_ORDER = (
    SeedRole.BOSS_R2_FREEFORM,
    SeedRole.BOSS_R2_FIXED,
    SeedRole.BOSS_R3_FREEFORM,
    SeedRole.BOSS_R3_FIXED,
    SeedRole.BOSS_CONTINUOUS_FREEFORM,
    SeedRole.BOSS_CONTINUOUS_FIXED,
)
# A text tournament's seeds: the instruct task of rounds 1-3, then the continuous-SFT task of the boss round.
TEXT_SEED_CREATION_ORDER = (
    SeedRole.TEXT_R1,
    SeedRole.TEXT_R2,
    SeedRole.TEXT_R3,
    SeedRole.TEXT_CONTINUOUS,
)
# The instruct seeds the boss alone trains and whose output is the input of a later task: R1-R3 chained on the two input
# seeds, the compression seeds on BOSS_INSTRUCT. Nothing creates them; a row of one is evaluated (usable full-model boss
# output) and replaced.
INSTRUCT_INPUT_SEED_ROLES = frozenset({*INPUT_SEED_ROLES, SeedRole.BOSS_INSTRUCT})
# The compression roles chained on BOSS_INSTRUCT (only an existing row can be redrawn).
CHAINED_SEED_TASK_TYPES = {
    SeedRole.BOSS_CHAINED_FREEFORM: TaskType.QUANTFREEFORMTASK,
    SeedRole.BOSS_CHAINED_FIXED: TaskType.QUANTFIXEDTASK,
}
# The instruct role whose output each of those compression roles compresses.
COMPRESSION_UPSTREAM_ROLE = {
    SeedRole.BOSS_CHAINED_FREEFORM: SeedRole.BOSS_INSTRUCT,
    SeedRole.BOSS_CHAINED_FIXED: SeedRole.BOSS_INSTRUCT,
}
# The roles whose task draws its own dataset (an ordinary synthetic instruct task). Only these can be redrawn onto a different
# dataset after a prep-stage rejection, so only their prep rejections are uncharged (replace_seed). The compressions reuse
# the instruct seed's data: a same-data redraw cannot fix a prep rejection there, so it counts against SEED_MAX_REPLACEMENTS.
DATASET_DRAWN_ROLES = INSTRUCT_INPUT_SEED_ROLES | TEXT_ROUND_SEED_ROLES
# Consecutive failed merges of one seed's adapter output before the human alert (the merge keeps retrying after it).
SEED_ADAPTER_MERGE_FAILURES_BEFORE_ALERT = 3
# After a failed merge the seed is not merged again until a cooldown has passed: it doubles per consecutive failure from the
# base to the max (seconds).
SEED_ADAPTER_MERGE_COOLDOWN_BASE_SECONDS = 600
SEED_ADAPTER_MERGE_COOLDOWN_MAX_SECONDS = 3600
# Consecutive failures after which the seed's merge is parked (no more attempts in this process) and a human is alerted.
SEED_ADAPTER_MERGE_MAX_FAILURES = 6
# Cycles in a row in which a trainer could not be asked about the merge job (and none had it) before a merge is dispatched anyway.
SEED_ADAPTER_MERGE_UNKNOWN_CYCLES_BEFORE_DISPATCH = 10
# Tasks that ended without a usable result: the seed is replaced.
_DEAD_TASK_STATUSES = frozenset({TaskStatus.PREP_TASK_FAILURE, TaskStatus.FAILURE})

# Tournaments already pinged about a halt. Process-local on purpose (a restart re-alerts a condition needing a human).
_SEED_HALT_ALERTED: set[str] = set()
_UNSEEDED_LOGGED: set[str] = set()
# Non-halting seed alerts already sent, keyed by what happened (tournament + role + kind); same process-local dedupe.
_SEED_ISSUE_ALERTED: set[str] = set()


def seed_creation_order(tournament_type: TournamentType) -> tuple[SeedRole, ...]:
    """The roles a tournament of this type creates seeds for."""
    return TEXT_SEED_CREATION_ORDER if tournament_type == TournamentType.TEXT else SEED_CREATION_ORDER


def _draw_band(role: SeedRole) -> tuple[float, float]:
    for spec in t_cst.TEXT_ROUND_PLAN:
        if spec.seed_role == role:
            return spec.model_size_b
    if role == SeedRole.INPUT_LARGE and not t_cst.SEED_INPUT_LARGE_FALLS_BACK_TO_SMALL:
        return t_cst.SEED_INPUT_LARGE_SIZE_B
    if role in INPUT_SEED_ROLES:
        return t_cst.SEED_INPUT_SMALL_SIZE_B
    raise ValueError(f"{role.value} has no size band of its own")


def _seed_quant_format(tournament_id: str, task_type: TaskType):
    """The tournament-seeded fixed format: stable across replacements and restarts."""
    return draw_quant_format(random.Random(f"{tournament_id}:seed:quant_format:{task_type.value}"))


def _boss_fixed_format(tournament_id: str, role: SeedRole):
    """The format of a fixed quant boss-round task: 2-bit or 3-bit, seeded by tournament AND role, so a restart or a replacement
    redraws the same one and the three fixed tasks may differ."""
    return random.Random(f"{tournament_id}:seed:quant_format:{role.value}").choice(t_cst.QUANT_BOSS_ROUND_FIXED_FORMATS)


async def live_seed_tasks(tournament_id: str, psql_db: PSQLDB) -> dict[SeedRole, TournamentSeedTask]:
    """The current (latest attempt) row of every role that is training, trained or ready."""
    return {seed.role: seed for seed in await seeds_sql.get_active_seed_tasks(tournament_id, psql_db)}


async def seed_halt_reason(tournament_id: str, psql_db: PSQLDB) -> str | None:
    """Why seeds halt the tournament (an input-seed role is out of replacements), or None.

    A quant boss-round compression role or a text role out of replacements is `failed` but never halts: the boss has no result
    on that task, the ordinary crown rule decides it (the challenger takes a task the boss has no valid score on), and the
    tournament goes on (an alert says so when the role fails).
    """
    never_halting = BOSS_ROUND_SEED_ROLES | TEXT_SEED_ROLES
    failed = [seed for seed in await seeds_sql.get_failed_seed_tasks(tournament_id, psql_db) if seed.role not in never_halting]
    if not failed:
        return None
    return "seed role(s) out of replacements: " + ", ".join(f"{seed.role.value} (task {seed.task_id})" for seed in failed)


# --- Boss pin and assignment ---------------------------------------------------------------------------------------


async def _boss_training_info(tournament_id: str, config: Config) -> TrainingRepoInfo:
    """The boss participant's repo and commit; pins the commit from the repo's HEAD when creation could not."""
    info = await tournament_sql.get_tournament_training_repo_and_commit(EMISSION_BURN_HOTKEY, tournament_id, config.psql_db)
    if info.training_repo and not info.training_commit_hash and "github.com/" in info.training_repo:
        commit = await get_latest_commit_hash_from_github(info.training_repo)
        if commit:
            await tournament_sql.set_tournament_participant_training_commit(
                tournament_id, EMISSION_BURN_HOTKEY, commit, config.psql_db
            )
            info = info._replace(training_commit_hash=commit)
    if not info.training_repo or not info.training_commit_hash:
        logger.warning(
            f"Tournament {tournament_id}: boss code is not pinned (repo={info.training_repo!r}, "
            f"commit={info.training_commit_hash!r}); seeds will train an unpinned boss"
        )
    return info


async def _boss_node(config: Config) -> Node:
    node = await get_node_by_hotkey(EMISSION_BURN_HOTKEY, config.psql_db)
    if node is None:
        raise RuntimeError(f"Boss node {EMISSION_BURN_HOTKEY} is not in the nodes table; cannot assign seed tasks")
    return node


async def _ensure_boss_assignment(seed: TournamentSeedTask, config: Config) -> None:
    """Assign the boss's node and expected repo to the seed's task (idempotent)."""
    assigned = {node.hotkey for node in await task_sql.get_nodes_assigned_to_task(str(seed.task_id), config.psql_db)}
    if EMISSION_BURN_HOTKEY in assigned:
        return
    node = await _boss_node(config)
    await task_sql.assign_node_to_task(str(seed.task_id), node, config.psql_db)
    expected_repo = f"tournament-{seed.tournament_id}-{seed.task_id}-{EMISSION_BURN_HOTKEY[:8]}"
    await task_sql.set_expected_repo_name(str(seed.task_id), node, config.psql_db, expected_repo)
    logger.info(f"Seed {seed.role.value} task {seed.task_id}: assigned the boss node, expected repo {expected_repo}")


# --- Drawing -------------------------------------------------------------------------------------------------------


async def _without_datasets(datasets: AsyncGenerator[Dataset, None], excluded: frozenset[str]) -> AsyncGenerator[Dataset, None]:
    """The dataset generator minus the given dataset ids. get_dataset pulls one dataset at a time with anext, so the
    filter simply never yields an excluded one (its own degenerate / known-near-duplicate / column checks still apply)."""
    async for dataset in datasets:
        if dataset.dataset_id not in excluded:
            yield dataset


def _rejected_datasets(history: list[TournamentSeedTask], tasks: dict[UUID, RawTask | None]) -> frozenset[str]:
    """Datasets of the tournament's prep-rejected dataset-drawn seed rows (any role: a dataset that failed prep for one
    seed is no better for another)."""
    return frozenset(
        tasks[row.task_id].ds
        for row in history
        if row.prep_rejected and row.role in DATASET_DRAWN_ROLES and tasks.get(row.task_id) is not None
    )


async def _excluded_datasets(tournament_id: str, extra: frozenset[str], config: Config) -> frozenset[str]:
    """Datasets this tournament's seeds must not draw again: the prep-rejected rows' (read from the history) plus extra
    (the row being replaced right now, which is only marked after its replacement is drawn)."""
    history = [row for row in await seeds_sql.get_seed_task_history(tournament_id, config.psql_db) if row.prep_rejected]
    tasks = {row.task_id: await task_sql.get_task(row.task_id, config.psql_db) for row in history}
    return _rejected_datasets(history, tasks) | extra


async def _draw_instruct(role: SeedRole, tournament_id: str, config: Config, rejected: frozenset[str] = frozenset()) -> RawTask:
    keypair = config.keypair
    # create_synthetic_instruct_text_task -> get_dataset also applies prod's known-bad history (degenerate loss, persisted
    # near-duplicate rate) and column validation to whatever this generator yields.
    datasets = _without_datasets(
        _get_instruct_text_datasets(keypair), await _excluded_datasets(tournament_id, rejected, config)
    )
    if role == SeedRole.BOSS_INSTRUCT:
        # The boss round's input (an ordinary synthetic instruct task, KL off) from the quant pool, not augmented: the
        # boss's output on it is compressed by the two chained compression seeds, which inherit its split at release. It
        # carves a calibration set like the input seeds, so those compression tasks publish one to both players.
        models = filter_quant_compatible(_get_text_models(keypair, *QUANT_MODEL_SIZE_B))
        return await create_synthetic_instruct_text_task(
            config, models, datasets, enable_kl=False, allow_augmentation=False, carve_calibration=True
        )
    # Input seeds: quant-compatible band, augmentation rolled, calibration carved.
    models = filter_quant_compatible(_get_text_models(keypair, *_draw_band(role)))
    return await create_synthetic_instruct_text_task(
        config, models, datasets, enable_kl=False, allow_augmentation=True, carve_calibration=True
    )


async def _draw_text_continuous(config: Config) -> RawTask:
    """The continuous-SFT chunk task of the text boss round. The crown rule counts exactly one continuous task, so the
    lineage table must hold exactly one lineage."""
    lineages = t_cst.CONTINUOUS_SFT_LINEAGES
    if len(lineages) != 1:
        raise ValueError(f"text_continuous needs exactly one continuous-SFT lineage, found {len(lineages)}: {sorted(lineages)}")
    lineage, seed_model = next(iter(lineages.items()))
    return await create_continuous_sft_task(config, lineage, seed_model)


async def _upstream_task(live: dict[SeedRole, TournamentSeedTask], config: Config) -> InstructTextRawTask | None:
    """The live BOSS_INSTRUCT task, the upstream of the compression seeds; None when it has none."""
    seed = live.get(SeedRole.BOSS_INSTRUCT)
    if seed is None:
        return None
    task = await task_sql.get_task(seed.task_id, config.psql_db)
    return task if isinstance(task, InstructTextRawTask) else None


async def _draw_seed_task(
    role: SeedRole,
    tournament_id: str,
    live: dict[SeedRole, TournamentSeedTask],
    config: Config,
    rejected: frozenset[str] = frozenset(),
) -> RawTask | None:
    """A fresh task for the role, or None when it cannot be drawn yet (a compression role without its instruct)."""
    if role in INSTRUCT_INPUT_SEED_ROLES or role in TEXT_ROUND_SEED_ROLES:
        return await _draw_instruct(role, tournament_id, config, rejected)
    if role == SeedRole.TEXT_CONTINUOUS:
        return await _draw_text_continuous(config)
    upstream_role = COMPRESSION_UPSTREAM_ROLE[role]
    upstream = await _upstream_task(live, config)
    if upstream is None:
        logger.warning(f"Tournament {tournament_id}: no live {upstream_role.value} seed; deferring {role.value}")
        return None
    task_type = CHAINED_SEED_TASK_TYPES[role]
    quant_format = _seed_quant_format(tournament_id, task_type) if task_type == TaskType.QUANTFIXEDTASK else None
    # Both players train this one task, so it takes the boss budget (the same hours for the challenger).
    return await create_chained_quant_task(config, task_type, upstream, quant_format=quant_format, boss=True)


async def _discard_unrecorded_task(task_id: UUID, config: Config) -> None:
    """Delete a just-created seed task that has no seed row, so it cannot be prepped and trained as an organic task."""
    try:
        await task_sql.delete_task(task_id, config.psql_db)
    except Exception:
        logger.error(f"Could not delete unrecorded seed task {task_id}; remove it by hand", exc_info=True)


async def _draw_compression_task(role: SeedRole, tournament_id: str, text_seed: TournamentSeedTask, config: Config) -> RawTask:
    """A new held quant task for a boss-round compression role, chained on the text seed task of its source role.

    The round-2 and round-3 compressions copy the text task's data at release (create_chained_quant_task); the continuous ones
    carry the same chunk as the text continuous task from creation, its index read from the text task's chunk record. Both
    players train the task, so it takes the boss budget. A fixed role plays a 2-bit or 3-bit format seeded by tournament and role.
    """
    spec = QUANT_SLOT_SPECS[QUANT_BOSS_SLOT_BY_ROLE[role]]
    upstream = await task_sql.get_task(text_seed.task_id, config.psql_db)
    if upstream is None:
        raise ValueError(f"text seed {text_seed.role.value} task {text_seed.task_id} is gone")
    quant_format = _boss_fixed_format(tournament_id, role) if spec.task_type == TaskType.QUANTFIXEDTASK else None
    if spec.text_role == SeedRole.TEXT_CONTINUOUS:
        if not isinstance(upstream, ChatRawTask):
            raise ValueError(f"text seed {text_seed.role.value} task {text_seed.task_id} is not a continuous-SFT chat task")
        chunk = await get_continuous_sft_task_chunk(upstream.task_id, config.psql_db)
        if chunk is None:
            raise ValueError(f"text seed {text_seed.role.value} task {text_seed.task_id} has no continuous chunk record")
        return await create_chained_continuous_quant_task(
            config,
            spec.task_type,
            upstream,
            quant_format=quant_format,
            train_index=chunk.train_index,
            tournament_id=tournament_id,
        )
    if not isinstance(upstream, InstructTextRawTask):
        raise ValueError(f"text seed {text_seed.role.value} task {text_seed.task_id} is not an instruct task")
    return await create_chained_quant_task(config, spec.task_type, upstream, quant_format=quant_format, boss=True)


async def _acquire_compression_task(
    role: SeedRole, tournament_id: str, attempt: int, replacing_task_id: UUID | None, config: Config
) -> RawTask | None:
    """The quant task of a boss-round compression role, with its source row recorded; None when it cannot be had now.

    The source row (quant_task_sources) is the link to the text seed task and holds the slot, so a task never exists without
    it and a slot never has two live tasks. A live source row whose task has no seed row yet (the process stopped between the
    two inserts), or whose seed row is this very attempt (the old attempt was not marked replaced), is reused instead of
    drawing another task. replacing_task_id is the task of the attempt being replaced: its
    source row is superseded together with the insert of the new one. Nothing is created while the quant tournament is not
    bound to a text tournament or the text role's replacement is pending.
    """
    binding = await quant_sources.bind_quant_source_tournament(tournament_id, config)
    if binding is None:
        return None
    slot = QUANT_BOSS_SLOT_BY_ROLE[role]
    live = await quant_sources_sql.get_live_quant_task_source(tournament_id, slot, config.psql_db)
    if live is not None and live.quant_task_id != replacing_task_id:
        recorded = await seeds_sql.get_seed_task_by_task_id(live.quant_task_id, config.psql_db)
        if recorded is not None and (recorded.role, recorded.attempt) != (role, attempt):
            logger.warning(
                f"Tournament {tournament_id}: slot {slot.value} already has seeded task {live.quant_task_id}; "
                f"not creating another"
            )
            return None
        held = await task_sql.get_task(live.quant_task_id, config.psql_db)
        if held is not None:
            logger.warning(
                f"Tournament {tournament_id}: reusing task {held.task_id} of slot {slot.value} for {role.value} attempt {attempt}"
            )
            return held
    text_seed = await quant_sources.slot_text_seed(binding, slot, config)
    if text_seed is None:
        logger.warning(
            f"Tournament {tournament_id}: text tournament {binding.source_tournament_id} has no live "
            f"{QUANT_SLOT_SPECS[slot].text_role.value} seed; deferring {role.value}"
        )
        return None
    task = await _draw_compression_task(role, tournament_id, text_seed, config)
    source = await quant_sources.record_new_quant_task(
        quant_sources.new_task_source(UUID(str(task.task_id)), binding, slot, text_seed), config, replacing_task_id
    )
    return task if source.quant_task_id == UUID(str(task.task_id)) else None


async def _create_seed(
    role: SeedRole,
    tournament_id: str,
    attempt: int,
    live: dict[SeedRole, TournamentSeedTask],
    config: Config,
    rejected: frozenset[str] = frozenset(),
    replacing_task_id: UUID | None = None,
) -> TournamentSeedTask | None:
    """Draw a task for the role, record the seed row straight after it (so the task is never mistaken for an organic
    one) and assign the boss. The boss's repo is looked up first, so its likeliest failure creates nothing; if the seed
    row cannot be inserted, the just-created task is deleted so no orphan is left to be prepped and trained.
    A quant compression role also records its source row with the task (_acquire_compression_task); a task that fails
    afterwards is deleted with it.
    rejected: datasets the draw must skip besides the tournament's recorded ones.
    replacing_task_id: the task of the attempt this one replaces (a compression role supersedes its source row)."""
    try:
        info = await _boss_training_info(tournament_id, config)
    except Exception as e:
        message = redact_url_queries(f"Tournament {tournament_id}: failed to read the boss's code for seed {role.value}: {e}")
        logger.error(message, exc_info=True)
        await alert_seed_issue(f"{tournament_id}:{role.value}:boss_info", f"{message}. It is retried every cycle.", config)
        return None
    try:
        if role in BOSS_ROUND_SEED_ROLES:
            task = await _acquire_compression_task(role, tournament_id, attempt, replacing_task_id, config)
        else:
            task = await _draw_seed_task(role, tournament_id, live, config, rejected)
    except Exception as e:
        message = redact_url_queries(f"Tournament {tournament_id}: failed to draw seed {role.value} attempt {attempt}: {e}")
        logger.error(message, exc_info=True)
        await alert_seed_issue(f"{tournament_id}:{role.value}:draw", f"{message}. It is retried every cycle.", config)
        return None
    if task is None:
        return None
    task_id = UUID(str(task.task_id))
    try:
        seed = await seeds_sql.insert_seed_task(
            TournamentSeedTask(
                tournament_id=tournament_id,
                role=role,
                attempt=attempt,
                task_id=task_id,
                status=SeedStatus.TRAINING,
                boss_repo=info.training_repo,
                boss_commit=info.training_commit_hash,
            ),
            config.psql_db,
        )
    except Exception as e:
        message = redact_url_queries(f"Tournament {tournament_id}: failed to record seed {role.value} attempt {attempt}: {e}")
        logger.error(message, exc_info=True)
        await _discard_unrecorded_task(task_id, config)
        await alert_seed_issue(f"{tournament_id}:{role.value}:insert", f"{message}. It is retried every cycle.", config)
        return None
    if seed.task_id != task_id:
        # Another pass recorded this attempt first: its task is the seed, this one is a duplicate.
        logger.warning(
            f"Tournament {tournament_id}: seed {role.value} attempt {attempt} is already task {seed.task_id}; deleting {task_id}"
        )
        await _discard_unrecorded_task(task_id, config)
        return None
    await _ensure_boss_assignment(seed, config)
    logger.info(
        f"Tournament {tournament_id}: seed {role.value} attempt {attempt} is task {seed.task_id} ({task.task_type.value}, "
        f"{task.model_id}), boss {seed.boss_repo}@{seed.boss_commit}"
    )
    return seed


async def create_seed_tasks(
    tournament_id: str, config: Config, tournament_type: TournamentType = TournamentType.QUANT
) -> list[TournamentSeedTask]:
    """Create the seed tasks of a tournament (its type's roles: seed_creation_order); a role that already has a live row
    is skipped (its boss assignment is re-checked). Returns the live row of every role."""
    order = seed_creation_order(tournament_type)
    live = await live_seed_tasks(tournament_id, config.psql_db)
    latest = {seed.role: seed for seed in await seeds_sql.get_latest_seed_tasks(tournament_id, config.psql_db)}
    for role in order:
        if role in live:
            await _ensure_boss_assignment(live[role], config)
            continue
        if role in latest:
            logger.warning(f"Tournament {tournament_id}: seed {role.value} is {latest[role].status.value}; not creating it")
            continue
        seed = await _create_seed(role, tournament_id, 1, live, config)
        if seed is not None:
            live[role] = seed
        elif role not in COMPRESSION_UPSTREAM_ROLE:
            logger.error(f"Tournament {tournament_id}: seed {role.value} could not be drawn; retrying next cycle")
    logger.info(f"Tournament {tournament_id}: {len(live)}/{len(order)} seed tasks live")
    return [live[role] for role in order if role in live]


# --- Status, replacement, halt -------------------------------------------------------------------------------------


async def _boss_output_rejection(task_id: UUID, config: Config) -> str | None:
    """Why the boss's output on an instruct seed cannot be an input, or None. A hub error other than a missing repo
    raises, so a transient failure waits for the next cycle instead of replacing a good seed."""
    repo = await chaining.chained_input_repo(task_id, EMISSION_BURN_HOTKEY, config.psql_db)
    if repo is None:
        return "no validator-assigned boss repo"
    try:
        rejection = await full_model_rejection(repo)
    except RepositoryNotFoundError:
        return f"{repo} not found"
    return f"{repo} is not a full model: {rejection}" if rejection else None


async def boss_output_awaits_merge(seed: TournamentSeedTask, config: Config) -> bool:
    """Whether the boss's raw output on an instruct seed is a LoRA adapter whose merge into a full model is not recorded yet.

    Such a seed waits for its merge (process_seed_adapter_merges); it is never replaced or charged for the adapter. A
    missing repo is not this case (a real defect: the seed is replaced). Hub errors other than a missing repo propagate.
    """
    if seed.merged_output_repo:
        return False
    repo = await chaining.chained_input_repo(seed.task_id, EMISSION_BURN_HOTKEY, config.psql_db)
    if repo is None:
        return False
    try:
        return await is_adapter_repo(repo)
    except RepositoryNotFoundError:
        return False


async def _boss_row_status(seed: TournamentSeedTask, config: Config) -> str | None:
    statuses = await tournament_sql.get_training_status_for_task_and_hotkeys(
        str(seed.task_id), [EMISSION_BURN_HOTKEY], config.psql_db
    )
    return statuses.get(EMISSION_BURN_HOTKEY)


async def _evaluate_seed(seed: TournamentSeedTask, task: RawTask | None, config: Config) -> tuple[SeedStatus | None, str | None]:
    """(new status, None) to move the row, (None, reason) to replace it, (None, None) to wait."""
    if task is None:
        return None, "task is gone"
    if seed.role in TEXT_SEED_ROLES:
        return await _evaluate_text_seed(seed, task, config)
    if seed.role in ANY_BOSS_COMPRESSION_SEED_ROLES and task.status != TaskStatus.PREP_TASK_FAILURE:
        if await _boss_row_status(seed, config) == TrainingStatus.FAILURE.value:
            # The boss's code cannot do this compression (e.g. a boss with no quant code): a redraw cannot change that and
            # only burns GPUs, and the cap would halt the tournament. This failed run is the boss's entry, and in the boss
            # round the challenger takes the task if its own run succeeds.
            if seed.status != SeedStatus.TRAINING:
                return None, None
            logger.warning(
                f"Tournament {seed.tournament_id}: the boss's {seed.role.label} entry (task {seed.task_id}) failed training; "
                f"recorded as the boss's failed entry, not redrawn"
            )
            return SeedStatus.TRAINED, None
    if task.status in _DEAD_TASK_STATUSES:
        return None, f"task ended {task.status}"
    if seed.role in ANY_BOSS_ROUND_SEED_ROLES and task.status == TaskStatus.SUCCESS:
        # Scored through the boss round (an attached seed whose boss row failed is a scored loss, not a dead seed).
        return SeedStatus.READY, None
    boss_status = await _boss_row_status(seed, config)
    if boss_status == TrainingStatus.FAILURE.value:
        return None, f"the boss's run failed after {t_cst.MAX_TRAINING_ATTEMPTS} attempts"

    if seed.role in INSTRUCT_INPUT_SEED_ROLES:
        if task.status != TaskStatus.SUCCESS:
            return None, None
        usable = await chaining.shared_input_candidates(seed.task_id, [EMISSION_BURN_HOTKEY], config.psql_db)
        if EMISSION_BURN_HOTKEY not in usable:
            return None, "the boss's output scored no fine-tune"
        if await boss_output_awaits_merge(seed, config):
            return None, None
        rejection = await _boss_output_rejection(seed.task_id, config)
        return (None, rejection) if rejection else (SeedStatus.READY, None)

    if task.status == TaskStatus.SUCCESS:
        return SeedStatus.READY, None
    if boss_status == TrainingStatus.SUCCESS.value and seed.status == SeedStatus.TRAINING:
        return SeedStatus.TRAINED, None
    return None, None


async def _charged_replacements_remain(seed: TournamentSeedTask, config: Config) -> bool:
    """Whether the role may still be replaced for a charged failure (replace_seed marks the role failed past the cap)."""
    history = await seeds_sql.get_seed_task_history(seed.tournament_id, config.psql_db)
    return _earlier_replacements(history, seed.role, seed.attempt, False) < t_cst.SEED_MAX_REPLACEMENTS


async def _evaluate_text_seed(seed: TournamentSeedTask, task: RawTask, config: Config) -> tuple[SeedStatus | None, str | None]:
    """_evaluate_seed for a text seed (a round's instruct task or the continuous-SFT task).

    The boss trains it at tournament start and the round's miners train it when the round attaches it, so the task is not
    SUCCESS until that round is over. A seed that no round has attached is replaced on any failure. Once attached it is
    only replaced for a prep failure (nobody has trained); a failed task is left to the round's own completion check. A
    failed boss run on an attached seed (or an unattached one out of replacements) is the boss's failed entry: the task
    stays the round's task and the boss scores nothing on it. A text seed that runs out of replacements is marked `failed` by
    replace_seed; that never halts the tournament (the round or the boss round goes on without the seed).
    """
    attached = await _was_attached(seed, config.psql_db)
    if task.status == TaskStatus.PREP_TASK_FAILURE:
        return None, f"task ended {task.status}"
    if task.status == TaskStatus.FAILURE:
        return (None, None) if attached else (None, f"task ended {task.status}")
    boss_status = await _boss_row_status(seed, config)
    if boss_status == TrainingStatus.FAILURE.value:
        if not attached and await _charged_replacements_remain(seed, config):
            return None, f"the boss's run failed after {t_cst.MAX_TRAINING_ATTEMPTS} attempts"
        if seed.status == SeedStatus.TRAINING:
            logger.warning(
                f"Tournament {seed.tournament_id}: the boss's {seed.role.label} run (task {seed.task_id}) failed training; "
                f"recorded as the boss's failed entry, not redrawn"
            )
            return SeedStatus.TRAINED, None
    elif boss_status == TrainingStatus.SUCCESS.value and seed.status == SeedStatus.TRAINING:
        return SeedStatus.TRAINED, None
    if task.status == TaskStatus.SUCCESS:
        return SeedStatus.READY, None
    return None, None


async def repoint_task_onto_seed(task, old_task: RawTask, new_task: RawTask, config: Config) -> bool:
    """Move a held chained task off a replaced seed task onto its replacement: the task itself (upstream, base, size, hours)
    and, for a quant task of a bound quant tournament, its quant_task_sources row. The source row moves first, so a
    stop in between is repaired by the release path (which finds the task still on the replaced seed and calls this again).

    A task on its own continuous chunk keeps its data: only the upstream moves, and only when the replacement trained on the
    same chunk (the replaced continuous seed is the same lineage's next chunk, so it is). Another chunk is refused: the task
    stays where it is, and an alert says so. Returns whether the task moved.
    """
    train_index = None
    if isinstance(new_task, ChatRawTask):
        chunk = await get_continuous_sft_task_chunk(new_task.task_id, config.psql_db)
        train_index = chunk.train_index if chunk else None
    try:
        repoint_chained_quant_task(task, new_task, pinned_data=task.ds == old_task.ds, upstream_train_index=train_index)
    except ValueError as e:
        await alert_seed_issue(
            f"{task.task_id}:repoint-chunk-mismatch", f"Held task {task.task_id} was not re-pointed: {e}", config
        )
        return False
    await quant_sources_sql.repoint_quant_task_source(task.task_id, new_task.task_id, config.psql_db)
    await task_sql.update_task(task, config.psql_db)
    logger.info(f"Re-pointed held task {task.task_id} from seed {old_task.task_id} to {new_task.task_id}")
    return True


async def _repoint_dependents(old_task: RawTask, new_task: RawTask, config: Config, tournament_id: str | None = None) -> None:
    """Move every held chained task (a round task chained on an input or text seed, a quant task of another tournament chained on
    a text seed) off the replaced seed onto the new one.
    The compression seeds of the seed's own tournament are not repointed: a replaced boss_instruct re-creates them (with a
    tournament_id, a held task that is a seed row of another tournament is repointed like any other)."""
    held = await task_sql.get_tasks_with_status(
        TaskStatus.AWAITING_UPSTREAM, config.psql_db, tournament_filter="all", include_not_ready_tasks=True
    )
    for task in held:
        if not chaining.is_chained(task) or str(task.upstream_task_id) != str(old_task.task_id):
            continue
        dependent_seed = await seeds_sql.get_seed_task_by_task_id(task.task_id, config.psql_db)
        if dependent_seed is not None and (tournament_id is None or dependent_seed.tournament_id == tournament_id):
            continue
        await repoint_task_onto_seed(task, old_task, new_task, config)


async def _alert_halt(tournament_id: str, reason: str, config: Config) -> None:
    message = f"Tournament {tournament_id} HALTED: {reason}. A human decides (replace the seed or abort)."
    logger.error(message)
    if tournament_id in _SEED_HALT_ALERTED or not config.discord_url:
        return
    _SEED_HALT_ALERTED.add(tournament_id)
    try:
        await send_to_discord(webhook=config.discord_url, message=message)
    except Exception as e:
        logger.error(f"Failed to send Discord notification: {e}")


async def alert_seed_issue(dedupe_key: str, message: str, config: Config) -> None:
    """A non-halting seed alert: logged loudly every time, sent to Discord once per key per process."""
    logger.error(message)
    if dedupe_key in _SEED_ISSUE_ALERTED or not config.discord_url:
        return
    _SEED_ISSUE_ALERTED.add(dedupe_key)
    try:
        await send_to_discord(webhook=config.discord_url, message=message)
    except Exception as e:
        logger.error(f"Failed to send Discord notification: {e}")


async def _alert_role_out_of_replacements(tournament_id: str, role: SeedRole, detail: str, config: Config) -> None:
    """A role ran out of replacements and its row is `failed`. That halts the tournament only for an input-seed role.
    A quant boss-round compression role and a text role neither halt nor block anything, so each is one alert per tournament
    and role."""
    if role in TEXT_SEED_ROLES:
        if role == SeedRole.TEXT_CONTINUOUS:
            effect = "the boss round draws its own continuous task, which both players train there (no upfront boss result)"
        else:
            effect = "the round plays a task the boss has no result on, which the challenger takes with a valid score"
        await alert_seed_issue(
            f"{tournament_id}:{role.value}:out-of-replacements",
            f"Tournament {tournament_id}: seed {role.value} {detail}. The tournament is not halted; {effect}. Quant tasks on "
            f"this role fall back to the base model unless the boss's model on the role's last task had landed.",
            config,
        )
        return
    if role not in BOSS_ROUND_SEED_ROLES:
        await _alert_halt(tournament_id, f"seed {role.value} {detail}", config)
        return
    await alert_seed_issue(
        f"{tournament_id}:{role.value}:compression-out-of-replacements",
        f"Tournament {tournament_id}: seed {role.value} {detail}. The tournament is not halted; the boss round goes on without "
        f"that task and the challenger is not credited with it.",
        config,
    )


async def _was_attached(seed: TournamentSeedTask, psql_db: PSQLDB) -> bool:
    """Whether a boss-round seed was ever attached to the boss round: the durable marker, or a live tournament_tasks row."""
    return seed.attached_round_id is not None or await seeds_sql.is_task_attached_to_round(seed.task_id, psql_db)


def boss_side_failure_key(tournament_id: str, task_id: UUID | str) -> str:
    """The one Discord dedupe key of a boss-round task's boss-side failure, whichever code path reports it first."""
    return f"{tournament_id}:{task_id}:boss-side-failure"


async def _replacement_blocked(seed: TournamentSeedTask, reason: str, config: Config, prep_failed: bool = False) -> bool:
    """Whether a seed must not be replaced: it was attached to its round (the round resolves the seed; a boss-round
    draw now would train the boss after R3). A seed the round has not attached yet may still be replaced, even
    when the round row exists: a role without an attached task is exactly what leaves that round PENDING. An attached
    text seed is replaced only for a prep failure (nobody has trained, the replacement takes over its place)."""
    if seed.role in TEXT_SEED_ROLES:
        if prep_failed or not await _was_attached(seed, config.psql_db):
            return False
        await alert_seed_issue(
            f"{seed.tournament_id}:{seed.task_id}:text-seed-attached-failure",
            f"Tournament {seed.tournament_id}: text seed {seed.role.value} task {seed.task_id} failed ({reason}) but is not "
            f"replaced (it was attached to its round); the round's completion check reports it",
            config,
        )
        return True
    if seed.role not in ANY_BOSS_ROUND_SEED_ROLES or not await _was_attached(seed, config.psql_db):
        return False
    await alert_seed_issue(
        boss_side_failure_key(seed.tournament_id, seed.task_id),
        f"Tournament {seed.tournament_id}: boss-round seed {seed.role.value} task {seed.task_id} failed ({reason}) but is not "
        f"replaced (was attached to the boss round); the boss round resolves it as a boss-side failure",
        config,
    )
    return True


def _earlier_replacements(history: list[TournamentSeedTask], role: SeedRole, before_attempt: int, prep_rejected: bool) -> int:
    """How many attempts of the role before before_attempt were replaced, charged (prep_rejected False) or uncharged."""
    return sum(
        1
        for row in history
        if row.role == role
        and row.attempt < before_attempt
        and row.status == SeedStatus.REPLACED
        and row.prep_rejected == prep_rejected
    )


def is_prep_rejection(seed: TournamentSeedTask, task: RawTask | None) -> bool:
    """Whether the seed's task died in the prep stage on a dataset a redraw can replace: a dataset-drawn role (an
    instruct task, so PREP_TASK_FAILURE is only ever set before training: near-duplicate rejection at the
    orchestrator's prep / model-prep checks, or an error in prep) that ended PREP_TASK_FAILURE. The quant-only
    PREP_TASK_FAILURE of the evaluation path (QuantReferenceUnavailable) cannot hit these roles. The chained
    compressions are never prep rejections here (see DATASET_DRAWN_ROLES)."""
    return seed.role in DATASET_DRAWN_ROLES and task is not None and task.status == TaskStatus.PREP_TASK_FAILURE


async def _swap_attached_seed(
    old_seed: TournamentSeedTask, new_seed: TournamentSeedTask, new_task: RawTask, config: Config
) -> None:
    """Give an attached text seed's place in its round to its replacement and mark the replacement attached."""
    round_id = await swap_seed_task_registration(str(old_seed.task_id), new_task, old_seed.tournament_id, config)
    await seeds_sql.mark_seed_attached(new_seed.task_id, round_id or old_seed.attached_round_id, config.psql_db)


async def _resume_interrupted_swaps(tournament_id: str, config: Config) -> None:
    """Finish the swap of a text seed whose replacement was drawn but not yet put in the round (the process stopped between
    the two steps): the earlier attempt is still registered in its round. A finished swap always detaches the earlier attempt,
    so that holds whatever its row says; a row already marked replaced (a round that registered the live seed just as the seed
    loop replaced it) is left as it is."""
    history = await seeds_sql.get_seed_task_history(tournament_id, config.psql_db)
    for seed in await seeds_sql.get_active_seed_tasks(tournament_id, config.psql_db):
        if seed.role not in TEXT_SEED_ROLES or seed.attached_round_id is not None:
            continue
        previous = next((row for row in history if row.role == seed.role and row.attempt == seed.attempt - 1), None)
        if previous is None:
            continue
        if not await seeds_sql.is_task_attached_to_round(previous.task_id, config.psql_db):
            continue
        task = await task_sql.get_task(seed.task_id, config.psql_db)
        if task is None:
            continue
        logger.warning(f"Tournament {tournament_id}: finishing the interrupted swap of {seed.role.value} to task {seed.task_id}")
        await _swap_attached_seed(previous, seed, task, config)
        if previous.status != SeedStatus.REPLACED:
            await seeds_sql.mark_seed_replaced(tournament_id, previous.role, previous.attempt, False, config.psql_db)


async def replace_seed(
    seed: TournamentSeedTask, reason: str, config: Config, prep_rejected: bool = False
) -> TournamentSeedTask | None:
    """Replace a failed seed with a fresh draw of its role (attempt + 1) and repoint what depended on it.

    prep_rejected (is_prep_rejection) marks a prep-stage dataset rejection: it does not consume SEED_MAX_REPLACEMENTS, the
    redraw skips every dataset this tournament already had rejected, and the role may do this SEED_MAX_PREP_REJECTIONS - 1
    times; the SEED_MAX_PREP_REJECTIONS-th marks the row `failed` and alerts once. Any other failure counts against
    SEED_MAX_REPLACEMENTS (attempts keep counting up for the primary key, the cap counts only the charged replaced rows);
    past it the row is `failed` and a Discord alert goes out. An input-seed role that fails halts the tournament
    (a human decides); a text or quant boss-round role never does (_alert_role_out_of_replacements). A replaced
    boss_instruct also re-creates both compression seeds on the new instruct task, uncharged to them (they were never
    trained). Returns the new row, or None.
    """
    tournament_id = seed.tournament_id
    old_task = await task_sql.get_task(seed.task_id, config.psql_db)
    prep_failed = old_task is not None and old_task.status == TaskStatus.PREP_TASK_FAILURE
    if await _replacement_blocked(seed, reason, config, prep_failed):
        return None
    history = await seeds_sql.get_seed_task_history(tournament_id, config.psql_db)
    if prep_rejected:
        rejections = _earlier_replacements(history, seed.role, seed.attempt, True) + 1
        if rejections >= t_cst.SEED_MAX_PREP_REJECTIONS:
            await seeds_sql.update_seed_status(tournament_id, seed.role, seed.attempt, SeedStatus.FAILED, config.psql_db)
            if seed.role in TEXT_SEED_ROLES:
                limit = t_cst.SEED_MAX_PREP_REJECTIONS
                detail = f"had {rejections} prep-stage dataset rejections (limit {limit}; last: {reason})"
                await _alert_role_out_of_replacements(tournament_id, seed.role, detail, config)
                return None
            message = (
                f"Tournament {tournament_id} HALTED: seed {seed.role.value} had {rejections} prep-stage dataset rejections "
                f"(limit {t_cst.SEED_MAX_PREP_REJECTIONS}; last: {reason}). A human decides (replace the seed or abort)."
            )
            await alert_seed_issue(f"{tournament_id}:{seed.role.value}:prep-rejection-limit", message, config)
            _SEED_HALT_ALERTED.add(tournament_id)  # this message is the halt alert; the generic one would repeat it
            return None
    elif _earlier_replacements(history, seed.role, seed.attempt, False) >= t_cst.SEED_MAX_REPLACEMENTS:
        await seeds_sql.update_seed_status(tournament_id, seed.role, seed.attempt, SeedStatus.FAILED, config.psql_db)
        detail = f"failed {seed.attempt} times (last: {reason})"
        await _alert_role_out_of_replacements(tournament_id, seed.role, detail, config)
        return None

    logger.warning(
        f"Tournament {tournament_id}: replacing seed {seed.role.value} attempt {seed.attempt} (task {seed.task_id}): {reason}"
    )
    live = await live_seed_tasks(tournament_id, config.psql_db)
    live.pop(seed.role, None)
    rejected = frozenset({old_task.ds}) if prep_rejected and old_task is not None else frozenset()
    new_seed = await _create_seed(
        seed.role, tournament_id, seed.attempt + 1, live, config, rejected, replacing_task_id=seed.task_id
    )
    if new_seed is None:
        message = f"Tournament {tournament_id}: could not draw a replacement for {seed.role.value}; retrying next cycle"
        await alert_seed_issue(f"{tournament_id}:{seed.role.value}:replace-draw", message, config)
        return None
    new_task = await task_sql.get_task(new_seed.task_id, config.psql_db)
    if seed.role in TEXT_SEED_ROLES and await _was_attached(seed, config.psql_db) and new_task is not None:
        # An attached text seed (a prep failure): the replacement takes over the round's place and the assigned nodes.
        await _swap_attached_seed(seed, new_seed, new_task, config)
    await seeds_sql.mark_seed_replaced(tournament_id, seed.role, seed.attempt, prep_rejected, config.psql_db)

    if old_task is not None and old_task.status == TaskStatus.AWAITING_UPSTREAM:
        old_task.status = TaskStatus.FAILURE
        await task_sql.update_task(old_task, config.psql_db)
    if old_task is not None and new_task is not None and (seed.role in INPUT_SEED_ROLES or seed.role in TEXT_SEED_ROLES):
        await _repoint_dependents(old_task, new_task, config, tournament_id)

    if seed.role == SeedRole.BOSS_INSTRUCT:
        live[seed.role] = new_seed
        for chained_role in (role for role, upstream_role in COMPRESSION_UPSTREAM_ROLE.items() if upstream_role == seed.role):
            old_chained = next(
                (row for row in await seeds_sql.get_latest_seed_tasks(tournament_id, config.psql_db) if row.role == chained_role),
                None,
            )
            if old_chained is None:
                continue
            # The old compression seed points at the replaced instruct and can never be released: retire it (and its held
            # task) first, so that if the re-draw fails the role simply has no live row and is retried later
            # (process_tournament_seed_tasks) instead of waiting forever on the dead instruct. The retirement is uncharged
            # (prep_rejected=True) whatever failed upstream: the chained seed never trained, so its own cap is untouched.
            await seeds_sql.mark_seed_replaced(tournament_id, chained_role, old_chained.attempt, True, config.psql_db)
            chained_task = await task_sql.get_task(old_chained.task_id, config.psql_db)
            if chained_task is not None and chained_task.status == TaskStatus.AWAITING_UPSTREAM:
                chained_task.status = TaskStatus.FAILURE
                await task_sql.update_task(chained_task, config.psql_db)
            live.pop(chained_role, None)
            replacement = await _create_seed(chained_role, tournament_id, old_chained.attempt + 1, live, config)
            if replacement is None:
                logger.error(f"Tournament {tournament_id}: could not re-create {chained_role.value}; retrying next cycle")
                continue
            live[chained_role] = replacement
            logger.warning(f"Tournament {tournament_id}: re-created {chained_role.value} on the new {seed.role.value} seed")
    return new_seed


async def _retry_missing_boss_round_seeds(tournament_id: str, config: Config) -> None:
    """Re-create a boss-round role that has no live row (its re-draw failed earlier), until the boss round attached
    one of the role's seeds. A boss round row without that attachment does not stop it: such a round is PENDING
    and raises on the missing role until the retry succeeds. Respects SEED_MAX_REPLACEMENTS like replace_seed."""
    live = await live_seed_tasks(tournament_id, config.psql_db)
    latest = {row.role: row for row in await seeds_sql.get_latest_seed_tasks(tournament_id, config.psql_db)}
    # A quant tournament holding rows of the BOSS_INSTRUCT compression roles re-creates only those two roles.
    roles = tuple(CHAINED_SEED_TASK_TYPES) if latest.keys() & set(CHAINED_SEED_TASK_TYPES) else SEED_CREATION_ORDER
    for role in roles:
        if role in live:
            continue
        previous = latest.get(role)
        if previous is not None and previous.status != SeedStatus.REPLACED:
            continue  # failed: out of replacements, the role stays without a task
        if await seeds_sql.role_ever_attached(tournament_id, role, config.psql_db):
            continue  # nothing boss-side trains after the boss round took the role
        history = await seeds_sql.get_seed_task_history(tournament_id, config.psql_db)
        if previous is not None and _earlier_replacements(history, role, previous.attempt, False) >= t_cst.SEED_MAX_REPLACEMENTS:
            await seeds_sql.update_seed_status(tournament_id, role, previous.attempt, SeedStatus.FAILED, config.psql_db)
            detail = f"failed {previous.attempt} times (no replacement left)"
            await _alert_role_out_of_replacements(tournament_id, role, detail, config)
            continue
        try:
            seed = await _create_seed(role, tournament_id, previous.attempt + 1 if previous else 1, live, config)
        except Exception as e:
            logger.error(f"Tournament {tournament_id}: error re-creating seed {role.value}: {e}", exc_info=True)
            continue
        if seed is not None:
            live[role] = seed
            logger.warning(f"Tournament {tournament_id}: re-created missing seed {role.value} (attempt {seed.attempt})")


async def process_tournament_seed_tasks(tournament_id: str, config: Config) -> None:
    """Advance the status of one tournament's seeds, replacing the ones that failed.

    A tournament with no seed rows at all started before the seed design (create_seed_tasks writes the input seeds when
    the tournament's first round is created and replaced rows stay in the history, so a seeded tournament always has rows): it is
    left alone, because creating boss-round seeds for it would train the boss for rounds that never chain on them.
    """
    history = await seeds_sql.get_seed_task_history(tournament_id, config.psql_db)
    if not history:
        if tournament_id not in _UNSEEDED_LOGGED:
            _UNSEEDED_LOGGED.add(tournament_id)
            logger.info(f"Tournament {tournament_id} has no seed tasks (it started before seeding existed); not seeding it")
        return
    is_text = any(row.role in TEXT_SEED_ROLES for row in history)
    if is_text:
        try:
            await _resume_interrupted_swaps(tournament_id, config)
        except Exception as e:
            logger.error(f"Tournament {tournament_id}: error resuming a seed swap: {e}", exc_info=True)
    for seed in await seeds_sql.get_active_seed_tasks(tournament_id, config.psql_db):
        if seed.status == SeedStatus.READY:
            continue
        try:
            task = await task_sql.get_task(seed.task_id, config.psql_db)
            new_status, failure = await _evaluate_seed(seed, task, config)
            if new_status is not None:
                await seeds_sql.update_seed_status(tournament_id, seed.role, seed.attempt, new_status, config.psql_db)
            elif failure is not None:
                # replace_seed refuses a boss-round seed that was ever attached (or once the boss round exists): the
                # round resolves it as a boss-side failure, and a replacement would train the boss after R3.
                await replace_seed(seed, failure, config, prep_rejected=is_prep_rejection(seed, task))
        except Exception as e:
            logger.error(f"Tournament {tournament_id}: error processing seed {seed.role.value}: {e}", exc_info=True)
    if not is_text:
        await _retry_missing_boss_round_seeds(tournament_id, config)
    reason = await seed_halt_reason(tournament_id, config.psql_db)
    if reason:
        await _alert_halt(tournament_id, reason, config)


async def process_seed_tasks(config: Config) -> None:
    """One pass over the seeds of every active quant and text tournament (a text tournament without seeds is left alone)."""
    for tournament in await tournament_sql.get_tournaments_with_status(TournamentStatus.ACTIVE, config.psql_db):
        if tournament.tournament_type in (TournamentType.QUANT, TournamentType.TEXT):
            await process_tournament_seed_tasks(tournament.tournament_id, config)


# --- Trainer priority ----------------------------------------------------------------------------------------------

# Lower ranks are tried first in a scheduling cycle. A running job is never preempted.
INPUT_SMALL_SEED_TRAINING_RANK = 0  # the small input seed
INPUT_LARGE_SEED_TRAINING_RANK = 1  # the large input seed
NON_SEED_TRAINING_RANK = 2  # miner-round rows (R1-R3) and every non-seed task
BOSS_INSTRUCT_SEED_TRAINING_RANK = 3  # the boss's entry on the boss round's input
DEFERRABLE_BOSS_SEED_TRAINING_RANK = 4  # the boss's two compression entries
# Rank comes from the task's role alone, the same for both players: the challenger's row on a task ties with the boss's row.
BOSS_ENTRY_TRAINING_RANKS = (BOSS_INSTRUCT_SEED_TRAINING_RANK, DEFERRABLE_BOSS_SEED_TRAINING_RANK)
# A row of a lower rank than this short of GPUs makes the boss entries (ranks 3-4) yield for the cycle.
FIRST_BOSS_ENTRY_TRAINING_RANK = BOSS_INSTRUCT_SEED_TRAINING_RANK


def training_rank(role: SeedRole | None) -> int:
    """Where a training row sits in a scheduling cycle (lower first): the two input seeds, every other row (miner rounds,
    non-seed tasks), then the boss round's input and its two compression tasks. The rank is the task's, the same for the
    boss's and the challenger's row. Only the input seeds are never deferred; the boss's own entries (see is_boss_entry_row)
    yield in a cycle where a row of rank below FIRST_BOSS_ENTRY_TRAINING_RANK is short of GPUs."""
    if role == SeedRole.INPUT_SMALL:
        return INPUT_SMALL_SEED_TRAINING_RANK
    if role == SeedRole.INPUT_LARGE:
        return INPUT_LARGE_SEED_TRAINING_RANK
    if role == SeedRole.BOSS_INSTRUCT:
        return BOSS_INSTRUCT_SEED_TRAINING_RANK
    if role in ANY_BOSS_ROUND_SEED_ROLES:
        return DEFERRABLE_BOSS_SEED_TRAINING_RANK
    return NON_SEED_TRAINING_RANK


def is_boss_entry_row(row_rank: int, hotkey: str) -> bool:
    """Whether a training row (its task's rank and hotkey) is one of the boss's own entries (the boss input or a compression):
    the only rows that yield to a starved miner row. The challenger's row on the same task ties in rank but never yields."""
    return row_rank in BOSS_ENTRY_TRAINING_RANKS and hotkey == EMISSION_BURN_HOTKEY


async def training_ranks(task_ids: list[str], psql_db: PSQLDB) -> dict[str, int]:
    """training_rank of each given task (non-seed tasks rank NON_SEED_TRAINING_RANK): one query for the whole cycle."""
    roles = await seeds_sql.get_seed_roles_by_task_ids(task_ids, psql_db)
    return {task_id: training_rank(roles.get(task_id)) for task_id in task_ids}
