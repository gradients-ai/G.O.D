"""Chained quant tasks: quant tasks fed by an upstream instruct task (`upstream_task_id`).

Chained quant tasks are on a seed the boss trained:
  * The quant seeds of rounds 1-3 (QUANT_ROUND_PLAN) and of the quant boss round: tasks chained on a seed of the
    bound TEXT tournament (text_r1/text_r2/text_r3/text_continuous). The text boss's model on the seed is the one shared input
    (the repo, or the text task's base model when the text boss produced none); every player compresses it. They are released
    by orchestrator._release_on_text_boss_model, not by the rules below.
  * Quant tasks chained on a BOSS_INSTRUCT seed are released once the seed task is SUCCESS. The only shared-input candidate
    is the boss's output on the seed.

A chained task is held in AWAITING_UPSTREAM until it can be released and is then an ordinary quant task. These helpers
only read; releasing held tasks is the orchestrator's job.

The lower-loss selection (order_by_upstream_loss, shared_input_candidates) serves shared-input boss tasks that are not on a
seed; on a seed the candidate list is just the boss.
"""

import math
from uuid import UUID

from core.constants.credentials import RAYONLABS_HF_USERNAME
from core.models.task_models import QUANT_TASK_TYPES
from validator.db import constants as db_cst
from validator.db.database import PSQLDB
from validator.db.sql import submissions_and_scoring as submissions_and_scoring_sql
from validator.db.sql import tasks as task_sql
from validator.db.sql import tournament_seeds as seeds_sql
from validator.db.sql import tournaments as tournament_sql
from validator.scoring.constants import EMISSION_BURN_HOTKEY
from validator.tournament.models import TrainingStatus


def is_chained(task) -> bool:
    """A quant task fed by an upstream instruct task (a miner round on an input seed, or a boss-round compression seed)."""
    return task.task_type in QUANT_TASK_TYPES and task.upstream_task_id is not None


async def chained_input_repo(upstream_task_id: UUID, hotkey: str, psql_db: PSQLDB) -> str | None:
    """The repo the trainer uploaded for hotkey on the upstream task, or None when it has none.

    Always the validator-assigned expected repo, never a miner-supplied id, so a player cannot choose
    which model is compressed. When the boss's output on an input seed was a LoRA adapter, its output is the full model
    the adapter was merged into (the seed's merged_output_repo).
    """
    if hotkey == EMISSION_BURN_HOTKEY:
        seed = await seeds_sql.get_seed_task_by_task_id(upstream_task_id, psql_db)
        if seed is not None and seed.merged_output_repo:
            return seed.merged_output_repo
    expected_repo_name = await task_sql.get_expected_repo_name(upstream_task_id, hotkey, psql_db)
    return f"{RAYONLABS_HF_USERNAME}/{expected_repo_name}" if expected_repo_name else None


def is_upstream_eligible(training_status: str | None, is_eligible_score: bool, test_loss: float | None) -> bool:
    """A player's upstream run counts when it trained, scored without the penalty and has a finite loss."""
    return (
        training_status == TrainingStatus.SUCCESS.value
        and is_eligible_score
        and test_loss is not None
        and math.isfinite(test_loss)
    )


def order_by_upstream_loss(losses: dict[str, float]) -> list[str]:
    """Hotkeys by upstream test loss, lowest first; an exact tie goes to the challenger.

    Shared-input boss tasks not on a seed only; an input seed has the boss as its one candidate.
    """
    return sorted(losses, key=lambda hotkey: (losses[hotkey], hotkey == EMISSION_BURN_HOTKEY, hotkey))


async def shared_input_candidates(upstream_task_id: UUID, hotkeys: list[str], psql_db: PSQLDB) -> list[str]:
    """The players whose upstream run is usable, in the order their output is tried as the shared input."""
    statuses = await tournament_sql.get_training_status_for_task_and_hotkeys(str(upstream_task_id), hotkeys, psql_db)
    eligible = await submissions_and_scoring_sql.get_eligible_hotkeys_for_task(upstream_task_id, psql_db)
    losses = {
        row[db_cst.HOTKEY]: row[db_cst.TEST_LOSS]
        for row in await submissions_and_scoring_sql.get_task_node_losses(upstream_task_id, psql_db)
    }
    usable = {
        hotkey: losses[hotkey]
        for hotkey in hotkeys
        if is_upstream_eligible(statuses.get(hotkey), hotkey in eligible, losses.get(hotkey))
    }
    return order_by_upstream_loss(usable)
