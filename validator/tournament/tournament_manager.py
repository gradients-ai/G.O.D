import asyncio
import math
import random
from datetime import datetime
from datetime import timedelta
from datetime import timezone

from fiber.chain.models import Node

import validator.core.constants as cst
from core.models.payload_models import TrainingRepoResponse
from core.models.tournament_models import Group
from core.models.tournament_models import GroupRound
from core.models.tournament_models import KnockoutRound
from core.models.tournament_models import RespondingNode
from core.models.tournament_models import Round
from core.models.tournament_models import RoundStatus
from core.models.tournament_models import RoundType
from core.models.tournament_models import TournamentData
from core.models.tournament_models import TournamentParticipant
from core.models.tournament_models import TournamentRoundData
from core.models.tournament_models import TournamentStatus
from core.models.tournament_models import TournamentTask
from core.models.tournament_models import TournamentType
from core.models.tournament_models import generate_round_id
from core.models.tournament_models import generate_tournament_id
from core.models.utility_models import TaskStatus
from validator.core.config import Config
from validator.core.constants import EMISSION_BURN_HOTKEY
from validator.core.models import AnyTypeTask
from validator.db.database import PSQLDB
from validator.db.sql import tasks as task_sql
from validator.db.sql.nodes import get_all_nodes
from validator.db.sql.nodes import get_node_by_hotkey
from validator.db.sql.tournaments import add_tournament_participants
from validator.db.sql.tournaments import create_tournament
from validator.db.sql.tournaments import eliminate_tournament_participants
from validator.db.sql.tournaments import get_active_tournament
from validator.db.sql.tournaments import get_latest_tournament_with_created_at
from validator.db.sql.tournaments import get_tournament
from validator.db.sql.tournaments import get_tournament_group_members
from validator.db.sql.tournaments import get_tournament_groups
from validator.db.sql.tournaments import get_tournament_pairs
from validator.db.sql.tournaments import get_tournament_participant
from validator.db.sql.tournaments import get_tournament_participants
from validator.db.sql.tournaments import get_tournament_rounds
from validator.db.sql.tournaments import get_tournament_rounds_with_status
from validator.db.sql.tournaments import get_tournament_tasks
from validator.db.sql.tournaments import get_tournaments_with_status
from validator.db.sql.tournaments import get_training_status_for_task
from validator.db.sql.tournaments import insert_tournament_groups_with_members
from validator.db.sql.tournaments import insert_tournament_pairs
from validator.db.sql.tournaments import insert_tournament_round
from validator.db.sql.tournaments import update_round_status
from validator.db.sql.tournaments import update_tournament_participant_backup_repo
from validator.db.sql.tournaments import update_tournament_participant_training_repo
from validator.db.sql.tournaments import update_tournament_status
from validator.db.sql.tournaments import update_tournament_winner_hotkey
from validator.db.sql.transfers import deduct_tournament_participation_fee
from validator.db.sql.transfers import get_coldkey_balance_by_address
from validator.tournament import constants as t_cst
from validator.tournament.benchmark_utils import create_benchmark_tasks_for_tournament_winner
from validator.tournament.repo_uploader import upload_tournament_participant_repository
from validator.tournament.task_creator import create_environment_tournament_tasks
from validator.tournament.task_creator import create_image_tournament_tasks
from validator.tournament.task_creator import create_text_tournament_tasks
from validator.tournament.task_creator import replace_tournament_task
from validator.tournament.utils import get_base_contestant
from validator.tournament.utils import get_latest_tournament_winner_participant
from validator.tournament.utils import get_round_winners
from validator.tournament.utils import notify_tournament_completed
from validator.tournament.utils import notify_tournament_started
from validator.tournament.utils import send_to_discord
from validator.tournament.utils import validate_repo_license
from validator.tournament.utils import validate_repo_obfuscation
from validator.utils.call_endpoint import process_non_stream_fiber_get
from validator.utils.logging import LogContext
from validator.utils.logging import get_logger


logger = get_logger(__name__)


def count_failed_trainings_percentage(trainings: dict[str, str]) -> bool:
    """Return True if fraction of failures exceeds threshold."""
    total = len(trainings)
    if total == 0:
        return False

    failures = sum(1 for status in trainings.values() if status in {TaskStatus.FAILURE.value, TaskStatus.PREP_TASK_FAILURE.value})
    return (failures / total) > t_cst.PERCENTAGE_OF_TASKS_SHOULD_BE_SUCCESS


def organise_tournament_round(nodes: list[Node], config: Config, tournament_type: TournamentType | None = None) -> Round:
    nodes_copy = nodes.copy()
    random.shuffle(nodes_copy)

    # Environment tournaments always use a single group round with all participants
    # Minimum group size of 5 required for environment tournaments
    if tournament_type == TournamentType.ENVIRONMENT:
        if len(nodes_copy) < t_cst.MIN_ENVIRONMENT_GROUP_SIZE:
            logger.warning(
                f"Environment tournament requires minimum 5 participants, but only {len(nodes_copy)} provided. "
                f"Cannot create tournament round."
            )
            raise ValueError(f"Environment tournament requires minimum 5 participants, got {len(nodes_copy)}")
        all_hotkeys = [node.hotkey for node in nodes_copy]
        single_group = Group(member_ids=all_hotkeys, task_ids=[])
        return GroupRound(groups=[single_group])

    if len(nodes_copy) <= t_cst.MAX_NUMBER_OF_MINERS_FOR_KNOCKOUT_ROUND:
        hotkeys = [node.hotkey for node in nodes_copy]

        if len(hotkeys) % 2 == 1:
            if cst.EMISSION_BURN_HOTKEY not in hotkeys:
                hotkeys.append(cst.EMISSION_BURN_HOTKEY)
            else:
                hotkeys.remove(cst.EMISSION_BURN_HOTKEY)

        random.shuffle(hotkeys)
        pairs = []
        for i in range(0, len(hotkeys), 2):
            pairs.append((hotkeys[i], hotkeys[i + 1]))
        random.shuffle(pairs)
        return KnockoutRound(pairs=pairs)
    else:
        num_groups = math.ceil(len(nodes_copy) / t_cst.EXPECTED_GROUP_SIZE)

        if len(nodes_copy) / num_groups < t_cst.MIN_GROUP_SIZE:
            num_groups = max(1, num_groups - 1)

        groups = [[] for _ in range(num_groups)]
        base_size = len(nodes_copy) // num_groups
        remainder = len(nodes_copy) % num_groups
        group_sizes = [base_size + (1 if i < remainder else 0) for i in range(num_groups)]

        random.shuffle(nodes_copy)
        idx = 0
        for g in range(num_groups):
            group_nodes = nodes_copy[idx : idx + group_sizes[g]]
            group_hotkeys = [node.hotkey for node in group_nodes]
            groups[g] = Group(member_ids=group_hotkeys, task_ids=[])
            idx += group_sizes[g]

        random.shuffle(groups)
        return GroupRound(groups=groups)


async def _create_first_round(
    tournament_id: str, tournament_type: TournamentType, nodes: list[Node], psql_db: PSQLDB, config: Config
):
    round_id = generate_round_id(tournament_id, 1)
    with LogContext(round_id=round_id):
        round_structure = organise_tournament_round(nodes, config, tournament_type)

        round_type = RoundType.KNOCKOUT if isinstance(round_structure, KnockoutRound) else RoundType.GROUP

        # Environment tournaments only have group stage, so mark it as final
        is_final_round = tournament_type == TournamentType.ENVIRONMENT

        round_data = TournamentRoundData(
            round_id=round_id,
            tournament_id=tournament_id,
            round_number=1,
            round_type=round_type,
            is_final_round=is_final_round,
            status=RoundStatus.PENDING,
        )

        await insert_tournament_round(round_data, psql_db)

        if isinstance(round_structure, GroupRound):
            await insert_tournament_groups_with_members(round_id, round_structure, psql_db)
        else:
            await insert_tournament_pairs(round_id, round_structure.pairs, psql_db)

        logger.info(f"Created first round {round_id}")


async def _create_tournament_tasks(
    tournament_id: str, round_id: str, round_structure: Round, tournament_type: TournamentType, is_final: bool, config: Config
) -> list[str]:
    if tournament_type == TournamentType.TEXT:
        tasks = await create_text_tournament_tasks(round_structure, tournament_id, round_id, config, is_final)
    elif tournament_type == TournamentType.IMAGE:
        tasks = await create_image_tournament_tasks(round_structure, tournament_id, round_id, config, is_final)
    elif tournament_type == TournamentType.ENVIRONMENT:
        tasks = await create_environment_tournament_tasks(round_structure, tournament_id, round_id, config, is_final)
    else:
        raise ValueError(f"Unknown tournament type: {tournament_type}")

    return tasks


async def assign_nodes_to_tournament_tasks(
    tournament_id: str, round_id: str, round_structure: Round, psql_db: PSQLDB, is_final_round: bool = False
) -> None:
    """Assign nodes to tournament tasks for the given round."""

    tournament = await get_tournament(tournament_id, psql_db)
    is_environment_tournament = tournament and tournament.tournament_type == TournamentType.ENVIRONMENT

    if isinstance(round_structure, GroupRound):
        # For environment tournaments, collect all participants from all groups
        if is_environment_tournament:
            logger.info("Processing ENVIRONMENT tournament - assigning all participants + boss to single task")
            all_participants = []
            for group in round_structure.groups:
                all_participants.extend(group.member_ids)

            if EMISSION_BURN_HOTKEY not in all_participants:
                all_participants.append(EMISSION_BURN_HOTKEY)
                logger.info(f"Adding boss contestant {EMISSION_BURN_HOTKEY} to environment tournament task")
        else:
            all_participants = None

        for i, group in enumerate(round_structure.groups):
            if is_environment_tournament:
                current_group_id = f"{round_id}_group_001"
                participants_to_assign = all_participants
            else:
                current_group_id = f"{round_id}_group_{i + 1:03d}"
                participants_to_assign = group.member_ids

            group_tasks = await get_tournament_tasks(round_id, psql_db)
            group_tasks = [task for task in group_tasks if task.group_id == current_group_id]

            for task in group_tasks:
                already_assigned_nodes = await task_sql.get_nodes_assigned_to_task(task.task_id, psql_db)
                already_assigned_hotkeys = {node.hotkey for node in already_assigned_nodes}

                for hotkey in participants_to_assign:
                    if hotkey in already_assigned_hotkeys:
                        logger.info(f"Node {hotkey} already assigned to task {task.task_id}")
                        continue

                    node = await get_node_by_hotkey(hotkey, psql_db)
                    if node:
                        await task_sql.assign_node_to_task(task.task_id, node, psql_db)

                        expected_repo_name = f"tournament-{tournament_id}-{task.task_id}-{hotkey[:8]}"
                        await task_sql.set_expected_repo_name(task.task_id, node, psql_db, expected_repo_name)

                        logger.info(
                            f"Assigned {hotkey} to group task {task.task_id} with expected_repo_name: {expected_repo_name}"
                        )

            if is_environment_tournament:
                logger.info(f"Finished assigning {len(all_participants)} participants to environment tournament task(s)")
                break
    else:
        logger.info("Processing KNOCKOUT round assignment")
        round_tasks = await get_tournament_tasks(round_id, psql_db)
        logger.info(f"Found {len(round_tasks)} tasks for round {round_id}")

        for i, pair in enumerate(round_structure.pairs):
            pair_id = f"{round_id}_pair_{i + 1:03d}"
            logger.info(f"Processing pair {i + 1}/{len(round_structure.pairs)}: {pair} -> {pair_id}")

            pair_tasks = [task for task in round_tasks if task.pair_id == pair_id]
            logger.info(f"Found {len(pair_tasks)} tasks for pair {pair_id}")

            for pair_task in pair_tasks:
                logger.info(f"Assigning nodes to task {pair_task.task_id}")

                already_assigned_nodes = await task_sql.get_nodes_assigned_to_task(pair_task.task_id, psql_db)
                already_assigned_hotkeys = {node.hotkey for node in already_assigned_nodes}

                participants_to_assign = list(pair)

                # For final rounds, also assign the boss contestant
                if is_final_round and EMISSION_BURN_HOTKEY not in participants_to_assign:
                    participants_to_assign.append(EMISSION_BURN_HOTKEY)
                    logger.info(
                        f"Final round detected - adding boss contestant {EMISSION_BURN_HOTKEY} to task {pair_task.task_id}"
                    )

                for hotkey in participants_to_assign:
                    if hotkey in already_assigned_hotkeys:
                        logger.info(f"Node {hotkey} already assigned to task {pair_task.task_id}")
                        continue

                    node = await get_node_by_hotkey(hotkey, psql_db)
                    if node:
                        await task_sql.assign_node_to_task(pair_task.task_id, node, psql_db)

                        expected_repo_name = f"tournament-{tournament_id}-{pair_task.task_id}-{hotkey[:8]}"
                        await task_sql.set_expected_repo_name(pair_task.task_id, node, psql_db, expected_repo_name)

                        logger.info(
                            f"Assigned {hotkey} to pair task {pair_task.task_id} with expected_repo_name: {expected_repo_name}"
                        )
                    else:
                        logger.warning(f"Could not find node for hotkey {hotkey} during task assignment")


async def create_next_round(
    tournament: TournamentData, completed_round: TournamentRoundData, winners: list[str], config, psql_db: PSQLDB
):
    """Create the next round of the tournament."""
    if tournament.tournament_type == TournamentType.ENVIRONMENT:
        logger.info(f"Environment tournament {tournament.tournament_id} only has group stage, no next round to create")
        return

    next_round_number = completed_round.round_number + 1
    next_round_id = generate_round_id(tournament.tournament_id, next_round_number)

    with LogContext(tournament_id=tournament.tournament_id, round_id=next_round_id):
        logger.info(f"Creating next round with {len(winners)} winners: {winners}")
        next_round_is_final = len(winners) == 1

        if len(winners) == 2:
            if cst.EMISSION_BURN_HOTKEY in winners:
                next_round_is_final = True
        elif len(winners) % 2 == 1:
            if cst.EMISSION_BURN_HOTKEY not in winners:
                winners.append(cst.EMISSION_BURN_HOTKEY)
                logger.info("Added burn hotkey to make even number of participants")
            else:
                if len(winners) == 1:
                    next_round_is_final = True
                else:
                    winners = [w for w in winners if w != cst.EMISSION_BURN_HOTKEY]
                    logger.info("Removed burn hotkey to make even number of participants")

        winner_nodes = []
        for hotkey in winners:
            node = await get_node_by_hotkey(hotkey, psql_db)
            if node:
                winner_nodes.append(node)
                logger.info(f"Found node for winner {hotkey}")
            else:
                logger.warning(
                    f"CRITICAL: Could not find node for winner {hotkey} - this winner will be excluded from next round!"
                )

        if not winner_nodes:
            logger.error("No winner nodes found, cannot create next round")
            return

        logger.info(f"Successfully found {len(winner_nodes)} nodes out of {len(winners)} winners")

        round_structure = organise_tournament_round(winner_nodes, config)

        round_type = RoundType.KNOCKOUT if isinstance(round_structure, KnockoutRound) else RoundType.GROUP

        round_data = TournamentRoundData(
            round_id=next_round_id,
            tournament_id=tournament.tournament_id,
            round_number=next_round_number,
            round_type=round_type,
            is_final_round=next_round_is_final,
            status=RoundStatus.PENDING,
        )

        await insert_tournament_round(round_data, psql_db)

        if isinstance(round_structure, GroupRound):
            await insert_tournament_groups_with_members(next_round_id, round_structure, psql_db)
        else:
            await insert_tournament_pairs(next_round_id, round_structure.pairs, psql_db)

        logger.info(f"Created next round {next_round_id}")


async def advance_tournament(tournament: TournamentData, completed_round: TournamentRoundData, config: Config, psql_db: PSQLDB):
    with LogContext(tournament_id=tournament.tournament_id, round_id=completed_round.round_id):
        logger.info("=== ADVANCE TOURNAMENT CALLED ===")
        logger.info(f"Tournament: {tournament.tournament_id}, Status: {tournament.status}")
        logger.info(f"Completed Round: {completed_round.round_id}, Round #: {completed_round.round_number}")
        logger.info(f"Is Final Round: {completed_round.is_final_round}")

        if tournament.winner_hotkey:
            logger.info(
                f"Tournament {tournament.tournament_id} already has winner {tournament.winner_hotkey}. "
                f"Skipping advance (tournament completed, awaiting manual status update)."
            )
            return

        logger.info(f"Advancing tournament {tournament.tournament_id} from round {completed_round.round_id}")

        winners = await get_round_winners(completed_round, psql_db, config)
        logger.info(f"Round winners: {winners}")
        logger.info(f"Number of winners: {len(winners)}")

        # Get all active participants and handle eliminations
        all_participants = await get_tournament_participants(tournament.tournament_id, psql_db)
        active_participants = [p.hotkey for p in all_participants if p.eliminated_in_round_id is None]
        logger.info(f"Active participants before elimination: {len(active_participants)} - {active_participants}")

        # Eliminate losers (those who didn't win)
        losers = [p for p in active_participants if p not in winners]
        logger.info(f"Losers to be eliminated: {len(losers)} - {losers}")

        all_eliminated = losers
        if all_eliminated:
            await eliminate_tournament_participants(tournament.tournament_id, completed_round.round_id, all_eliminated, psql_db)

        logger.info(f"Final winners: {len(winners)} - {winners}")

        if len(winners) == 0:
            logger.warning(
                f"No winners found for round {completed_round.round_id}. Setting base contestant as winner of the tournament."
            )
            # Keep EMISSION_BURN_HOTKEY as the winner when defending champion wins by default
            winner = cst.EMISSION_BURN_HOTKEY
            await update_tournament_winner_hotkey(tournament.tournament_id, winner, psql_db)
            # await update_tournament_status(tournament.tournament_id, TournamentStatus.COMPLETED, psql_db)
            logger.info(f"Tournament {tournament.tournament_id} completed with winner: {winner}. Please update DB manually.")

            await notify_tournament_completed(
                tournament.tournament_id, tournament.tournament_type.value, winner, config.discord_url
            )

            await upload_participant_repository(tournament.tournament_id, tournament.tournament_type, winner, 1, config, psql_db)
            return
        
        if len(winners) == 1 or tournament.tournament_type == TournamentType.ENVIRONMENT:
            winner = winners[0]
            # Keep the winner as-is (EMISSION_BURN_HOTKEY if defending champion won)
            # The base_winner_hotkey field already tracks the actual identity for display purposes
            logger.info(f"Processing final round completion for tournament {tournament.tournament_id}")
            logger.info(f"Final round winner: {winner}")
            if winner == cst.EMISSION_BURN_HOTKEY and tournament.base_winner_hotkey:
                logger.info(
                    f"Defending champion {tournament.base_winner_hotkey} successfully defended (stored as EMISSION_BURN_HOTKEY)"
                )

            round_tasks = await get_tournament_tasks(completed_round.round_id, psql_db)
            logger.info(f"Found {len(round_tasks)} tasks in final round")

            # Get task IDs for logging
            task_ids = [task.task_id for task in round_tasks]
            logger.info(f"Tournament task IDs: {task_ids}")

            await update_tournament_winner_hotkey(tournament.tournament_id, winner, psql_db)
            # await update_tournament_status(tournament.tournament_id, TournamentStatus.COMPLETED, psql_db)
            logger.info(f"Tournament {tournament.tournament_id} completed with winner: {winner}. Please update DB manually.")

            await notify_tournament_completed(
                tournament.tournament_id, tournament.tournament_type.value, winner, config.discord_url
            )

            if completed_round.is_final_round and tournament.tournament_type == TournamentType.ENVIRONMENT:
                logger.info("Uploading winner and 2nd place repositories")
                if winner != cst.EMISSION_BURN_HOTKEY:
                    try:
                        logger.info(f"Creating benchmark tasks for tournament winner {winner}")
                        benchmark_task_ids = await create_benchmark_tasks_for_tournament_winner(
                            tournament.tournament_id, winner, config
                        )
                        logger.info(f"Created {len(benchmark_task_ids)} benchmark tasks for tournament winner {winner}")
                    except Exception as e:
                        logger.error(f"Error creating benchmark tasks for tournament winner {winner}: {str(e)}")

                logger.info(f"Uploading position 1 repository for hotkey: {winner}")
                await upload_participant_repository(
                    tournament.tournament_id, tournament.tournament_type, winner, 1, config, psql_db
                )
                
                if len(winners) >= 2:
                    second_place = winners[1]
                    logger.info(f"Uploading position 2 repository for hotkey: {second_place}")
                    await upload_participant_repository(
                        tournament.tournament_id, tournament.tournament_type, second_place, 2, config, psql_db
                    )
            else:
                try:
                    participant1, participant2 = await get_final_round_participants(completed_round, psql_db)
                    logger.info(f"Final round participants from DB: {participant1}, {participant2}")
                    logger.info(f"Winner determined by get_round_winners: {winner}")
                    logger.info(f"Tournament base_winner_hotkey (previous champion): {tournament.base_winner_hotkey}")

                    loser = participant2 if participant1 == winner else participant1
                    logger.info(f"Loser determined: {loser}")

                    position_1_upload = winner
                    position_2_upload = loser

                    if winner != cst.EMISSION_BURN_HOTKEY:
                        try:
                            logger.info(f"Creating benchmark tasks for tournament winner {winner}")
                            benchmark_task_ids = await create_benchmark_tasks_for_tournament_winner(
                                tournament.tournament_id, winner, config
                            )
                            logger.info(f"Created {len(benchmark_task_ids)} benchmark tasks for tournament winner {winner}")
                        except Exception as e:
                            logger.error(f"Error creating benchmark tasks for tournament winner {winner}: {str(e)}")

                    logger.info(f"Uploading position 1 repository for hotkey: {position_1_upload}")
                    await upload_participant_repository(
                        tournament.tournament_id, tournament.tournament_type, position_1_upload, 1, config, psql_db
                    )

                    logger.info(f"Uploading position 2 repository for hotkey: {position_2_upload}")
                    await upload_participant_repository(
                        tournament.tournament_id, tournament.tournament_type, position_2_upload, 2, config, psql_db
                    )
                except Exception as e:
                    logger.error(f"Error determining final round participants: {e}")
                    await upload_participant_repository(
                        tournament.tournament_id, tournament.tournament_type, winner, 1, config, psql_db
                    )
            return
        else:
            await create_next_round(tournament, completed_round, winners, config, psql_db)


async def create_basic_tournament(tournament_type: TournamentType, psql_db: PSQLDB, config: Config) -> str:
    """Create a basic tournament in the database without participants or rounds."""
    tournament_id = generate_tournament_id()

    base_contestant = await get_base_contestant(psql_db, tournament_type, config)

    # Get the actual champion's hotkey (not EMISSION_BURN_HOTKEY)
    latest_winner = await get_latest_tournament_winner_participant(psql_db, tournament_type, config)
    base_winner_hotkey = latest_winner.hotkey if latest_winner else None

    logger.info(f"Base winner hotkey (actual champion): {base_winner_hotkey}")
    logger.info(f"Base contestant hotkey (EMISSION_BURN): {base_contestant.hotkey if base_contestant else None}")

    tournament_data = TournamentData(
        tournament_id=tournament_id,
        tournament_type=tournament_type,
        status=TournamentStatus.PENDING,
        base_winner_hotkey=base_winner_hotkey,  # Store actual champion's hotkey
    )

    await create_tournament(tournament_data, psql_db)

    if base_contestant and base_contestant.hotkey:
        base_participant = TournamentParticipant(
            tournament_id=tournament_id,
            hotkey=base_contestant.hotkey,  # This will be EMISSION_BURN_HOTKEY
            training_repo=base_contestant.training_repo,
            training_commit_hash=base_contestant.training_commit_hash,
        )
        await add_tournament_participants([base_participant], psql_db)

    logger.info(f"Created basic tournament {tournament_id} with type {tournament_type.value}")

    return tournament_id


async def populate_tournament_participants(tournament_id: str, config: Config, psql_db: PSQLDB) -> int:
    logger.info(
        f"Populating participants for tournament {tournament_id} with minimum requirement of {cst.MIN_MINERS_FOR_TOURN} miners"
    )

    tournament = await get_tournament(tournament_id, psql_db)
    if not tournament:
        logger.error(f"Tournament {tournament_id} not found")
        return 0

    if tournament.tournament_type == TournamentType.TEXT:
        participation_fee_rao = t_cst.TOURNAMENT_TEXT_PARTICIPATION_FEE_RAO
        fee_description = "0.2 TAO"
    elif tournament.tournament_type == TournamentType.IMAGE:
        participation_fee_rao = t_cst.TOURNAMENT_IMAGE_PARTICIPATION_FEE_RAO
        fee_description = "0.15 TAO"
    elif tournament.tournament_type == TournamentType.ENVIRONMENT:
        participation_fee_rao = t_cst.TOURNAMENT_ENVIRONMENT_PARTICIPATION_FEE_RAO
        fee_description = "0.20 TAO"
    else:
        raise ValueError(f"Unknown tournament type: {tournament.tournament_type}")

    logger.info(f"Tournament type: {tournament.tournament_type.value}, participation fee: {fee_description}")

    while True:
        all_nodes = await get_all_nodes(psql_db)

        # Get all nodes except base contestant
        eligible_nodes = [node for node in all_nodes if node.hotkey != cst.EMISSION_BURN_HOTKEY]

        if not eligible_nodes:
            logger.warning("No eligible nodes found for tournament")
            return 0

        logger.info(f"Found {len(eligible_nodes)} eligible nodes in database")

        responding_nodes: list[RespondingNode] = []
        batch_size = t_cst.TOURNAMENT_PARTICIPANT_PING_BATCH_SIZE

        for i in range(0, len(eligible_nodes), batch_size):
            batch = eligible_nodes[i : i + batch_size]
            logger.info(
                f"Processing batch {i // batch_size + 1}/{(len(eligible_nodes) + batch_size - 1) // batch_size} "
                f"with {len(batch)} nodes"
            )

            batch_results = await asyncio.gather(
                *[_get_miner_training_repo(node, config, tournament.tournament_type) for node in batch],
                return_exceptions=True,
            )

            for node, result in zip(batch, batch_results):
                with LogContext(node_hotkey=node.hotkey):
                    if isinstance(result, Exception):
                        logger.warning(f"Exception pinging {node.hotkey}: {result}")
                    elif result:
                        responding_node = RespondingNode(node=node, training_repo_response=result)
                        responding_nodes.append(responding_node)
                        logger.info(f"Node responded with training repo {result.github_repo}@{result.commit_hash}")

        logger.info(f"Got {len(responding_nodes)} responding nodes")

        logger.info(f"Processing {len(responding_nodes)} responding nodes")

        logger.info("Validating obfuscation, license, and balance for participants...")
        validated_nodes: list[RespondingNode] = []

        for responding_node in responding_nodes:
            with LogContext(node_hotkey=responding_node.node.hotkey):
                repo_url = responding_node.training_repo_response.github_repo
                logger.info(f"Checking obfuscation for {responding_node.node.hotkey}'s repo: {repo_url}")

                is_not_obfuscated = await validate_repo_obfuscation(repo_url)

                if not is_not_obfuscated:
                    logger.warning(
                        f"Repository {repo_url} failed obfuscation validation for hotkey {responding_node.node.hotkey}. "
                        f"Excluding from tournament."
                    )
                    continue

                logger.info(f"Checking license for {responding_node.node.hotkey}'s repo: {repo_url}")

                has_valid_license = await validate_repo_license(repo_url)

                if not has_valid_license:
                    logger.warning(
                        f"Repository {repo_url} failed license validation for hotkey {responding_node.node.hotkey}. "
                        f"Excluding from tournament."
                    )
                    continue

                balance = await get_coldkey_balance_by_address(psql_db, responding_node.node.coldkey)

                if not balance or balance.balance_rao < participation_fee_rao:
                    logger.warning(
                        f"Skipping {responding_node.node.hotkey} - insufficient balance. "
                        f"Required: {participation_fee_rao:,} RAO, "
                        f"Available: {balance.balance_rao if balance else 0:,} RAO"
                    )
                    continue

                fee_deducted = await deduct_tournament_participation_fee(
                    psql_db, responding_node.node.coldkey, participation_fee_rao, tournament_id
                )

                if not fee_deducted:
                    logger.warning(f"Failed to deduct participation fee for {responding_node.node.hotkey}. Skipping node.")
                    continue

                validated_nodes.append(responding_node)
                logger.info(
                    f"Repository {repo_url} passed obfuscation, license, and balance checks for hotkey "
                    f"{responding_node.node.hotkey} (deducted {participation_fee_rao:,} RAO participation fee)"
                )

        logger.info(
            f"Validation complete: {len(validated_nodes)} participants selected from {len(responding_nodes)} responding nodes"
        )

        miners_that_accept_and_give_repos = 0

        for responding_node in validated_nodes:
            with LogContext(node_hotkey=responding_node.node.hotkey):
                participant = TournamentParticipant(
                    tournament_id=tournament_id,
                    hotkey=responding_node.node.hotkey,
                )
                await add_tournament_participants([participant], psql_db)

                await update_tournament_participant_training_repo(
                    tournament_id,
                    responding_node.node.hotkey,
                    responding_node.training_repo_response.github_repo,
                    responding_node.training_repo_response.commit_hash,
                    psql_db,
                )

                miners_that_accept_and_give_repos += 1
                logger.info(f"Added {responding_node.node.hotkey} to tournament {tournament_id}")

        logger.info(f"Successfully populated {miners_that_accept_and_give_repos} participants for tournament {tournament_id}")

        if tournament.tournament_type == TournamentType.ENVIRONMENT:
            min_required_miners = cst.MIN_MINERS_FOR_ENV_TOURN
        else:
            min_required_miners = cst.MIN_MINERS_FOR_TOURN

        if miners_that_accept_and_give_repos >= min_required_miners:
            logger.info(
                f"Tournament {tournament_id} has sufficient miners "
                f"({miners_that_accept_and_give_repos} >= {min_required_miners})"
            )
            return miners_that_accept_and_give_repos

        logger.warning(
            f"Tournament {tournament_id} only has {miners_that_accept_and_give_repos} miners that accept and give repos, "
            f"need at least {min_required_miners}. Waiting 30 minutes and retrying..."
        )
        await asyncio.sleep(30 * 60)


async def _get_miner_training_repo(node: Node, config: Config, tournament_type: TournamentType) -> TrainingRepoResponse | None:
    """Get training repo from a miner, similar to how submissions are fetched in the main validator cycle."""
    try:
        url = f"{cst.TRAINING_REPO_ENDPOINT}/{tournament_type.value}"
        response = await process_non_stream_fiber_get(url, config, node)

        if response and isinstance(response, dict):
            return TrainingRepoResponse(**response)
        else:
            logger.warning(f"Invalid response format from {node.hotkey}: {response}")
            return None

    except Exception as e:
        logger.error(f"Failed to get training repo from {node.hotkey}: {e}")
        return None


async def create_first_round_for_active_tournament(tournament_id: str, config: Config, psql_db: PSQLDB) -> bool:
    logger.info(f"Checking if tournament {tournament_id} needs first round creation")

    existing_rounds = await get_tournament_rounds(tournament_id, psql_db)
    if existing_rounds:
        logger.info(f"Tournament {tournament_id} already has {len(existing_rounds)} rounds")
        return False

    tournament = await get_tournament(tournament_id, psql_db)
    if not tournament:
        logger.error(f"Tournament {tournament_id} not found")
        return False

    participants = await get_tournament_participants(tournament_id, psql_db)
    if not participants:
        logger.error(f"No participants found for tournament {tournament_id}")
        return False

    participant_nodes = []
    for participant in participants:
        if participant.hotkey == cst.EMISSION_BURN_HOTKEY:
            continue

        node = await get_node_by_hotkey(participant.hotkey, psql_db)
        if node:
            participant_nodes.append(node)

    if not participant_nodes:
        logger.error(f"No valid nodes found for tournament {tournament_id} participants")
        return False

    logger.info(f"Creating first round for tournament {tournament_id} with {len(participant_nodes)} participants")

    await _create_first_round(tournament_id, tournament.tournament_type, participant_nodes, psql_db, config)

    logger.info(f"Successfully created first round for tournament {tournament_id}")
    return True


async def process_pending_tournaments(config: Config) -> list[str]:
    """
    Process all pending tournaments by populating participants and activating them.
    """
    while True:
        logger.info("Processing pending tournaments...")

        try:
            pending_tournaments = await get_tournaments_with_status(TournamentStatus.PENDING, config.psql_db)

            logger.info(f"Found {len(pending_tournaments)} pending tournaments")

            activated_tournaments = []

            for tournament in pending_tournaments:
                with LogContext(tournament_id=tournament.tournament_id):
                    logger.info(f"Processing pending tournament {tournament.tournament_id}")

                    num_participants = await populate_tournament_participants(tournament.tournament_id, config, config.psql_db)

                    if num_participants > 0:
                        await update_tournament_status(tournament.tournament_id, TournamentStatus.ACTIVE, config.psql_db)
                        activated_tournaments.append(tournament.tournament_id)
                        logger.info(f"Activated tournament {tournament.tournament_id} with {num_participants} participants")

                        await notify_tournament_started(
                            tournament.tournament_id, tournament.tournament_type.value, num_participants, config.discord_url
                        )
                    else:
                        logger.warning(f"Tournament {tournament.tournament_id} has no participants, skipping activation")

            logger.info(f"Activated tournaments: {activated_tournaments}")
        except Exception as e:
            logger.error(f"Error processing pending tournaments: {e}")
        finally:
            await asyncio.sleep(t_cst.TOURNAMENT_PENDING_CYCLE_INTERVAL)


async def check_if_all_tasks_have_nodes_assigned(round_id: str, config: Config) -> bool:
    """
    True if all tasks have nodes assigned, False otherwise
    """
    logger.info(f"Checking if all tasks in round {round_id} have nodes assigned...")

    round_tasks = await get_tournament_tasks(round_id, config.psql_db)

    if not round_tasks:
        logger.warning(f"No tasks found for round {round_id}")
        return False

    logger.info(f"Found {len(round_tasks)} tasks for round {round_id}")

    tasks_without_nodes = []
    tasks_with_nodes = []

    for task in round_tasks:
        assigned_nodes = await task_sql.get_nodes_assigned_to_task(task.task_id, config.psql_db)
        if assigned_nodes:
            tasks_with_nodes.append(task.task_id)
            logger.info(f"Task {task.task_id} has {len(assigned_nodes)} nodes assigned")
        else:
            tasks_without_nodes.append(task.task_id)
            logger.warning(f"Task {task.task_id} has no nodes assigned")

    if tasks_without_nodes:
        logger.warning(f"Round {round_id} has {len(tasks_without_nodes)} tasks without nodes: {tasks_without_nodes}")
        logger.warning(f"Round {round_id} has {len(tasks_with_nodes)} tasks with nodes: {tasks_with_nodes}")
        return False

    logger.info(f"All {len(round_tasks)} tasks in round {round_id} have nodes assigned")
    return True


async def process_pending_rounds(config: Config):
    """
    Process all pending rounds by creating tasks and assigning nodes to them.
    """
    logger.info("Processing pending rounds...")

    while True:
        try:
            pending_rounds = await get_tournament_rounds_with_status(RoundStatus.PENDING, config.psql_db)

            logger.info(f"Found {len(pending_rounds)} pending rounds")

            for round_data in pending_rounds:
                with LogContext(tournament_id=round_data.tournament_id, round_id=round_data.round_id):
                    logger.info(f"Processing pending round {round_data.round_id} (type: {round_data.round_type})")

                    try:
                        tournament = await get_tournament(round_data.tournament_id, config.psql_db)
                        logger.info(f"Found tournament {tournament.tournament_id} with status {tournament.status}")

                        if round_data.round_type == RoundType.GROUP:
                            logger.info("Processing GROUP round")
                            groups_data = await get_tournament_groups(round_data.round_id, config.psql_db)
                            logger.info(f"Found {len(groups_data)} groups")
                            groups = []
                            for group_data in groups_data:
                                members = await get_tournament_group_members(group_data.group_id, config.psql_db)
                                member_ids = [member.hotkey for member in members]
                                groups.append(Group(member_ids=member_ids))
                                logger.info(f"Group {group_data.group_id}: {len(member_ids)} members")
                            round_structure = GroupRound(groups=groups)
                        else:
                            logger.info("Processing KNOCKOUT round")
                            pairs = await get_tournament_pairs(round_data.round_id, config.psql_db)
                            logger.info(f"Found {len(pairs)} pairs: {[(p.hotkey1, p.hotkey2) for p in pairs]}")
                            round_structure = KnockoutRound(pairs=[(pair.hotkey1, pair.hotkey2) for pair in pairs])

                        logger.info(f"About to create tournament tasks for round {round_data.round_id}")
                        tasks = await _create_tournament_tasks(
                            round_data.tournament_id,
                            round_data.round_id,
                            round_structure,
                            tournament.tournament_type,
                            round_data.is_final_round,
                            config,
                        )
                        logger.info(f"Created {len(tasks)} tasks for round {round_data.round_id}")

                        logger.info("About to assign nodes to tournament tasks")
                        await assign_nodes_to_tournament_tasks(
                            round_data.tournament_id,
                            round_data.round_id,
                            round_structure,
                            config.psql_db,
                            round_data.is_final_round,
                        )
                        logger.info("Finished assigning nodes to tournament tasks")

                        if await check_if_all_tasks_have_nodes_assigned(round_data.round_id, config):
                            logger.info(f"Setting round {round_data.round_id} to ACTIVE status")
                            await update_round_status(round_data.round_id, RoundStatus.ACTIVE, config.psql_db)
                            logger.info(f"Successfully processed pending round {round_data.round_id} with {len(tasks)} tasks")
                        else:
                            logger.warning(
                                f"Round {round_data.round_id} has tasks without nodes assigned. Keeping round as PENDING."
                            )
                            logger.info(f"Round {round_data.round_id} will be processed again in the next cycle")

                    except Exception as e:
                        logger.error(f"Error processing pending round {round_data.round_id}: {e}")

        except Exception as e:
            logger.error(f"Error processing pending rounds: {e}")
        finally:
            await asyncio.sleep(t_cst.TOURNAMENT_PENDING_ROUND_CYCLE_INTERVAL)


async def process_active_tournaments(config: Config):
    """
    Process all active tournaments by advancing them if needed.
    """
    logger.info("Processing active tournaments...")

    while True:
        try:
            active_tournaments = await get_tournaments_with_status(TournamentStatus.ACTIVE, config.psql_db)
            for tournament in active_tournaments:
                with LogContext(tournament_id=tournament.tournament_id):
                    logger.info(f"Processing active tournament {tournament.tournament_id}")
                    rounds = await get_tournament_rounds(tournament.tournament_id, config.psql_db)
                    if not rounds:
                        logger.info(f"Tournament {tournament.tournament_id} has no rounds, creating first round...")
                        await create_first_round_for_active_tournament(tournament.tournament_id, config, config.psql_db)
                    else:
                        current_round = rounds[-1]
                        if current_round.status == RoundStatus.ACTIVE:
                            if await check_if_round_is_completed(current_round, config):
                                await update_round_status(current_round.round_id, RoundStatus.COMPLETED, config.psql_db)
                                logger.info(
                                    f"Tournament {tournament.tournament_id} round {current_round.round_id} is completed, "
                                    f"advancing..."
                                )
                                await advance_tournament(tournament, current_round, config, config.psql_db)
                        elif current_round.status == RoundStatus.COMPLETED:
                            # If the round is already completed but tournament is still active,
                            # we need to advance the tournament
                            logger.info(
                                f"Tournament {tournament.tournament_id} round {current_round.round_id} is already completed, "
                                f"checking if tournament should advance..."
                            )
                            await advance_tournament(tournament, current_round, config, config.psql_db)
        except Exception as e:
            logger.error(f"Error processing active tournaments: {e}", exc_info=True)
        finally:
            await asyncio.sleep(t_cst.TOURNAMENT_ACTIVE_CYCLE_INTERVAL)


async def _notify_discord(message: str, config: Config) -> None:
    discord_url = config.discord_url
    if discord_url:
        try:
            await send_to_discord(
                webhook=discord_url,
                message=message,
            )
        except Exception as e:
            logger.error(f"Failed to send Discord notification: {e}")


async def _more_than_half_failures(tournament_task: TournamentTask, config: Config) -> bool:
    """
    Check if more than half of the trainings failed and handle Discord notification.

    Returns True if majority failure detected, False otherwise.
    """
    trainings = await get_training_status_for_task(tournament_task.task_id, config.psql_db)
    is_more_than_half_failure = count_failed_trainings_percentage(trainings)

    if is_more_than_half_failure:
        logger.info(f"More than half of the trainings for task {tournament_task.task_id} failed. Please investigate.")
        message = (
            f"Warning: Task {tournament_task.task_id} in Tournament Round {tournament_task.round_id} "
            f"has more than half tasks failed, please investigate."
        )
        await _notify_discord(message, config)
        return True

    return False


async def is_tourn_task_completed(
    tournament_task: TournamentTask, task_obj: AnyTypeTask, config: Config, final_round: bool = False
) -> tuple[bool, str]:
    """
    Checks if the tournament task is completed.
    If completed successfully, checks if majority trainings failed.
    If completed with failure, notifies via Discord.
    If completed with prep task failure, creates a replacement immediately.
    If not completed, returns False.

    Returns a tuple of (is_completed, reason)
    """

    if task_obj.status == TaskStatus.SUCCESS.value:
        if await _more_than_half_failures(tournament_task, config):
            return False, "More than half of the trainings failed"
        return True, "Task completed successfully"

    elif task_obj.status == TaskStatus.FAILURE.value:
        discord_message = (
            f"Warning: Task {tournament_task.task_id} in Tournament Round {tournament_task.round_id} "
            f"has failed, please investigate."
        )
        await _notify_discord(discord_message, config)
        return False, "Tournament task failed."

    elif task_obj.status == TaskStatus.PREP_TASK_FAILURE.value:
        logger.info(f"Task {task_obj.task_id} failed during preparation, creating replacement immediately.")
        new_task_id = await replace_tournament_task(
            tournament_task.task_id,
            tournament_task.tournament_id,
            tournament_task.round_id,
            tournament_task.group_id,
            tournament_task.pair_id,
            config,
        )
        return False, f"Task failed during preparation. Replaced with a new task {new_task_id}."

    return False, "Tournament task not completed. Status: " + task_obj.status


async def check_if_round_is_completed(round_data: TournamentRoundData, config: Config) -> bool:
    """Check if a round should be marked as completed based on task completion."""
    logger.info(f"Checking if round {round_data.round_id} should be completed...")

    round_tasks = await get_tournament_tasks(round_data.round_id, config.psql_db)

    if not round_tasks:
        logger.info(f"No tasks found for round {round_data.round_id}")
        return False

    task_objects = [await task_sql.get_task(task.task_id, config.psql_db) for task in round_tasks]

    task_completion_checks = [
        await is_tourn_task_completed(tournament_task, task_obj, config, round_data.is_final_round)
        for tournament_task, task_obj in zip(round_tasks, task_objects)
    ]

    assert len(round_tasks) == len(task_completion_checks), "Number of tasks and completion checks do not match"

    logger.info("Task completion check summary:")
    for task, completion_check in zip(round_tasks, task_completion_checks):
        if not completion_check[0]:
            logger.info(f"Task {task.task_id} is not completed. Reason: {completion_check[1]}")
        else:
            logger.info(f"Task {task.task_id} is completed. Reason: {completion_check[1]}")

    if all(completion_check[0] for completion_check in task_completion_checks):
        logger.info(f"Round {round_data.round_id} is completed.")
        return True
    else:
        logger.info(f"Round {round_data.round_id} is not completed.")
        return False


async def process_tournament_scheduling(config: Config):
    """
    Process tournament scheduling to automatically start new tournaments when the previous ones finish.
    Checks text, image, and environment tournaments independently.

    Tournaments only start during their scheduled day/hour window. If a scheduled start is missed,
    the system waits until the next week's scheduled time rather than starting late.
    """
    logger.info("Processing tournament scheduling...")

    while True:
        try:
            # Check all tournament types
            for tournament_type in [TournamentType.TEXT, TournamentType.IMAGE, TournamentType.ENVIRONMENT]:
                await check_and_start_tournament(tournament_type, config.psql_db, config)

        except Exception as e:
            logger.error(f"Error processing tournament scheduling: {e}")
        finally:
            await asyncio.sleep(t_cst.TOURNAMENT_ACTIVE_CYCLE_INTERVAL)


async def check_and_start_tournament(tournament_type: TournamentType, psql_db: PSQLDB, config: Config):
    """
    Check if we should start a new tournament of the given type.
    """
    with LogContext(tournament_type=tournament_type.value):
        # Check if there's already an active tournament of this type
        active_tournament = await get_active_tournament(psql_db, tournament_type)
        if active_tournament:
            logger.info(f"Active {tournament_type.value} tournament exists: {active_tournament.tournament_id}")
            return

        # Check if there's a pending tournament of this type
        pending_tournaments = await get_tournaments_with_status(TournamentStatus.PENDING, psql_db)
        pending_of_type = [t for t in pending_tournaments if t.tournament_type == tournament_type]
        if pending_of_type:
            logger.info(f"Pending {tournament_type.value} tournament exists: {pending_of_type[0].tournament_id}")
            return

        # Get the latest tournament and check if enough time has passed
        latest_tournament, created_at = await get_latest_tournament_with_created_at(psql_db, tournament_type)

        if latest_tournament and latest_tournament.status not in {TournamentStatus.ACTIVE, TournamentStatus.PENDING}:
            completed_at = await get_tournament_completion_time(latest_tournament.tournament_id, psql_db)
            if await should_start_new_tournament_after_interval(completed_at or created_at, tournament_type):
                logger.info(
                    f"Starting new {tournament_type.value} tournament at scheduled time "
                    f"(previous tournament {latest_tournament.tournament_id} completed)"
                )

                # Create new tournament of the same type
                new_tournament_id = await create_basic_tournament(tournament_type, psql_db, config)
                logger.info(f"Created new {tournament_type.value} tournament: {new_tournament_id}")
            else:
                logger.info(f"Waiting for next scheduled start time for {tournament_type.value} tournament")
        elif not latest_tournament:
            # No tournaments of this type exist, create the first one
            logger.info(f"No {tournament_type.value} tournaments found, creating first one")
            new_tournament_id = await create_basic_tournament(tournament_type, psql_db, config)
            logger.info(f"Created first {tournament_type.value} tournament: {new_tournament_id}")


async def get_tournament_completion_time(tournament_id: str, psql_db: PSQLDB) -> datetime | None:
    """Get the completion time (updated_at) for a completed tournament."""
    async with await psql_db.connection() as connection:
        query = """
            SELECT updated_at
            FROM tournaments
            WHERE tournament_id = $1 AND status = 'completed'
        """
        result = await connection.fetchval(query, tournament_id)
        return result


async def should_start_new_tournament_after_interval(last_created_at, tournament_type: TournamentType) -> bool:
    """
    Check if we're currently in the scheduled tournament start window for the given tournament type.
    Tournaments only start during their scheduled day and hour. If the window is missed,
    we wait until the next week's scheduled time.
    """
    if not last_created_at:
        return True

    now = datetime.now(timezone.utc)

    if last_created_at.tzinfo is None:
        last_created_at = last_created_at.replace(tzinfo=timezone.utc)

    scheduled_day, scheduled_hour = _get_tournament_schedule(tournament_type)

    next_start_time = _calculate_next_tournament_start_time(last_created_at, tournament_type)

    # Check if we're currently in the scheduled window (same day and hour as the scheduled time)
    # We allow starting anytime during the scheduled hour
    is_scheduled_day = now.weekday() == scheduled_day
    is_scheduled_hour = now.hour == scheduled_hour
    is_after_next_start = now >= next_start_time

    if is_scheduled_day and is_scheduled_hour and is_after_next_start:
        days_since_last = (now - last_created_at).days
        logger.info(
            f"{tournament_type.value} tournament - Days since last: {days_since_last}, "
            f"currently in scheduled window ({next_start_time.strftime('%Y-%m-%d %H:%M UTC')}) - ready to start"
        )
        return True
    else:
        time_until_start = (next_start_time - now).total_seconds() / 3600
        if is_after_next_start and not (is_scheduled_day and is_scheduled_hour):
            next_next_start = _calculate_next_tournament_start_time(now, tournament_type)
            time_until_start = (next_next_start - now).total_seconds() / 3600
            logger.info(
                f"{tournament_type.value} tournament - Missed scheduled window, "
                f"next opportunity: {next_next_start.strftime('%Y-%m-%d %H:%M UTC')} "
                f"(in {time_until_start:.2f} hours)"
            )
        else:
            logger.info(
                f"{tournament_type.value} tournament - Next scheduled start: {next_start_time.strftime('%Y-%m-%d %H:%M UTC')} "
                f"(in {time_until_start:.2f} hours) - waiting for scheduled time"
            )
        return False


def _get_tournament_schedule(tournament_type: TournamentType) -> tuple[int, int]:
    """
    Get the scheduled day of week and hour for a specific tournament type.

    Returns (day_of_week, hour) where day_of_week is 0=Monday through 6=Sunday.
    """
    if tournament_type == TournamentType.ENVIRONMENT:
        return (cst.TOURNAMENT_SCHEDULE_ENVIRONMENT_DAY_OF_WEEK, cst.TOURNAMENT_SCHEDULE_ENVIRONMENT_HOUR)
    elif tournament_type == TournamentType.TEXT:
        return (cst.TOURNAMENT_SCHEDULE_TEXT_DAY_OF_WEEK, cst.TOURNAMENT_SCHEDULE_TEXT_HOUR)
    elif tournament_type == TournamentType.IMAGE:
        return (cst.TOURNAMENT_SCHEDULE_IMAGE_DAY_OF_WEEK, cst.TOURNAMENT_SCHEDULE_IMAGE_HOUR)
    else:
        # Default fallback
        return (0, 14)


def _calculate_next_tournament_start_time(last_created_at: datetime, tournament_type: TournamentType) -> datetime:
    """
    Calculate the next valid tournament start time based on configured day of week and hour for the tournament type.

    Returns the next occurrence of the scheduled day/hour that is after last_created_at.
    """
    # Get current time for reference
    now = datetime.now(timezone.utc)

    # Use the later of last_created_at or now as our reference point
    reference_time = max(last_created_at, now)

    target_weekday, target_hour = _get_tournament_schedule(tournament_type)

    # Find the next occurrence of the target day
    current_weekday = reference_time.weekday()
    days_until_target = (target_weekday - current_weekday) % 7

    # If days_until_target is 0, we're on the target day - check if time has passed
    if days_until_target == 0:
        candidate = reference_time.replace(hour=target_hour, minute=0, second=0, microsecond=0)
        if reference_time.hour == target_hour:
            return candidate
        if candidate > reference_time:
            return candidate
        # Time has passed today, schedule for next week
        days_until_target = 7

    next_target_day = reference_time + timedelta(days=days_until_target)
    next_start = next_target_day.replace(hour=target_hour, minute=0, second=0, microsecond=0)

    return next_start


async def get_final_round_participants(completed_round: TournamentRoundData, psql_db: PSQLDB) -> tuple[str, str]:
    if completed_round.round_type == RoundType.KNOCKOUT:
        pairs = await get_tournament_pairs(completed_round.round_id, psql_db)
        if not pairs:
            raise ValueError(f"No pairs found for final round {completed_round.round_id}")

        pair = pairs[0]

        return pair.hotkey1, pair.hotkey2
    else:
        raise ValueError(f"Expected a knockout round, got {completed_round.round_type}")


async def upload_participant_repository(
    tournament_id: str, tournament_type: str, hotkey: str, position: int, config: Config, psql_db: PSQLDB
):
    logger.info(f"Uploading repository for tournament participant: {hotkey} (position: {position})")

    participant = await get_tournament_participant(tournament_id, hotkey, psql_db)

    if not participant or not participant.training_repo:
        logger.warning(f"No training repository found for participant {hotkey}")
        return None

    backup_repo_url = await upload_tournament_participant_repository(
        tournament_id=tournament_id,
        tournament_type=tournament_type,
        participant_hotkey=hotkey,
        training_repo=participant.training_repo,
        commit_hash=participant.training_commit_hash or "",
        config=config,
        position=position,
    )

    if backup_repo_url:
        await update_tournament_participant_backup_repo(tournament_id, hotkey, backup_repo_url, psql_db)
        logger.info(f"Successfully stored backup repository URL for {hotkey}: {backup_repo_url}")
    else:
        logger.warning(f"Repository upload failed for participant {hotkey}")

    return backup_repo_url
