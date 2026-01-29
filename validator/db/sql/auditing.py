import json

from asyncpg import Connection
from fastapi import Depends
from fastapi import HTTPException
from loguru import logger  # noqa

from core.models.utility_models import ImageTextPair
from core.models.utility_models import RewardFunction
from core.models.utility_models import TaskType
from validator.core.config import Config
from validator.core.dependencies import get_config
from validator.core.models import AnyTypeTask
from validator.core.models import AnyTypeTaskWithHotkeyDetails
from validator.core.models import ChatTask
from validator.core.models import ChatTaskWithHotkeyDetails
from validator.core.models import DpoTask
from validator.core.models import EnvTask
from validator.core.models import EnvTaskWithHotkeyDetails
from validator.core.models import DpoTaskWithHotkeyDetails
from validator.core.models import GrpoTask
from validator.core.models import GrpoTaskWithHotkeyDetails
from validator.core.models import HotkeyDetails
from validator.core.models import ImageTask
from validator.core.models import ImageTaskWithHotkeyDetails
from validator.core.models import InstructTextTask
from validator.core.models import InstructTextTaskWithHotkeyDetails
from validator.db import constants as cst
from validator.db.sql import tasks as tasks_sql
from validator.utils.util import hide_sensitive_data_till_finished
from validator.utils.util import normalise_float


async def get_recent_tasks(
    hotkeys: list[str] | None = None,
    limit: int = 100,
    page: int = 1,
    config: Config = Depends(get_config),
    include_tournament_tasks=False,
) -> list[AnyTypeTask]:
    tournament_tasks_clause = (
        "" if include_tournament_tasks else f"WHERE {cst.TASK_ID} NOT IN (SELECT {cst.TASK_ID} FROM {cst.TOURNAMENT_TASKS_TABLE})"
    )
    tournament_tasks_clause_hotkeys = (
        "" if include_tournament_tasks else f"AND {cst.TASK_ID} NOT IN (SELECT {cst.TASK_ID} FROM {cst.TOURNAMENT_TASKS_TABLE})"
    )

    # Always exclude benchmark tasks from auditing
    benchmark_tasks_clause = f"""
        AND {cst.TASK_ID} NOT IN (
            SELECT {cst.TASK_ID} FROM {cst.BENCHMARK_ROOT_TASKS_TABLE}
            UNION
            SELECT {cst.COPY_TASK_ID} FROM {cst.BENCHMARK_TASK_COPIES_TABLE}
        )
    """

    async with await config.psql_db.connection() as connection:
        connection: Connection
        base_query = f"""
        WITH task_ids AS (
            {
            f'''
                SELECT DISTINCT s.{cst.TASK_ID}
                FROM {cst.SUBMISSIONS_TABLE} s
                WHERE s.{cst.HOTKEY} = ANY($1)
                {tournament_tasks_clause_hotkeys}
                {benchmark_tasks_clause}
                ORDER BY s.{cst.CREATED_ON} DESC
                LIMIT $2 OFFSET $3
                '''
            if hotkeys is not None
            else f'''
                SELECT {cst.TASK_ID}
                FROM {cst.TASKS_TABLE}
                {tournament_tasks_clause}
                {benchmark_tasks_clause}
                ORDER BY {cst.CREATED_AT} DESC
                LIMIT $1 OFFSET $2
                '''
        }
        ),
        image_pairs AS (
            SELECT
                itp.{cst.TASK_ID},
                ARRAY_AGG(json_build_object(
                    'image_url', itp.{cst.IMAGE_URL},
                    'text_url', itp.{cst.TEXT_URL}
                ) ORDER BY itp.{cst.ID}) as image_text_pairs
            FROM task_ids
            JOIN {cst.IMAGE_TEXT_PAIRS_TABLE} itp ON task_ids.{cst.TASK_ID} = itp.{cst.TASK_ID}
            GROUP BY itp.{cst.TASK_ID}
        ),
        reward_functions AS (
            SELECT
                gtf.{cst.TASK_ID},
                ARRAY_AGG(json_build_object(
                    'reward_func', rf.{cst.REWARD_FUNC},
                    'func_hash', rf.{cst.FUNC_HASH},
                    'is_generic', rf.{cst.IS_GENERIC},
                    'reward_weight', gtf.{cst.REWARD_WEIGHT}
                )::text) as reward_functions
            FROM task_ids
            JOIN {cst.GRPO_TASK_FUNCTIONS_TABLE} gtf ON task_ids.{cst.TASK_ID} = gtf.{cst.TASK_ID}
            JOIN {cst.REWARD_FUNCTIONS_TABLE} rf ON rf.{cst.REWARD_ID} = gtf.{cst.REWARD_ID}
            GROUP BY gtf.{cst.TASK_ID}
        )
        -- Main query joining all necessary tables
        SELECT
            t.*,
            itt.field_system as itt_field_system,
            itt.field_instruction,
            itt.field_input,
            itt.field_output,
            itt.format as itt_format,
            itt.no_input_format,
            itt.file_format as itt_file_format,
            it.model_type,
            ip.image_text_pairs,
            dt.field_prompt as dpo_field_prompt,
            dt.field_chosen,
            dt.field_rejected,
            dt.prompt_format,
            dt.chosen_format,
            dt.rejected_format,
            dt.file_format as dpo_file_format,
            gt.field_prompt as grpo_field_prompt,
            gt.file_format as grpo_file_format,
            rf.reward_functions,
            ct.chat_template,
            ct.chat_column,
            ct.chat_role_field,
            ct.chat_content_field,
            ct.chat_user_reference,
            ct.chat_assistant_reference,
            ct.file_format as chat_file_format
        FROM task_ids
        JOIN {cst.TASKS_TABLE} t ON t.{cst.TASK_ID} = task_ids.{cst.TASK_ID}
        LEFT JOIN {cst.INSTRUCT_TEXT_TASKS_TABLE} itt ON t.{cst.TASK_ID} = itt.{cst.TASK_ID}
        LEFT JOIN {cst.IMAGE_TASKS_TABLE} it ON t.{cst.TASK_ID} = it.{cst.TASK_ID}
        LEFT JOIN image_pairs ip ON t.{cst.TASK_ID} = ip.{cst.TASK_ID}
        LEFT JOIN {cst.DPO_TASKS_TABLE} dt ON t.{cst.TASK_ID} = dt.{cst.TASK_ID}
        LEFT JOIN {cst.GRPO_TASKS_TABLE} gt ON t.{cst.TASK_ID} = gt.{cst.TASK_ID}
        LEFT JOIN {cst.CHAT_TASKS_TABLE} ct ON t.{cst.TASK_ID} = ct.{cst.TASK_ID}
        LEFT JOIN reward_functions rf ON t.{cst.TASK_ID} = rf.{cst.TASK_ID}
        """

        if hotkeys is not None:
            rows = await connection.fetch(base_query, hotkeys, limit, (page - 1) * limit)
        else:
            rows = await connection.fetch(base_query, limit, (page - 1) * limit)

        tasks_processed = []
        for row in rows:
            task_data = dict(row)
            task_type = task_data[cst.TASK_TYPE]

            if task_type == TaskType.INSTRUCTTEXTTASK.value:
                task_data["field_system"] = task_data.pop("itt_field_system")
                task_data["format"] = task_data.pop("itt_format")
                task_data["file_format"] = task_data.pop("itt_file_format")
                task = InstructTextTask(**{k: v for k, v in task_data.items() if k in InstructTextTask.model_fields})
            elif task_type == TaskType.IMAGETASK.value:
                image_text_pairs = task_data.pop("image_text_pairs") or []
                if isinstance(image_text_pairs, str):
                    try:
                        image_text_pairs = json.loads(image_text_pairs)
                    except json.JSONDecodeError:
                        image_text_pairs = []
                elif isinstance(image_text_pairs, list):
                    try:
                        image_text_pairs = [
                            ImageTextPair(**pair) if isinstance(pair, dict) else ImageTextPair(**json.loads(pair))
                            for pair in image_text_pairs
                        ]
                    except json.JSONDecodeError:
                        image_text_pairs = []

                task = ImageTask(
                    **{k: v for k, v in task_data.items() if k in ImageTask.model_fields}, image_text_pairs=image_text_pairs
                )
            elif task_type == TaskType.DPOTASK.value:
                task_data["field_prompt"] = task_data.pop("dpo_field_prompt")
                task_data["file_format"] = task_data.pop("dpo_file_format")
                task = DpoTask(**{k: v for k, v in task_data.items() if k in DpoTask.model_fields})
            elif task_type == TaskType.ENVIRONMENTTASK.value:
                task_data["environment_name"] = task_data.pop("environment_name")
                task = EnvTask(**{k: v for k, v in task_data.items() if k in EnvTask.model_fields})
            elif task_type == TaskType.GRPOTASK.value:
                task_data["field_prompt"] = task_data.pop("grpo_field_prompt")
                task_data["file_format"] = task_data.pop("grpo_file_format")

                reward_functions = []
                if task_data.get("reward_functions"):
                    for rf_str in task_data["reward_functions"]:
                        try:
                            rf_dict = json.loads(rf_str)
                            reward_functions.append(RewardFunction(**rf_dict))
                        except (json.JSONDecodeError, TypeError):
                            continue
                task_data["reward_functions"] = reward_functions

                task = GrpoTask(**{k: v for k, v in task_data.items() if k in GrpoTask.model_fields})
            elif task_type == TaskType.CHATTASK.value:
                task_data["file_format"] = task_data.pop("chat_file_format")
                task = ChatTask(**{k: v for k, v in task_data.items() if k in ChatTask.model_fields})
            else:
                logger.warning(f"Unknown task type: {task_type}, skipping task {task_data.get('task_id')}")
                continue

            task = hide_sensitive_data_till_finished(task)
            tasks_processed.append(task)

        return tasks_processed


async def _process_task_batch(
    connection, hotkey: str, task_ids: list[str], include_tournament_tasks=False
) -> list[AnyTypeTaskWithHotkeyDetails]:
    """
    Helper function to process a batch of task IDs.
    """
    tournament_tasks_clause = (
        "" if include_tournament_tasks else f"AND {cst.TASK_ID} NOT IN (SELECT {cst.TASK_ID} FROM {cst.TOURNAMENT_TASKS_TABLE})"
    )

    # Always exclude benchmark tasks from auditing
    benchmark_tasks_clause = f"""
        AND {cst.TASK_ID} NOT IN (
            SELECT {cst.TASK_ID} FROM {cst.BENCHMARK_ROOT_TASKS_TABLE}
            UNION
            SELECT {cst.COPY_TASK_ID} FROM {cst.BENCHMARK_TASK_COPIES_TABLE}
        )
    """

    tasks_with_details = []

    tasks_by_id = {}
    if task_ids:
        task_placeholders = ", ".join("$%d::uuid" % (i + 1) for i in range(len(task_ids)))
        tasks_query = f"""
            SELECT
                t.*
            FROM
                {cst.TASKS_TABLE} t
            WHERE
                t.{cst.TASK_ID} IN ({task_placeholders})
                {tournament_tasks_clause}
                {benchmark_tasks_clause}
        """

        tasks_rows = await connection.fetch(tasks_query, *task_ids)

        tasks_by_id = {str(row[cst.TASK_ID]): dict(row) for row in tasks_rows}
    else:
        return []

    # Step 3: Get all hotkey-specific details for these tasks in a single query
    details_rows = []
    if task_ids:
        details_placeholders = ", ".join("$%d::uuid" % (i + 2) for i in range(len(task_ids)))
        details_query = f"""
            SELECT
                t.{cst.TASK_ID}::text AS task_id,
                s.{cst.SUBMISSION_ID} AS submission_id,
                tn.{cst.QUALITY_SCORE} AS quality_score,
                tn.{cst.TEST_LOSS} AS test_loss,
                tn.{cst.SYNTH_LOSS} AS synth_loss,
                tn.{cst.SCORE_REASON} AS score_reason,
                RANK() OVER (PARTITION BY t.{cst.TASK_ID} ORDER BY tn.{cst.QUALITY_SCORE} DESC) AS rank,
                s.{cst.REPO} AS repo,
                o.{cst.OFFER_RESPONSE} AS offer_response,
                t.{cst.TASK_TYPE} AS task_type
            FROM
                {cst.TASKS_TABLE} t
            LEFT JOIN
                {cst.TASK_NODES_TABLE} tn ON t.{cst.TASK_ID} = tn.{cst.TASK_ID} AND tn.{cst.HOTKEY} = $1
            LEFT JOIN
                {cst.SUBMISSIONS_TABLE} s ON t.{cst.TASK_ID} = s.{cst.TASK_ID} AND s.{cst.HOTKEY} = $1
            LEFT JOIN
                {cst.OFFER_RESPONSES_TABLE} o ON t.{cst.TASK_ID} = o.{cst.TASK_ID} AND o.{cst.HOTKEY} = $1
            WHERE
                t.{cst.TASK_ID} IN ({details_placeholders})
        """

        details_rows = await connection.fetch(details_query, hotkey, *task_ids)

    # Step 4: Group details by task_id
    details_by_task_id = {}
    for row in details_rows:
        task_id = row["task_id"]
        if task_id not in details_by_task_id:
            details_by_task_id[task_id] = []

        detail = dict(row)

        if detail.get("offer_response"):
            try:
                detail["offer_response"] = json.loads(detail["offer_response"])
            except (json.JSONDecodeError, TypeError):
                detail["offer_response"] = None

        for field in ["quality_score", "test_loss", "synth_loss"]:
            if detail.get(field) is not None:
                detail[field] = normalise_float(detail[field])

        details_by_task_id[task_id].append(detail)

    # Step 5: Get type-specific data for each task type
    instruct_text_task_ids = []
    image_task_ids = []
    dpo_task_ids = []
    grpo_task_ids = []
    chat_task_ids = []
    env_task_ids = []

    for task_id, task_data in tasks_by_id.items():
        task_type = task_data.get(cst.TASK_TYPE)
        if task_type == TaskType.INSTRUCTTEXTTASK.value:
            instruct_text_task_ids.append(task_id)
        elif task_type == TaskType.IMAGETASK.value:
            image_task_ids.append(task_id)
        elif task_type == TaskType.DPOTASK.value:
            dpo_task_ids.append(task_id)
        elif task_type == TaskType.GRPOTASK.value:
            grpo_task_ids.append(task_id)
        elif task_type == TaskType.ENVIRONMENTTASK.value:
            env_task_ids.append(task_id)
        elif task_type == TaskType.CHATTASK.value:
            chat_task_ids.append(task_id)

    # Get all InstructTextTask specific data in one query
    instruct_text_task_data = {}
    if instruct_text_task_ids:
        placeholders = ", ".join("$%d::uuid" % (i + 1) for i in range(len(instruct_text_task_ids)))
        query = f"""
            SELECT * FROM {cst.INSTRUCT_TEXT_TASKS_TABLE}
            WHERE {cst.TASK_ID} IN ({placeholders})
        """
        rows = await connection.fetch(query, *instruct_text_task_ids)
        instruct_text_task_data = {str(row[cst.TASK_ID]): dict(row) for row in rows}

    # Get all ChatTask specific data in one query
    chat_task_data = {}
    if chat_task_ids:
        placeholders = ", ".join("$%d::uuid" % (i + 1) for i in range(len(chat_task_ids)))
        query = f"""
            SELECT * FROM {cst.CHAT_TASKS_TABLE}
            WHERE {cst.TASK_ID} IN ({placeholders})
        """
        rows = await connection.fetch(query, *chat_task_ids)
        chat_task_data = {str(row[cst.TASK_ID]): dict(row) for row in rows}

    # Get all ImageTask specific data in one query
    image_task_data = {}
    if image_task_ids:
        placeholders = ", ".join("$%d::uuid" % (i + 1) for i in range(len(image_task_ids)))
        query = f"""
            SELECT * FROM {cst.IMAGE_TASKS_TABLE}
            WHERE {cst.TASK_ID} IN ({placeholders})
        """
        rows = await connection.fetch(query, *image_task_ids)
        image_task_data = {str(row[cst.TASK_ID]): dict(row) for row in rows}

    # Get all DpoTask specific data in one query
    dpo_task_data = {}
    if dpo_task_ids:
        placeholders = ", ".join("$%d::uuid" % (i + 1) for i in range(len(dpo_task_ids)))
        query = f"""
            SELECT * FROM {cst.DPO_TASKS_TABLE}
            WHERE {cst.TASK_ID} IN ({placeholders})
        """
        rows = await connection.fetch(query, *dpo_task_ids)
        dpo_task_data = {str(row[cst.TASK_ID]): dict(row) for row in rows}

    # Get all EnvTask specific data in one query
    env_task_data = {}
    if env_task_ids:
        placeholders = ", ".join("$%d::uuid" % (i + 1) for i in range(len(env_task_ids)))
        query = f"""
            SELECT {cst.TASK_ID}, {cst.ENVIRONMENT_NAME} FROM {cst.ENV_TASKS_TABLE}
            WHERE {cst.TASK_ID} IN ({placeholders})
        """
        rows = await connection.fetch(query, *env_task_ids)
        env_task_data = {str(row[cst.TASK_ID]): dict(row) for row in rows}

    # Get all GrpoTask specific data in one query
    grpo_task_data = {}
    if grpo_task_ids:
        placeholders = ", ".join("$%d::uuid" % (i + 1) for i in range(len(grpo_task_ids)))
        query = f"""
            SELECT * FROM {cst.GRPO_TASKS_TABLE}
            WHERE {cst.TASK_ID} IN ({placeholders})
        """
        rows = await connection.fetch(query, *grpo_task_ids)
        grpo_task_data = {str(row[cst.TASK_ID]): dict(row) for row in rows}

        # Fetch reward functions for each GRPO task
        for task_id in grpo_task_ids:
            reward_functions_query = f"""
                SELECT rf.{cst.REWARD_FUNC}, rf.{cst.FUNC_HASH}, rf.{cst.IS_GENERIC}, gtf.{cst.REWARD_WEIGHT}
                FROM {cst.REWARD_FUNCTIONS_TABLE} rf
                JOIN {cst.GRPO_TASK_FUNCTIONS_TABLE} gtf ON rf.{cst.REWARD_ID} = gtf.{cst.REWARD_ID}
                WHERE gtf.{cst.TASK_ID} = $1
            """
            reward_rows = await connection.fetch(reward_functions_query, task_id)
            reward_functions = [
                RewardFunction(
                    reward_func=row[cst.REWARD_FUNC],
                    func_hash=row[cst.FUNC_HASH],
                    is_generic=row[cst.IS_GENERIC],
                    reward_weight=row[cst.REWARD_WEIGHT],
                )
                for row in reward_rows
            ]

            if task_id in grpo_task_data:
                grpo_task_data[task_id]["reward_functions"] = reward_functions

    # Step 6: Assemble final results
    for task_id in task_ids:
        if task_id not in tasks_by_id:
            continue

        task_data = tasks_by_id[task_id].copy()
        task_type = task_data.get(cst.TASK_TYPE)

        if task_type == TaskType.INSTRUCTTEXTTASK.value and task_id in instruct_text_task_data:
            task_data.update(instruct_text_task_data[task_id])
        elif task_type == TaskType.IMAGETASK.value and task_id in image_task_data:
            task_data.update(image_task_data[task_id])
        elif task_type == TaskType.DPOTASK.value and task_id in dpo_task_data:
            task_data.update(dpo_task_data[task_id])
        elif task_type == TaskType.GRPOTASK.value and task_id in grpo_task_data:
            task_data.update(grpo_task_data[task_id])
        elif task_type == TaskType.ENVIRONMENTTASK.value and task_id in env_task_data:
            task_data.update(env_task_data[task_id])
        elif task_type == TaskType.CHATTASK.value and task_id in chat_task_data:
            task_data.update(chat_task_data[task_id])

        hotkey_details = []
        if task_id in details_by_task_id:
            for detail in details_by_task_id[task_id]:
                hotkey_details.append(
                    HotkeyDetails(
                        hotkey=hotkey,
                        submission_id=detail.get("submission_id"),
                        quality_score=detail.get("quality_score"),
                        test_loss=detail.get("test_loss"),
                        synth_loss=detail.get("synth_loss"),
                        score_reason=detail.get("score_reason"),
                        rank=detail.get("rank"),
                        repo=detail.get("repo"),
                        offer_response=detail.get("offer_response"),
                    )
                )

        if task_type == TaskType.INSTRUCTTEXTTASK.value:
            task_fields = {k: v for k, v in task_data.items() if k in InstructTextTask.model_fields}
            task = InstructTextTask(**task_fields)
            task = hide_sensitive_data_till_finished(task)
            tasks_with_details.append(InstructTextTaskWithHotkeyDetails(**task.model_dump(), hotkey_details=hotkey_details))
        elif task_type == TaskType.CHATTASK.value:
            task_fields = {k: v for k, v in task_data.items() if k in ChatTask.model_fields}
            task = ChatTask(**task_fields)
            task = hide_sensitive_data_till_finished(task)
            tasks_with_details.append(ChatTaskWithHotkeyDetails(**task.model_dump(), hotkey_details=hotkey_details))
        elif task_type == TaskType.IMAGETASK.value:
            task_fields = {k: v for k, v in task_data.items() if k in ImageTask.model_fields}
            task = ImageTask(**task_fields)
            task = hide_sensitive_data_till_finished(task)
            tasks_with_details.append(ImageTaskWithHotkeyDetails(**task.model_dump(), hotkey_details=hotkey_details))
        elif task_type == TaskType.DPOTASK.value:
            task_fields = {k: v for k, v in task_data.items() if k in DpoTask.model_fields}
            task = DpoTask(**task_fields)
            task = hide_sensitive_data_till_finished(task)
            tasks_with_details.append(DpoTaskWithHotkeyDetails(**task.model_dump(), hotkey_details=hotkey_details))
        elif task_type == TaskType.GRPOTASK.value:
            task_fields = {k: v for k, v in task_data.items() if k in GrpoTask.model_fields}
            task = GrpoTask(**task_fields)
            task = hide_sensitive_data_till_finished(task)
            tasks_with_details.append(GrpoTaskWithHotkeyDetails(**task.model_dump(), hotkey_details=hotkey_details))
        elif task_type == TaskType.ENVIRONMENTTASK.value:
            task_fields = {k: v for k, v in task_data.items() if k in EnvTask.model_fields}
            task = EnvTask(**task_fields)
            task = hide_sensitive_data_till_finished(task)
            tasks_with_details.append(EnvTaskWithHotkeyDetails(**task.model_dump(), hotkey_details=hotkey_details))

    return tasks_with_details


async def get_recent_tasks_for_hotkey(
    hotkey: str, limit: int = 100, page: int = 1, config: Config = Depends(get_config), include_tournament_tasks=False
) -> list[AnyTypeTaskWithHotkeyDetails]:
    """
    Retrieves recent tasks for a specific hotkey with detailed information.
    """
    MAX_BATCH_SIZE = 500
    tournament_tasks_clause = (
        "" if include_tournament_tasks else f"AND {cst.TASK_ID} NOT IN (SELECT {cst.TASK_ID} FROM {cst.TOURNAMENT_TASKS_TABLE})"
    )

    # Always exclude benchmark tasks from auditing
    benchmark_tasks_clause = f"""
        AND {cst.TASK_ID} NOT IN (
            SELECT {cst.TASK_ID} FROM {cst.BENCHMARK_ROOT_TASKS_TABLE}
            UNION
            SELECT {cst.COPY_TASK_ID} FROM {cst.BENCHMARK_TASK_COPIES_TABLE}
        )
    """

    async with await config.psql_db.connection() as connection:
        task_ids_query = f"""
            SELECT
                s.{cst.TASK_ID}::text AS task_id
            FROM
                {cst.SUBMISSIONS_TABLE} s
            WHERE
                s.{cst.HOTKEY} = $1
                {tournament_tasks_clause}
                {benchmark_tasks_clause}
            ORDER BY
                s.{cst.CREATED_ON} DESC
            LIMIT $2 OFFSET $3
        """
        offset = (page - 1) * limit
        task_ids_rows = await connection.fetch(task_ids_query, hotkey, limit, offset)

        if not task_ids_rows:
            return []

        task_ids = [row["task_id"] for row in task_ids_rows]

        if len(task_ids) > MAX_BATCH_SIZE:
            all_results = []
            for i in range(0, len(task_ids), MAX_BATCH_SIZE):
                batch_ids = task_ids[i : i + MAX_BATCH_SIZE]
                batch_results = await _process_task_batch(connection, hotkey, batch_ids, include_tournament_tasks)
                all_results.extend(batch_results)
            return all_results

        return await _process_task_batch(connection, hotkey, task_ids, include_tournament_tasks)


async def get_task_with_hotkey_details(task_id: str, config: Config = Depends(get_config)) -> AnyTypeTaskWithHotkeyDetails:
    # First get all the task details like normal
    task_raw = await tasks_sql.get_task_by_id(task_id, config.psql_db)
    if task_raw is None:
        raise HTTPException(status_code=404, detail="Task not found")

    logger.info("Got a task!!")

    task = hide_sensitive_data_till_finished(task_raw)

    query = f"""
        SELECT
            tn.{cst.HOTKEY},
            s.{cst.SUBMISSION_ID},
            tn.{cst.QUALITY_SCORE},
            tn.{cst.TEST_LOSS},
            tn.{cst.SYNTH_LOSS},
            tn.{cst.SCORE_REASON},
            RANK() OVER (ORDER BY tn.{cst.QUALITY_SCORE} DESC) as rank,
            s.{cst.REPO},
            o.{cst.OFFER_RESPONSE}
        FROM {cst.TASK_NODES_TABLE} tn
        LEFT JOIN {cst.SUBMISSIONS_TABLE} s
            ON tn.{cst.TASK_ID} = s.{cst.TASK_ID}
            AND tn.{cst.HOTKEY} = s.{cst.HOTKEY}
        LEFT JOIN {cst.OFFER_RESPONSES_TABLE} o
            ON tn.{cst.TASK_ID} = o.{cst.TASK_ID}
            AND tn.{cst.HOTKEY} = o.{cst.HOTKEY}
        WHERE tn.{cst.TASK_ID} = $1
    """
    async with await config.psql_db.connection() as connection:
        connection: Connection
        results = await connection.fetch(query, task_id)

    logger.info(f"Got {len(results)} results for task {task_id}")

    hotkey_details = []
    for result in results:
        result_dict = dict(result)
        if result_dict[cst.OFFER_RESPONSE] is not None:
            result_dict[cst.OFFER_RESPONSE] = json.loads(result_dict[cst.OFFER_RESPONSE])

        float_fields = [cst.QUALITY_SCORE, cst.TEST_LOSS, cst.SYNTH_LOSS]
        for field in float_fields:
            result_dict[field] = normalise_float(result_dict[field])

        hotkey_details.append(HotkeyDetails(**result_dict))

    if task.task_type == TaskType.INSTRUCTTEXTTASK:
        return InstructTextTaskWithHotkeyDetails(**task.model_dump(), hotkey_details=hotkey_details)
    elif task.task_type == TaskType.IMAGETASK:
        return ImageTaskWithHotkeyDetails(**task.model_dump(), hotkey_details=hotkey_details)
    elif task.task_type == TaskType.DPOTASK:
        return DpoTaskWithHotkeyDetails(**task.model_dump(), hotkey_details=hotkey_details)
    elif task.task_type == TaskType.GRPOTASK:
        return GrpoTaskWithHotkeyDetails(**task.model_dump(), hotkey_details=hotkey_details)
    elif task.task_type == TaskType.ENVIRONMENTTASK:
        return EnvTaskWithHotkeyDetails(**task.model_dump(), hotkey_details=hotkey_details)
    elif task.task_type == TaskType.CHATTASK:
        return ChatTaskWithHotkeyDetails(**task.model_dump(), hotkey_details=hotkey_details)


async def store_latest_scores_url(url: str, config: Config = Depends(get_config)) -> None:
    async with await config.psql_db.connection() as connection:
        connection: Connection

        # First expire all existing URLs
        expire_query = f"""
            UPDATE {cst.LATEST_SCORES_URL_TABLE}
            SET expired_at = NOW()
            WHERE expired_at IS NULL
        """
        await connection.execute(expire_query)

        # Then insert the new URL
        insert_query = f"""
            INSERT INTO {cst.LATEST_SCORES_URL_TABLE} (url)
            VALUES ($1)
        """
        await connection.execute(insert_query, url)


async def get_latest_scores_url(config: Config = Depends(get_config)) -> str | None:
    async with await config.psql_db.connection() as connection:
        connection: Connection

        query = f"""
            SELECT url FROM {cst.LATEST_SCORES_URL_TABLE} WHERE expired_at IS NULL ORDER BY created_at DESC LIMIT 1
        """
        return await connection.fetchval(query)
