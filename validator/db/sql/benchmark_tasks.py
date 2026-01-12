from typing import List

from core.models.tournament_models import BenchmarkInstance
from core.models.tournament_models import BenchmarkTaskCopy
from core.models.tournament_models import BenchmarkTimeline
from core.models.tournament_models import BenchmarkTimelineResponse
from core.models.utility_models import TaskType
from validator.db import constants as cst
from validator.db.database import PSQLDB
from validator.db.sql.tournament_performance import get_task_scores_batch


async def get_all_benchmark_copies(psql_db: PSQLDB) -> List[BenchmarkTaskCopy]:
    """
    Get all benchmark task copies with their root task details.
    Filters out entries with null tournament_id.
    """
    async with await psql_db.connection() as connection:
        query = f"""
            SELECT 
                btc.{cst.COPY_TASK_ID}::text as copy_task_id,
                btc.{cst.ROOT_TASK_ID}::text as root_task_id,
                btc.{cst.PARTICIPANT_HOTKEY} as participant_hotkey,
                btc.{cst.TOURNAMENT_ID} as tournament_id,
                btc.{cst.CREATED_AT} as created_at,
                t.{cst.TASK_TYPE} as task_type,
                t.{cst.MODEL_ID} as model_id,
                t.{cst.DS} as dataset,
                t.{cst.HOURS_TO_COMPLETE} as hours_to_complete,
                t.{cst.MODEL_PARAMS_COUNT} as model_params_count,
                t.{cst.IS_ORGANIC} as is_organic,
                t.{cst.CREATED_AT} as task_created_at
            FROM {cst.BENCHMARK_TASK_COPIES_TABLE} btc
            JOIN {cst.BENCHMARK_ROOT_TASKS_TABLE} brt ON brt.{cst.TASK_ID} = btc.{cst.ROOT_TASK_ID}
            JOIN {cst.TASKS_TABLE} t ON brt.{cst.TASK_ID} = t.{cst.TASK_ID}
            WHERE btc.{cst.TOURNAMENT_ID} IS NOT NULL
            ORDER BY btc.{cst.ROOT_TASK_ID}, btc.{cst.CREATED_AT}
        """
        
        results = await connection.fetch(query)
        return [
            BenchmarkTaskCopy(
                copy_task_id=row['copy_task_id'],
                root_task_id=row['root_task_id'],
                participant_hotkey=row['participant_hotkey'],
                tournament_id=row['tournament_id'],
                created_at=row['created_at'],
                task_type=TaskType(row['task_type']),
                model_id=row['model_id'],
                dataset=row['dataset'],
                hours_to_complete=row['hours_to_complete'],
                model_params_count=row['model_params_count'],
                is_organic=row['is_organic'],
                task_created_at=row['task_created_at']
            )
            for row in results
        ]


async def get_benchmark_copies_by_tournament(tournament_id: str, psql_db: PSQLDB) -> List[BenchmarkTaskCopy]:
    """
    Get benchmark task copies filtered by tournament ID.
    """
    async with await psql_db.connection() as connection:
        query = f"""
            SELECT 
                btc.copy_task_id::text as copy_task_id,
                btc.root_task_id::text as root_task_id,
                btc.participant_hotkey as participant_hotkey,
                btc.tournament_id as tournament_id,
                btc.created_at as created_at,
                t.task_type as task_type,
                t.model_id as model_id,
                t.ds as dataset,
                t.hours_to_complete as hours_to_complete,
                t.model_params_count as model_params_count,
                t.is_organic as is_organic,
                t.created_at as task_created_at
            FROM {cst.BENCHMARK_TASK_COPIES_TABLE} btc
            JOIN {cst.BENCHMARK_ROOT_TASKS_TABLE} brt ON btc.root_task_id = brt.root_task_id
            JOIN {cst.TASKS_TABLE} t ON brt.root_task_id = t.task_id
            WHERE btc.{cst.TOURNAMENT_ID} = $1
            ORDER BY btc.{cst.ROOT_TASK_ID}, btc.{cst.CREATED_AT}
        """
        
        results = await connection.fetch(query, tournament_id)
        return [
            BenchmarkTaskCopy(
                copy_task_id=row['copy_task_id'],
                root_task_id=row['root_task_id'],
                participant_hotkey=row['participant_hotkey'],
                tournament_id=row['tournament_id'],
                created_at=row['created_at'],
                task_type=TaskType(row['task_type']),
                model_id=row['model_id'],
                dataset=row['dataset'],
                hours_to_complete=row['hours_to_complete'],
                model_params_count=row['model_params_count'],
                is_organic=row['is_organic'],
                task_created_at=row['task_created_at']
            )
            for row in results
        ]


async def build_benchmark_timelines(benchmark_copies: List[BenchmarkTaskCopy], psql_db: PSQLDB) -> List[BenchmarkTimeline]:
    """
    Build BenchmarkTimeline objects from benchmark copies data.
    """
    if not benchmark_copies:
        return []
    
    copy_task_ids = list(set(copy.copy_task_id for copy in benchmark_copies))
    
    task_results = await get_task_scores_batch(copy_task_ids, psql_db)
    
    timelines_dict = {}
    
    for copy in benchmark_copies:
        root_task_id = copy.root_task_id
        
        if root_task_id not in timelines_dict:
            timelines_dict[root_task_id] = BenchmarkTimeline(
                root_task_id=root_task_id,
                task_type=copy.task_type,
                model_id=copy.model_id,
                dataset=copy.dataset,
                hours_to_complete=copy.hours_to_complete,
                model_params_count=copy.model_params_count,
                is_organic=copy.is_organic,
                task_created_at=copy.task_created_at,
                benchmarks=[]
            )
        
        test_loss = None
        copy_results = task_results.get(copy.copy_task_id, [])
        for result in copy_results:
            if result.hotkey == copy.participant_hotkey:
                test_loss = result.test_loss
                break
        
        benchmark_instance = BenchmarkInstance(
            copy_task_id=copy.copy_task_id,
            participant_hotkey=copy.participant_hotkey,
            tournament_id=copy.tournament_id,
            created_at=copy.created_at,
            test_loss=test_loss
        )
        
        timelines_dict[root_task_id].benchmarks.append(benchmark_instance)
    
    return sorted(timelines_dict.values(), key=lambda x: x.root_task_id)


async def get_benchmark_timeline(psql_db: PSQLDB) -> BenchmarkTimelineResponse:
    """
    Get the complete benchmark timeline data.
    Groups benchmark task copies by root task and includes all evaluation results.
    """
    benchmark_copies = await get_all_benchmark_copies(psql_db)
    timelines = await build_benchmark_timelines(benchmark_copies, psql_db)
    return BenchmarkTimelineResponse(timelines=timelines)


async def get_benchmark_timeline_by_tournament(tournament_id: str, psql_db: PSQLDB) -> BenchmarkTimelineResponse:
    """
    Get benchmark timeline data filtered by tournament ID.
    """
    benchmark_copies = await get_benchmark_copies_by_tournament(tournament_id, psql_db)
    timelines = await build_benchmark_timelines(benchmark_copies, psql_db)
    return BenchmarkTimelineResponse(timelines=timelines)