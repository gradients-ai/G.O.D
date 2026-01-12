from uuid import UUID

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException

from core.models.payload_models import AddRewardFunctionRequest
from core.models.payload_models import RewardFunctionInfo
from core.models.payload_models import RewardFunctionsResponse
from validator.core.config import Config
from validator.core.dependencies import get_api_key
from validator.core.dependencies import get_config
from validator.db.sql import grpo as grpo_sql
from validator.utils.logging import get_logger
from validator.utils.reward_functions import extract_docstring
from validator.utils.reward_functions import extract_function_name
from validator.utils.reward_functions import process_reward_function_code


logger = get_logger(__name__)


REWARD_FUNCTIONS_ENDPOINT = "/v1/grpo/reward_functions"


async def get_reward_functions(
    config: Config = Depends(get_config),
) -> RewardFunctionsResponse:
    """Get all GRPO reward functions from the database.

    Returns a dictionary mapping function names to their details including
    description and code.
    """
    reward_functions = await grpo_sql.get_all_reward_functions(config.psql_db)

    result = {}
    for rf in reward_functions:
        func_name = extract_function_name(rf.reward_func)
        description = extract_docstring(rf.reward_func)

        result[func_name] = RewardFunctionInfo(
            reward_id=rf.reward_id, name=func_name, description=description, code=rf.reward_func
        )

    return RewardFunctionsResponse(reward_functions=result)


async def add_reward_function(
    request: AddRewardFunctionRequest,
    config: Config = Depends(get_config),
) -> dict[str, UUID]:
    """Add a new GRPO reward function to the database.

    Args:
        request: Contains name, description, code, and optional reward_weight

    Returns:
        Dictionary with the created reward function ID
    """
    try:
        logger.error(f"ENDPOINT DEBUG: About to process code for '{request.name}', code length: {len(request.code)}")
        logger.error(f"ENDPOINT DEBUG: Code contains 'restricted_execution': {'restricted_execution' in request.code}")
        
        code_to_store = process_reward_function_code(request.code)
        
        logger.error(f"ENDPOINT DEBUG: Processed code length: {len(code_to_store)}")
        logger.error(f"ENDPOINT DEBUG: Processed code contains 'def restricted_execution': {'def restricted_execution' in code_to_store}")

        reward_id = await grpo_sql.add_reward_function(
            config.psql_db,
            name=request.name,
            description=request.description,
            code=code_to_store,
            reward_weight=request.reward_weight,
        )

        logger.info(f"Successfully added reward function '{request.name}' with ID {reward_id}")
        return {"reward_id": reward_id}

    except Exception as e:
        logger.error(f"Failed to add reward function: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add reward function: {str(e)}")


def factory_router() -> APIRouter:
    router = APIRouter(tags=["GRPO"], dependencies=[Depends(get_api_key)])
    router.add_api_route(REWARD_FUNCTIONS_ENDPOINT, get_reward_functions, methods=["GET"])
    router.add_api_route(REWARD_FUNCTIONS_ENDPOINT, add_reward_function, methods=["POST"])
    return router
