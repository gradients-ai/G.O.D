from fastapi import APIRouter
from fastapi import Body
from fastapi import Depends
from fiber.chain.models import Node

from validator.app.config import Config
from validator.app.dependencies import get_config
from validator.db import sql
from core.logging import get_logger


logger = get_logger(__name__)


# This is only for dev purposes.
# It's very useful and we can keep it, but only enable if
# the env is not prod
async def add_node(
    coldkey: str = Body(..., embed=True),
    node_id: int = Body(..., embed=True),
    ip: str = Body(..., embed=True),
    ip_type: str = Body(..., embed=True),
    port: int = Body(..., embed=True),
    symmetric_key: str = Body(..., embed=True),
    network: float = Body(..., embed=True),
    stake: float = Body(..., embed=True),
    config: Config = Depends(get_config),
):
    node = Node(
        node_id=node_id,
        coldkey=coldkey,
        ip=ip,
        ip_type=ip_type,
        port=port,
        symmetric_key=symmetric_key,
        network=network,
        stake=stake,
    )
    node_id = await sql.add_node(node, config.psql_db)

    logger.info(f"Node {node_id} added.")
    return {"success": True, "node_id": node_id}


def factory_router() -> APIRouter:
    router = APIRouter()

    router.add_api_route(
        "/nodes/add",
        add_node,
        tags=["nodes"],
        methods=["POST"],
    )

    return router
