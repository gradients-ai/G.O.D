from typing import Any

from substrateinterface import SubstrateInterface

from core.logging import get_logger


logger = get_logger(__name__)


def query_substrate(substrate: SubstrateInterface, module: str, method: str, params: list[Any], return_value: bool = True) -> Any:
    try:
        query_result = substrate.query(module, method, params)

        return_val = query_result.value if return_value else query_result

        return substrate, return_val
    except Exception:
        substrate = SubstrateInterface(url=substrate.url)

        query_result = substrate.query(module, method, params)

        return_val = query_result.value if return_value else query_result

        return substrate, return_val
