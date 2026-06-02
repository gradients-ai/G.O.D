import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fiber.logging_utils import get_logger
from fiber.miner.core import configuration

from miner.endpoints.training_repo import factory_router as tuning_factory_router


logger = get_logger(__name__)


def factory_app(debug: bool = False) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        config = configuration.factory_config()
        metagraph = config.metagraph
        sync_thread = None
        if metagraph.substrate is not None:
            sync_thread = threading.Thread(target=metagraph.periodically_sync_nodes, daemon=True)
            sync_thread.start()

        yield

        logger.info("Shutting down...")

        metagraph.shutdown()
        if metagraph.substrate is not None and sync_thread is not None:
            sync_thread.join()

    app = FastAPI(lifespan=lifespan, debug=debug)

    return app


logger = get_logger(__name__)

app = factory_app(debug=True)


tuning_router = tuning_factory_router()

app.include_router(tuning_router)

# if os.getenv("ENV", "prod").lower() == "dev":
#    configure_extra_logging_middleware(app)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=7999)
