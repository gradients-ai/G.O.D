import os

import fastapi
from fastapi import HTTPException
from fastapi import Security
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.security import HTTPBearer

from validator.app.config import Config


auth_scheme = HTTPBearer()


async def get_config(request: fastapi.Request) -> Config:
    config = request.app.state.config
    await config.psql_db.connect()

    return config


async def get_api_key(credentials: HTTPAuthorizationCredentials = Security(auth_scheme)):
    expected_api_key = os.getenv("FRONTEND_API_KEY")

    if not expected_api_key:
        raise HTTPException(
            status_code=500,
            detail="API key is not configured in the environment, for some reason",
        )

    if credentials.credentials != expected_api_key:
        raise HTTPException(status_code=403, detail="Invalid API key")

    return credentials.credentials
