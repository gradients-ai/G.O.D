import asyncio
import json
import os
import re
from dataclasses import dataclass

import httpx


API_URL = "https://api.depot.dev"


@dataclass
class DepotBuildResult:
    build_id: str
    returncode: int
    output: str
    timed_out: bool = False


class DepotClient:
    def __init__(self) -> None:
        self.token = os.getenv("DEPOT_TOKEN")
        self.organization_id = os.getenv("DEPOT_ORGANIZATION_ID")
        if not self.token or not self.organization_id:
            raise RuntimeError("DEPOT_TOKEN and DEPOT_ORGANIZATION_ID are required when image builds are enabled")
        self.region = os.getenv("DEPOT_REGION", "us-east-1")
        self.cli = os.getenv("DEPOT_CLI_PATH", "depot")

    async def _post(self, service: str, method: str, payload: dict) -> dict:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                f"{API_URL}/{service}/{method}",
                headers={
                    "Authorization": f"Bearer {self.token}",
                    "Content-Type": "application/json",
                    "Connect-Protocol-Version": "1",
                },
                json=payload,
            )
            response.raise_for_status()
            return response.json()

    async def create_project(self, name: str) -> str:
        projects = await self._post(
            "depot.core.v1.ProjectService",
            "ListProjects",
            {},
        )
        existing = next(
            (
                project
                for project in projects.get("projects", [])
                if project.get("name") == name or project.get("Name") == name
            ),
            None,
        )
        if existing:
            return existing.get("projectId") or existing["project_id"]
        # Prefer the CLI for project creation: it accepts org tokens and returns stable JSON.
        process = await asyncio.create_subprocess_exec(
            self.cli,
            "projects",
            "create",
            name,
            "--organization",
            self.organization_id,
            "--region",
            self.region,
            "--cache-storage-policy",
            "50",
            "--cache-policy-keep-days",
            os.getenv("DEPOT_CACHE_KEEP_DAYS", "14"),
            "--max-concurrent-builds",
            os.getenv("DEPOT_BUILDS_PER_INSTANCE", "1"),
            "--token",
            self.token,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        output, _ = await process.communicate()
        text = output.decode(errors="replace")
        if process.returncode:
            raise RuntimeError(f"depot projects create failed: {text[-1000:]}")
        payload = json.loads(text)
        return payload["project_id"]

    async def create_build(self, project_id: str) -> tuple[str, str]:
        result = await self._post(
            "depot.build.v1.BuildService",
            "CreateBuild",
            {"projectId": project_id},
        )
        build_id = result.get("buildId") or result.get("build_id")
        build_token = result.get("buildToken") or result.get("build_token")
        if not build_id or not build_token:
            raise RuntimeError(f"CreateBuild returned unexpected payload: {sorted(result)}")
        return build_id, build_token

    async def delete_project(self, project_id: str) -> None:
        try:
            await self._post(
                "depot.core.v1.ProjectService",
                "DeleteProject",
                {"projectId": project_id},
            )
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 404:
                raise

    async def run_build(
        self,
        *,
        project_id: str,
        build_id: str,
        build_token: str,
        context: str,
        dockerfile: str,
        platform: str,
        tag: str,
        github_token: str | None,
        timeout_seconds: int,
    ) -> DepotBuildResult:
        env = os.environ.copy()
        env.update(
            {
                "DEPOT_PROJECT_ID": project_id,
                "DEPOT_BUILD_ID": build_id,
                "DEPOT_TOKEN": build_token,
            }
        )
        if github_token:
            env["GITHUB_TOKEN"] = github_token
        args = [
            self.cli,
            "build",
            "--project",
            project_id,
            "--platform",
            platform,
            "--file",
            dockerfile,
            "--save",
            "--save-tag",
            tag,
            "--progress",
            "plain",
        ]
        if github_token:
            args.extend(["--secret", "id=GIT_AUTH_TOKEN,env=GITHUB_TOKEN"])
        args.append(context)
        process = await asyncio.create_subprocess_exec(
            *args,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            output, _ = await asyncio.wait_for(process.communicate(), timeout=timeout_seconds)
        except TimeoutError:
            process.terminate()
            await process.wait()
            await self.cancel_build(project_id, build_id)
            return DepotBuildResult(build_id, 124, "Build exceeded deadline", timed_out=True)
        return DepotBuildResult(build_id, process.returncode or 0, output.decode(errors="replace"))

    async def cancel_build(self, project_id: str, build_id: str) -> None:
        process = await asyncio.create_subprocess_exec(
            self.cli,
            "build",
            "cancel",
            build_id,
            "--project",
            project_id,
            "--token",
            self.token,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await process.wait()

    async def find_image(self, project_id: str, tag: str) -> tuple[str, str] | None:
        result = await self._post(
            "depot.build.v1.RegistryService",
            "ListImages",
            {"projectId": project_id, "pageSize": 100},
        )
        for image in result.get("images", []):
            if image["tag"].endswith(f":{tag}"):
                return image["tag"], image["digest"]
        return None

    async def delete_image(self, project_id: str, tag: str) -> None:
        await self._post(
            "depot.build.v1.RegistryService",
            "DeleteImage",
            {"projectId": project_id, "imageTags": [tag]},
        )


def classify_failure(output: str, timed_out: bool) -> str:
    if timed_out:
        return "timeout"
    lowered = output.lower()
    deterministic = (
        "failed to read dockerfile",
        "dockerfile parse error",
        "did not complete successfully: exit code",
        "unable to read tree",
        "no such file or directory",
    )
    if any(marker in lowered for marker in deterministic):
        return "dockerfile"
    if re.search(r"(timeout|connection reset|unavailable|provision|rate limit)", lowered):
        return "infrastructure"
    return "unknown"
