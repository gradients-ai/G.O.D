"""Localhost ``registry:3`` used as a staging area for prepared images.

Builds push here (fast local disk), then blobs are copied to R2. Pulls download blobs from R2 into the same
storage and ``docker pull`` from localhost. The storage is only a staging area: R2 is the source of truth and
the Docker image store keeps the pulled images, so the storage is wiped whenever no build or pull holds a lease.
"""

import os
import shutil
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import docker
import httpx

from core.image_registry import validate_digest
from core.logging import get_logger
from trainer import constants as cst


logger = get_logger(__name__)

_REGISTRY_CONFIG = """version: 0.1
log:
  level: warn
storage:
  filesystem:
    rootdirectory: /var/lib/registry
  delete:
    enabled: true
  maintenance:
    uploadpurging:
      enabled: false
http:
  addr: :5000
"""


class LocalRegistry:
    def __init__(
        self,
        storage_path: str = cst.LOCAL_REGISTRY_STORAGE_PATH,
        port: int = cst.LOCAL_REGISTRY_PORT,
        container_name: str = cst.LOCAL_REGISTRY_CONTAINER_NAME,
        image: str = cst.LOCAL_REGISTRY_IMAGE,
    ):
        self.base = Path(storage_path)
        self.data = self.base / "data"
        self.port = port
        self.container_name = container_name
        self.image = image
        self._leases = 0
        self._lock = threading.Lock()
        self._start_lock = threading.Lock()

    @property
    def host(self) -> str:
        return f"127.0.0.1:{self.port}"

    @property
    def root(self) -> Path:
        return self.data / "docker" / "registry" / "v2"

    def ref(self, repository: str, digest: str) -> str:
        return f"{self.host}/{repository}@{validate_digest(digest)}"

    def tag_ref(self, repository: str, tag: str) -> str:
        return f"{self.host}/{repository}:{tag}"

    def blob_path(self, digest: str) -> Path:
        hex_digest = validate_digest(digest).split(":", 1)[1]
        return self.root / "blobs" / "sha256" / hex_digest[:2] / hex_digest / "data"

    def _write_link(self, path: Path, digest: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(digest)

    def write_layer_link(self, repository: str, digest: str) -> None:
        hex_digest = validate_digest(digest).split(":", 1)[1]
        self._write_link(self.root / "repositories" / repository / "_layers" / "sha256" / hex_digest / "link", digest)

    def write_revision_link(self, repository: str, digest: str) -> None:
        hex_digest = validate_digest(digest).split(":", 1)[1]
        path = self.root / "repositories" / repository / "_manifests" / "revisions" / "sha256" / hex_digest / "link"
        self._write_link(path, digest)

    def _matches(self, container) -> bool:
        """True when an existing container serves this storage path on this port (settings may have changed)."""
        mounts = {m.get("Destination"): m.get("Source") for m in container.attrs.get("Mounts") or []}
        bindings = (container.attrs.get("HostConfig") or {}).get("PortBindings") or {}
        host_ports = {b.get("HostPort") for b in bindings.get("5000/tcp") or []}
        return mounts.get("/var/lib/registry") == str(self.data.resolve()) and str(self.port) in host_ports

    def ensure_running(self) -> None:
        """Start (or recreate) the localhost registry container and wait until it answers."""
        with self._start_lock:
            self.data.mkdir(parents=True, exist_ok=True)
            config_path = self.base / "config.yml"
            config_path.write_text(_REGISTRY_CONFIG)
            client = docker.from_env()
            try:
                container = client.containers.get(self.container_name)
                if not self._matches(container):
                    logger.warning("Local image registry container has stale storage/port settings; recreating it")
                    container.remove(force=True)
                    raise docker.errors.NotFound("recreate")
                if container.status != "running":
                    container.start()
                elif self._healthy():
                    return
            except docker.errors.NotFound:
                try:
                    client.images.get(self.image)
                except docker.errors.ImageNotFound:
                    client.images.pull(self.image)
                client.containers.run(
                    self.image,
                    name=self.container_name,
                    detach=True,
                    restart_policy={"Name": "unless-stopped"},
                    ports={"5000/tcp": ("127.0.0.1", self.port)},
                    # Run as the trainer's user so the trainer can wipe the storage it shares with the registry.
                    user=f"{os.getuid()}:{os.getgid()}",
                    environment={"OTEL_TRACES_EXPORTER": "none"},
                    volumes={
                        str(self.data.resolve()): {"bind": "/var/lib/registry", "mode": "rw"},
                        str(config_path.resolve()): {"bind": "/etc/distribution/config.yml", "mode": "ro"},
                    },
                )
            for _ in range(30):
                if self._healthy():
                    return
                time.sleep(1)
            raise RuntimeError(f"Local image registry did not become healthy on {self.host}")

    def _healthy(self) -> bool:
        try:
            return httpx.get(f"http://{self.host}/v2/", timeout=3).status_code == 200
        except httpx.HTTPError:
            return False

    @contextmanager
    def lease(self):
        """Hold the staging storage; it is wiped when the last holder releases it."""
        with self._lock:
            self._leases += 1
        try:
            yield self
        finally:
            with self._lock:
                self._leases -= 1
                if self._leases == 0:
                    self._purge_unlocked()

    def _purge_unlocked(self) -> None:
        # Rename (instant) and delete in the background: releases happen on the trainer's event loop, and an
        # rmtree of tens of GB there would stall every request.
        target = self.data / "docker"
        if not target.exists():
            return
        trash = self.base / f"trash-{uuid.uuid4().hex}"
        target.rename(trash)
        threading.Thread(target=self._delete_trash, daemon=True).start()
        logger.info("Purged local image registry staging storage")

    def _delete_trash(self) -> None:
        for trash in self.base.glob("trash-*"):
            shutil.rmtree(trash, ignore_errors=True)


local_registry = LocalRegistry()
