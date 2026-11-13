"""Large-file scan of a training repo's build context, shared by validator registration and the trainer's image build.

The rules ban weights-derived payloads over about 1 MB (for example precomputed model weights or deltas). Whether a
large file is weights-derived needs a human, so this only lists candidates; it never rejects. Model weights fetched
at runtime are never in the build context, so they are never listed.
"""

import os
from pathlib import Path

from pydantic import BaseModel


GIT_DIR_NAME = ".git"


class LargeFile(BaseModel):
    path: str  # relative to the scanned root, POSIX separators
    size_bytes: int


def find_large_files(root: Path, threshold_bytes: int) -> list[LargeFile]:
    """Regular files under root larger than threshold_bytes, largest first. Skips .git and never follows symlinks."""
    found: list[LargeFile] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = [name for name in dirnames if name != GIT_DIR_NAME]
        for filename in filenames:
            path = Path(dirpath) / filename
            if path.is_symlink() or not path.is_file():
                continue
            size = path.stat().st_size
            if size > threshold_bytes:
                found.append(LargeFile(path=path.relative_to(root).as_posix(), size_bytes=size))
    return sorted(found, key=lambda large_file: (-large_file.size_bytes, large_file.path))


def describe_large_files(large_files: list[LargeFile], limit: int = 20) -> str:
    shown = ", ".join(f"{large_file.path} ({large_file.size_bytes} B)" for large_file in large_files[:limit])
    more = len(large_files) - limit
    return shown + (f", and {more} more" if more > 0 else "")
