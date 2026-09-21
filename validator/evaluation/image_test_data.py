"""Deterministic test-image loading and preparation."""

import hashlib
import io
import logging
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image

logger = logging.getLogger(__name__)


def _fmt_bytes(size: int) -> str:
    if size >= 1024**3:
        return f"{size / 1024**3:.2f} GiB"
    if size >= 1024**2:
        return f"{size / 1024**2:.1f} MiB"
    if size >= 1024:
        return f"{size / 1024:.1f} KiB"
    return f"{size} B"


def _download_dataset(source: str, download_path: Path) -> Path:
    host = urlparse(source).netloc or "remote"
    logger.info("downloading test dataset host=%s dest=%s", host, download_path.name)
    last_pct = -10
    downloaded = 0
    with urllib.request.urlopen(source, timeout=120) as response, download_path.open("wb") as out:
        total_size = int(response.headers.get("Content-Length") or 0)
        while True:
            chunk = response.read(8 * 1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                pct = min(100, int(downloaded * 100 / total_size))
                if pct >= last_pct + 10 or pct == 100:
                    last_pct = pct
                    logger.info(
                        "test dataset download %s%% (%s / %s)",
                        pct,
                        _fmt_bytes(min(downloaded, total_size)),
                        _fmt_bytes(total_size),
                    )
            elif downloaded - max(last_pct, 0) >= 32 * 1024 * 1024:
                last_pct = downloaded
                logger.info("test dataset download %s so far", _fmt_bytes(downloaded))
    logger.info("test dataset download complete size=%s", _fmt_bytes(download_path.stat().st_size))
    return download_path


EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def adjust_image(image):
    w, h = image.size
    if min(w, h) <= 0:
        raise ValueError("Invalid image dimensions")
    size = (1024, int(h / w * 1024)) if w > h else (int(w / h * 1024), 1024)
    image = image.resize(size, Image.Resampling.LANCZOS)
    cw, ch = (v // 16 * 16 for v in size)
    left, top = (size[0] - cw) // 2, (size[1] - ch) // 2
    return image.crop((left, top, left + cw, top + ch)).convert("RGB")


def read_dataset(source, download_path):
    """Read a nested local directory/ZIP or HTTP ZIP without extracting archive paths."""
    if str(source).startswith(("https://", "http://")):
        source = _download_dataset(str(source), Path(download_path))
    source = Path(source)
    logger.info("reading test dataset path=%s", source.name if source.is_file() else source)
    rows = []
    if source.is_dir():
        for path in sorted(source.rglob("*")):
            if path.is_file() and path.suffix.lower() in EXTENSIONS:
                if not path.resolve().is_relative_to(source.resolve()):
                    raise ValueError("Dataset image escapes its directory")
                caption = path.with_suffix(".txt")
                if caption.exists() and not caption.resolve().is_relative_to(source.resolve()):
                    raise ValueError("Dataset caption escapes its directory")
                rows.append((str(path.relative_to(source)), path.read_bytes(), caption.read_text() if caption.exists() else None))
    else:
        with zipfile.ZipFile(source) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise ValueError("Duplicate archive entries")
            for name in sorted(names):
                if name.lower().endswith(EXTENSIONS):
                    caption = str(Path(name).with_suffix(".txt"))
                    rows.append((name, archive.read(name), archive.read(caption).decode("utf-8") if caption in names else None))
    if not rows:
        raise ValueError("Empty image dataset")
    logger.info("loaded %s test images", len(rows))
    return rows


def decoded_images(rows):
    logger.info("decoding %s test images", len(rows))
    result = []
    for index, (name, raw, caption) in enumerate(rows, start=1):
        if not caption or not caption.strip():
            raise ValueError("Image evaluation requires a caption for every image")
        image = adjust_image(Image.open(io.BytesIO(raw)))
        digest = hashlib.sha256(image.tobytes() + str(image.size).encode()).hexdigest()
        result.append({"name": name, "image": image, "caption": caption, "sha256": digest})
        if index == 1 or index == len(rows) or index % 8 == 0:
            logger.info("decoded test image %s/%s name=%s", index, len(rows), name)
    return result
