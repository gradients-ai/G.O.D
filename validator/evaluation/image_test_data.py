"""Deterministic image preparation and training/test duplicate screening."""

import hashlib
import io
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image


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


def difference_hash(image):
    gray = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS))
    bits = (gray[:, 1:] > gray[:, :-1]).flatten()
    return sum(int(bit) << i for i, bit in enumerate(bits))


def read_dataset(source, download_path, require_captions=True):
    """Read a nested local directory/ZIP or HTTP ZIP without extracting archive paths."""
    if str(source).startswith(("https://", "http://")):
        with urllib.request.urlopen(source, timeout=120) as response, download_path.open("wb") as out:
            shutil.copyfileobj(response, out)
        source = download_path
    source = Path(source)
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
    if not require_captions:
        rows = [
            (name, raw, caption if caption and caption.strip() else "training duplicate screen") for name, raw, caption in rows
        ]
    return rows


def decoded_images(rows):
    result = []
    for name, raw, caption in rows:
        if not caption or not caption.strip():
            raise ValueError("Image evaluation requires a caption for every image")
        image = adjust_image(Image.open(io.BytesIO(raw)))
        digest = hashlib.sha256(image.tobytes() + str(image.size).encode()).hexdigest()
        result.append({"name": name, "image": image, "caption": caption, "sha256": digest, "dhash": difference_hash(image)})
    return result


def held_out_images(training, validation):
    """Conservatively screen held-out images against ALL actual training images.

    A 64-bit difference-hash distance <= 4 is a heuristic near-duplicate flag,
    not a guarantee of semantic deduplication. Record every exclusion for review.
    """
    kept, excluded, seen = [], [], set()
    for item in validation:
        matches = [t["name"] for t in training if (item["dhash"] ^ t["dhash"]).bit_count() <= 4]
        if matches or item["sha256"] in seen:
            excluded.append({"image": item["name"], "training_matches": matches, "duplicate_validation": not matches})
        else:
            kept.append(item)
            seen.add(item["sha256"])
    if not kept:
        raise ValueError("No held-out images remain after duplicate screening")
    return kept, excluded
