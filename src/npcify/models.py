"""Download and cache model files.

Model weights are not stored in git. On first use they are downloaded into
~/.cache/npcify/models/ (override with the NPCIFY_CACHE_DIR environment
variable) and checked against a known SHA-256 checksum. Only model files are
ever downloaded; frames never leave the device.

Download everything in advance with:
    uv run npcify-download-models
"""

from __future__ import annotations

import hashlib
import os
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelFile:
    filename: str
    url: str
    sha256: str
    description: str


_ZOO = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models"

MODELS: dict[str, ModelFile] = {
    "yunet": ModelFile(
        filename="face_detection_yunet_2026may.onnx",
        url=f"{_ZOO}/face_detection_yunet/face_detection_yunet_2026may.onnx",
        sha256="ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0",
        description="YuNet face detector (OpenCV Zoo, MIT licence)",
    ),
    "sface": ModelFile(
        filename="face_recognition_sface_2021dec.onnx",
        url=f"{_ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        sha256="0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
        description="SFace face recogniser (OpenCV Zoo, Apache 2.0 licence)",
    ),
    "face_landmarker": ModelFile(
        filename="face_landmarker_v1.task",
        url="https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        sha256="64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
        description="MediaPipe Face Landmarker (Apache 2.0 licence)",
    ),
}


class ModelDownloadError(RuntimeError):
    """A model could not be downloaded or failed its checksum."""


def cache_dir() -> Path:
    override = os.environ.get("NPCIFY_CACHE_DIR")
    base = Path(override) if override else Path.home() / ".cache" / "npcify"
    return base / "models"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_model(name: str) -> Path:
    """Return the local path to a model, downloading and verifying it if needed."""
    model = MODELS[name]
    path = cache_dir() / model.filename
    if path.is_file() and sha256_of(path) == model.sha256:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    try:
        urllib.request.urlretrieve(model.url, tmp)
    except OSError as e:
        tmp.unlink(missing_ok=True)
        raise ModelDownloadError(_manual_help(model, path, f"Download failed: {e}")) from e

    actual = sha256_of(tmp)
    if actual != model.sha256:
        tmp.unlink(missing_ok=True)
        raise ModelDownloadError(
            _manual_help(model, path, f"Checksum mismatch (expected {model.sha256}, got {actual}).")
        )
    tmp.replace(path)
    return path


def _manual_help(model: ModelFile, path: Path, reason: str) -> str:
    return (
        f"{reason}\n"
        f"Could not get the {model.description}. To install it by hand:\n"
        f"  1. Download {model.url}\n"
        f"  2. Save it as {path}\n"
        f"  3. Check: shasum -a 256 '{path}' should print {model.sha256}"
    )


def main() -> int:
    """Download every known model into the cache."""
    for name in MODELS:
        try:
            path = ensure_model(name)
        except ModelDownloadError as e:
            print(f"[{name}] {e}", file=sys.stderr)
            return 1
        print(f"[{name}] OK  {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
