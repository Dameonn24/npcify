import hashlib
from pathlib import Path

import pytest

from npcify import models
from npcify.models import ModelDownloadError, ModelFile, ensure_model


@pytest.fixture
def fake_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, bytes]:
    """A 'remote' model served from a local file:// URL, with a temp cache dir."""
    content = b"not really a model"
    remote = tmp_path / "remote.onnx"
    remote.write_bytes(content)
    monkeypatch.setenv("NPCIFY_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setitem(
        models.MODELS,
        "fake",
        ModelFile("fake.onnx", remote.as_uri(), hashlib.sha256(content).hexdigest(), "fake"),
    )
    return remote, content


def test_downloads_into_cache_and_verifies(fake_model: tuple[Path, bytes], tmp_path: Path) -> None:
    _, content = fake_model
    path = ensure_model("fake")
    assert path == tmp_path / "cache" / "models" / "fake.onnx"
    assert path.read_bytes() == content


def test_uses_cached_copy_without_downloading(fake_model: tuple[Path, bytes]) -> None:
    remote, _ = fake_model
    path = ensure_model("fake")
    remote.unlink()  # a second download would now fail
    assert ensure_model("fake") == path


def test_checksum_mismatch_raises_and_leaves_nothing(
    fake_model: tuple[Path, bytes], monkeypatch: pytest.MonkeyPatch
) -> None:
    good = models.MODELS["fake"]
    monkeypatch.setitem(models.MODELS, "fake", ModelFile(good.filename, good.url, "0" * 64, "x"))
    with pytest.raises(ModelDownloadError, match="Checksum mismatch"):
        ensure_model("fake")
    assert not list(models.cache_dir().iterdir())


def test_failed_download_explains_manual_steps(
    fake_model: tuple[Path, bytes], monkeypatch: pytest.MonkeyPatch
) -> None:
    good = models.MODELS["fake"]
    missing = ModelFile(good.filename, good.url + ".missing", good.sha256, "x")
    monkeypatch.setitem(models.MODELS, "fake", missing)
    with pytest.raises(ModelDownloadError, match="Save it as"):
        ensure_model("fake")
