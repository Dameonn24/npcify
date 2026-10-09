from pathlib import Path

import cv2
import numpy as np
import pytest

from npcify.capture import FrameSource

WIDTH, HEIGHT, N_FRAMES = 160, 120, 30


@pytest.fixture
def test_video(tmp_path: Path) -> Path:
    """A short video: left half red, right half blue."""
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (WIDTH, HEIGHT))
    assert writer.isOpened()
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    frame[:, : WIDTH // 2] = (0, 0, 255)  # BGR red
    frame[:, WIDTH // 2 :] = (255, 0, 0)  # BGR blue
    for _ in range(N_FRAMES):
        writer.write(frame)
    writer.release()
    return path


def read_all(source: FrameSource) -> list[np.ndarray]:
    frames = []
    while (frame := source.read()) is not None:
        frames.append(frame)
    return frames


def test_reads_all_frames_at_right_size(test_video: Path) -> None:
    with FrameSource(test_video) as source:
        assert source.resolution == (WIDTH, HEIGHT)
        assert not source.mirror  # off by default for files
        frames = read_all(source)
    assert len(frames) == N_FRAMES
    assert all(f.shape == (HEIGHT, WIDTH, 3) for f in frames)


def test_mirror_flips_horizontally(test_video: Path) -> None:
    with FrameSource(test_video) as source:
        normal = source.read()
    with FrameSource(test_video, mirror=True) as source:
        mirrored = source.read()
    assert normal is not None and mirrored is not None
    # Left side starts red; after mirroring the left side is blue.
    assert normal[HEIGHT // 2, 10, 2] > 200 and normal[HEIGHT // 2, 10, 0] < 50
    assert mirrored[HEIGHT // 2, 10, 0] > 200 and mirrored[HEIGHT // 2, 10, 2] < 50


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        FrameSource(tmp_path / "nope.mp4")
