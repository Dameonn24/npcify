from pathlib import Path

import cv2
import numpy as np
import pytest

from npcify.detect import YuNetDetector, detection_size, to_detections
from npcify.types import BoundingBox

LOCAL_ASSETS = Path(__file__).parent / "assets" / "local"


def raw_row(x, y, w, h, landmarks=None, score=0.9) -> list[float]:
    lm = landmarks or [(0.0, 0.0)] * 5
    return [x, y, w, h, *[v for p in lm for v in p], score]


def test_detection_size_downscales_keeping_aspect() -> None:
    assert detection_size(1280, 720, 640) == (640, 360)
    assert detection_size(1920, 1080, 640) == (640, 360)


def test_detection_size_never_upscales() -> None:
    assert detection_size(320, 240, 640) == (320, 240)


def test_scaling_box_and_landmarks_to_full_frame() -> None:
    # Detected at 640x360 on a 1280x720 frame: everything doubles.
    lm = [(110.0, 70.0), (130.0, 70.0), (120.0, 85.0), (112.0, 100.0), (128.0, 100.0)]
    raw = np.array([raw_row(100, 50, 40, 60, lm, 0.87)], dtype=np.float32)
    [det] = to_detections(raw, 2.0, 2.0, 1280, 720)
    assert det.box == BoundingBox(200, 100, 80, 120)
    assert det.confidence == pytest.approx(0.87)
    assert det.landmarks == ((220, 140), (260, 140), (240, 170), (224, 200), (256, 200))


def test_scaling_with_unequal_axes() -> None:
    raw = np.array([raw_row(10, 10, 20, 20)], dtype=np.float32)
    [det] = to_detections(raw, 3.0, 1.5, 1000, 1000)
    assert det.box == BoundingBox(30, 15, 60, 30)


def test_fractional_boxes_round_outwards() -> None:
    # 10.5..30.5 must become 10..31 so the box never shrinks.
    raw = np.array([raw_row(10.5, 10.5, 20.0, 20.0)], dtype=np.float32)
    [det] = to_detections(raw, 1.0, 1.0, 100, 100)
    assert det.box == BoundingBox(10, 10, 21, 21)


def test_boxes_clipped_to_frame() -> None:
    raw = np.array(
        [raw_row(-20, -10, 60, 50), raw_row(90, 80, 40, 40)],
        dtype=np.float32,
    )
    a, b = to_detections(raw, 1.0, 1.0, 100, 100)
    assert a.box == BoundingBox(0, 0, 40, 40)
    assert b.box == BoundingBox(90, 80, 10, 20)


def test_invalid_boxes_dropped() -> None:
    raw = np.array(
        [
            raw_row(150, 10, 20, 20),  # entirely off the right edge
            raw_row(10, -50, 20, 20),  # entirely above the frame
            raw_row(10, 10, 0, 20),  # zero width
            raw_row(10, 10, 20, -5),  # negative height
            raw_row(10, 10, 20, 20),  # the only valid one
        ],
        dtype=np.float32,
    )
    dets = to_detections(raw, 1.0, 1.0, 100, 100)
    assert [d.box for d in dets] == [BoundingBox(10, 10, 20, 20)]


def test_no_raw_output_means_no_detections() -> None:
    assert to_detections(None, 1.0, 1.0, 100, 100) == []


def test_detector_returns_nothing_on_blank_image() -> None:
    detector = YuNetDetector()
    assert detector.detect(np.zeros((720, 1280, 3), dtype=np.uint8)) == []


def _local_face_images() -> list[Path]:
    if not LOCAL_ASSETS.is_dir():
        return []
    return sorted(p for p in LOCAL_ASSETS.iterdir() if p.suffix.lower() in {".jpg", ".png"})


@pytest.mark.skipif(not _local_face_images(), reason="no local face images in tests/assets/local/")
@pytest.mark.parametrize("path", _local_face_images(), ids=lambda p: p.name)
def test_detector_finds_face_in_local_image(path: Path) -> None:
    image = cv2.imread(str(path))
    dets = YuNetDetector().detect(image)
    assert dets, f"no face found in {path.name}"
    h, w = image.shape[:2]
    for d in dets:
        assert 0 <= d.box.x and d.box.x + d.box.width <= w
        assert 0 <= d.box.y and d.box.y + d.box.height <= h
