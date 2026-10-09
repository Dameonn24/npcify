import pytest

from npcify.timing import FPSCounter, StageTimer


def test_fps_zero_before_two_frames() -> None:
    counter = FPSCounter()
    assert counter.fps == 0.0
    counter.tick(now=0.0)
    assert counter.fps == 0.0
    assert counter.frame_ms == 0.0


def test_fps_steady_rate() -> None:
    counter = FPSCounter(window=30)
    for i in range(100):
        counter.tick(now=i / 30)
    assert counter.fps == pytest.approx(30.0)
    assert counter.frame_ms == pytest.approx(1000 / 30)


def test_fps_only_uses_recent_window() -> None:
    counter = FPSCounter(window=10)
    t = 0.0
    for _ in range(20):  # slow start: 10 FPS
        t += 0.1
        counter.tick(now=t)
    for _ in range(10):  # recent: 50 FPS
        t += 0.02
        counter.tick(now=t)
    assert counter.fps == pytest.approx(50.0)


def test_stage_timer_averages_with_fake_clock() -> None:
    now = [0.0]
    timer = StageTimer(clock=lambda: now[0])
    for duration in (0.010, 0.020):
        with timer.stage("process"):
            now[0] += duration
    with timer.stage("capture"):
        now[0] += 0.005
    avgs = timer.averages()
    assert list(avgs) == ["process", "capture"]
    assert avgs["process"] == pytest.approx(15.0)
    assert avgs["capture"] == pytest.approx(5.0)
