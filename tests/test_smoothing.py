import numpy as np

from npcify.smoothing import OneEuroFilter

FPS = 30.0


def test_reduces_jitter_when_still() -> None:
    rng = np.random.default_rng(0)
    f = OneEuroFilter(min_cutoff=0.5, beta=0.02)
    raw = 10 + rng.normal(0, 2, 300)
    out = np.array([f([v], i / FPS)[0] for i, v in enumerate(raw)])
    assert out[30:].std() < 0.4 * raw[30:].std()


def test_follows_a_sudden_step_within_a_few_frames() -> None:
    f = OneEuroFilter(min_cutoff=0.5, beta=0.02)
    for i in range(60):
        f([0.0], i / FPS)
    out = [f([30.0], (60 + k) / FPS)[0] for k in range(10)]
    assert out[4] > 0.8 * 30  # 80% of the way within 5 frames
    assert out[-1] > 0.95 * 30


def test_vector_elements_are_filtered_independently() -> None:
    f = OneEuroFilter(min_cutoff=1.0, beta=8.0)
    for i in range(30):
        f([0.0, 0.0], i / FPS)
    out = f([1.0, 0.0], 30 / FPS)
    assert out[1] == 0.0 and out[0] > 0


def test_first_sample_passes_straight_through_and_reset_works() -> None:
    f = OneEuroFilter()
    assert f([5.0], 0.0)[0] == 5.0
    f.reset()
    assert f([-3.0], 1.0)[0] == -3.0
