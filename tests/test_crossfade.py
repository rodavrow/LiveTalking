import unittest

import numpy as np

from utils.crossfade import CrossFade


class FakeClock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def _blend(src: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
    return src * (1 - alpha) + target * alpha


def _make(clock: FakeClock) -> CrossFade:
    return CrossFade(0.25, _blend, clock=clock)


class CrossFadeTest(unittest.TestCase):
    def test_no_start_returns_target(self):
        target = np.full((2, 2), 10.0)
        self.assertIs(_make(FakeClock()).apply(target), target)

    def test_start_with_none_returns_target(self):
        fade = _make(FakeClock())
        fade.start(None)
        target = np.full((2, 2), 10.0)
        self.assertIs(fade.apply(target), target)

    def test_midway_blends(self):
        clock = FakeClock()
        fade = _make(clock)
        fade.start(np.zeros((2, 2)))
        clock.now = 0.125
        np.testing.assert_allclose(fade.apply(np.full((2, 2), 10.0)), np.full((2, 2), 5.0))

    def test_after_duration_returns_target(self):
        clock = FakeClock()
        fade = _make(clock)
        fade.start(np.zeros((2, 2)))
        clock.now = 0.25
        target = np.full((2, 2), 10.0)
        self.assertIs(fade.apply(target), target)

    def test_restart_replaces_source(self):
        clock = FakeClock()
        fade = _make(clock)
        fade.start(np.zeros((2, 2)))
        clock.now = 1.0
        fade.start(np.full((2, 2), 4.0))
        clock.now = 1.125
        np.testing.assert_allclose(fade.apply(np.full((2, 2), 10.0)), np.full((2, 2), 7.0))


if __name__ == "__main__":
    unittest.main()
