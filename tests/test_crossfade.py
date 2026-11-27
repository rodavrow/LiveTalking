import unittest

import numpy as np

from utils.crossfade import CrossFade


def _blend(src: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
    return src * (1 - alpha) + target * alpha


def _make(frames: int = 4) -> CrossFade:
    return CrossFade(frames, _blend)


class CrossFadeTest(unittest.TestCase):
    def test_no_start_returns_target(self):
        target = np.full((2, 2), 10.0)
        self.assertIs(_make().apply(target), target)

    def test_start_with_none_returns_target(self):
        fade = _make()
        fade.start(None)
        target = np.full((2, 2), 10.0)
        self.assertIs(fade.apply(target), target)

    def test_alpha_steps_once_per_applied_frame(self):
        fade = _make(4)
        fade.start(np.zeros((2, 2)))
        target = np.full((2, 2), 8.0)
        values = [float(fade.apply(target)[0, 0]) for _ in range(3)]
        self.assertEqual(values, [2.0, 4.0, 6.0])

    def test_last_fade_frame_returns_target(self):
        fade = _make(4)
        fade.start(np.zeros((2, 2)))
        target = np.full((2, 2), 8.0)
        for _ in range(3):
            fade.apply(target)
        self.assertIs(fade.apply(target), target)

    def test_regression_burst_of_frames_still_fades_in(self):
        # Frames for a whole batch are produced within milliseconds, far ahead of playback.
        # A wall-clock fade left every one of them at alpha ~0, so the first talking
        # batch went out looking idle. Counting frames makes the fade independent of timing.
        fade = _make(4)
        fade.start(np.zeros((2, 2)))
        target = np.full((2, 2), 8.0)
        out = [fade.apply(target) for _ in range(16)]
        self.assertIs(out[3], target)
        self.assertTrue(all(o is target for o in out[3:]))

    def test_restart_replaces_source(self):
        fade = _make(4)
        fade.start(np.zeros((2, 2)))
        fade.apply(np.full((2, 2), 8.0))
        fade.start(np.full((2, 2), 4.0))
        np.testing.assert_allclose(fade.apply(np.full((2, 2), 8.0)), np.full((2, 2), 5.0))

    def test_rejects_non_positive_frames(self):
        with self.assertRaises(ValueError):
            CrossFade(0, _blend)


if __name__ == "__main__":
    unittest.main()
