import time
from typing import Callable

import numpy as np

BlendFn = Callable[[np.ndarray, np.ndarray, float], np.ndarray]


class CrossFade:
    """Time-based fade from a captured source frame into the frames that follow.

    The blend function is injected so this module stays free of cv2.
    """

    def __init__(
        self,
        duration_s: float,
        blend: BlendFn,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._duration_s = duration_s
        self._blend = blend
        self._clock = clock
        self._source: np.ndarray | None = None
        self._started_at = 0.0

    def start(self, from_frame: np.ndarray | None) -> None:
        """Begin a fade from from_frame; None means there is nothing to fade from."""
        self._source = from_frame
        self._started_at = self._clock()

    def apply(self, target: np.ndarray) -> np.ndarray:
        """Return target blended with the source while the fade is running."""
        if self._source is None:
            return target
        elapsed = self._clock() - self._started_at
        if elapsed >= self._duration_s:
            return target
        alpha = min(1.0, elapsed / self._duration_s)
        return self._blend(self._source, target, alpha)
