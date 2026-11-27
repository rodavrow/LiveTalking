from typing import Callable

import numpy as np

BlendFn = Callable[[np.ndarray, np.ndarray, float], np.ndarray]


class CrossFade:
    """Fade from a captured source frame into the next `frames` output frames.

    The fade counts frames, not wall-clock time: the avatar renders a whole batch within
    milliseconds, well ahead of playback, so a timed fade would be over (or barely started)
    before any of those frames is seen. One apply() is one output video frame.
    The blend function is injected so this module stays free of cv2.
    """

    def __init__(self, frames: int, blend: BlendFn) -> None:
        if frames < 1:
            raise ValueError("frames must be >= 1")
        self._frames = frames
        self._blend = blend
        self._source: np.ndarray | None = None
        self._applied = 0

    def start(self, from_frame: np.ndarray | None) -> None:
        """Begin a fade from from_frame; None means there is nothing to fade from."""
        self._source = from_frame
        self._applied = 0

    def apply(self, target: np.ndarray) -> np.ndarray:
        """Return target blended with the source while the fade is running."""
        if self._source is None:
            return target
        self._applied += 1
        if self._applied >= self._frames:
            self._source = None
            return target
        return self._blend(self._source, target, self._applied / self._frames)
