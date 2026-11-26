"""Emotion labels, avatar config parsing and the pending-emotion slot."""
import json
import os
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from utils.logger import logger

NEUTRAL = "neutral"
CONFIG_FILENAME = "custom_config.json"


def parse_labels(raw: str) -> tuple[str, ...]:
    """Split a comma-separated label list; lower-cased, de-duplicated, neutral first if absent."""
    labels: list[str] = []
    for part in raw.split(","):
        name = part.strip().lower()
        if name and name not in labels:
            labels.append(name)
    if not labels:
        raise ValueError(f"no emotion labels in {raw!r}")
    if NEUTRAL not in labels:
        labels.insert(0, NEUTRAL)
    return tuple(labels)


def resolve_label(name: str | None, labels: tuple[str, ...]) -> str:
    """Map a caller-supplied name to a known label, falling back to neutral."""
    if name is None or not name.strip():
        return NEUTRAL
    label = name.strip().lower()
    if label not in labels:
        logger.warning("unknown emotion %r, using %s", name, NEUTRAL)
        return NEUTRAL
    return label


@dataclass(frozen=True)
class EmotionConfig:
    emotions: dict[str, str]
    preload: bool

    @classmethod
    def empty(cls) -> "EmotionConfig":
        return cls({}, True)


def _parse_emotions(raw: dict, labels: tuple[str, ...]) -> dict[str, str]:
    emotions: dict[str, str] = {}
    for key, avatar_id in raw.items():
        label = str(key).strip().lower()
        if label == NEUTRAL:
            # neutral is always the avatar's own frames, so a mapping would be dead config
            logger.warning("emotions: ignoring %r, neutral always uses the avatar itself", key)
        elif label not in labels:
            logger.warning("emotions: ignoring %r, not in the configured labels", key)
        elif not isinstance(avatar_id, str) or not avatar_id:
            raise ValueError(f"emotions: avatar id for {key!r} must be a non-empty string")
        else:
            emotions[label] = avatar_id
    return emotions


def parse_custom_config(
    raw: object, labels: tuple[str, ...]
) -> tuple[list[dict], EmotionConfig]:
    """Parse custom_config.json: the legacy list form or {custom, emotions, emotion_preload}."""
    if isinstance(raw, list):
        return raw, EmotionConfig.empty()
    if not isinstance(raw, dict):
        raise ValueError(f"custom config must be a list or an object, got {type(raw).__name__}")
    custom = raw.get("custom", [])
    if not isinstance(custom, list):
        raise ValueError("custom config: 'custom' must be a list")
    emotions = raw.get("emotions", {})
    if not isinstance(emotions, dict):
        raise ValueError("custom config: 'emotions' must be an object")
    preload = bool(raw.get("emotion_preload", True))
    return custom, EmotionConfig(_parse_emotions(emotions, labels), preload)


def read_avatar_config(
    avatar_dir: str, labels: tuple[str, ...]
) -> tuple[list[dict] | None, EmotionConfig]:
    """Read <avatar_dir>/custom_config.json; a missing file yields (None, empty config)."""
    path = os.path.join(avatar_dir, CONFIG_FILENAME)
    if not os.path.exists(path):
        return None, EmotionConfig.empty()
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        return parse_custom_config(raw, labels)
    except ValueError as exc:  # JSONDecodeError is a ValueError
        raise ValueError(f"invalid {path}: {exc}") from exc


class PendingEmotion:
    """One emotion waiting for the next utterance; stale after ttl_s seconds."""

    def __init__(self, ttl_s: float, clock: Callable[[], float] = time.monotonic) -> None:
        self._ttl_s = ttl_s
        self._clock = clock
        self._lock = threading.Lock()
        self._name: str | None = None
        self._set_at = 0.0

    def set(self, name: str) -> None:
        with self._lock:
            self._name = name
            self._set_at = self._clock()

    def take(self) -> str | None:
        with self._lock:
            name = self._fresh()
            self._name = None
            return name

    def peek(self) -> str | None:
        with self._lock:
            return self._fresh()

    def _fresh(self) -> str | None:
        if self._name is None or self._clock() - self._set_at >= self._ttl_s:
            return None
        return self._name


@dataclass(frozen=True, eq=False)
class FrameSet:
    """Full frames, face crops and crop coordinates of one talking loop."""

    name: str
    frames: Sequence[np.ndarray]
    faces: Sequence[np.ndarray] | None
    coords: Sequence

    def __len__(self) -> int:
        return len(self.frames)

    @property
    def nbytes(self) -> int:
        total = sum(f.nbytes for f in self.frames)
        if self.faces is not None:
            total += sum(f.nbytes for f in self.faces)
        return total


SetStatus = Literal["absent", "loading", "ready", "failed"]


def _start_daemon_thread(fn: Callable[[], None]) -> None:
    threading.Thread(target=fn, daemon=True, name="emotion-load").start()


def _validate(candidate: FrameSet, reference: FrameSet) -> str | None:
    """Return why candidate cannot replace reference, or None when compatible."""
    if len(candidate.frames) == 0:
        return "no frames"
    if len(candidate.coords) != len(candidate.frames):
        return f"{len(candidate.frames)} frames but {len(candidate.coords)} coords"
    if (candidate.faces is None) != (reference.faces is None):
        return "face crops present in only one of the sets"
    if candidate.faces is not None and len(candidate.faces) != len(candidate.frames):
        return f"{len(candidate.frames)} frames but {len(candidate.faces)} face crops"
    if candidate.frames[0].shape != reference.frames[0].shape:
        return f"frame shape {candidate.frames[0].shape} != {reference.frames[0].shape}"
    if candidate.faces is not None and reference.faces is not None:
        if candidate.faces[0].shape != reference.faces[0].shape:
            return f"face shape {candidate.faces[0].shape} != {reference.faces[0].shape}"
    return None


class EmotionSetStore:
    """Process-wide cache of emotion frame sets, shared by all sessions.

    Loading happens off the render path; a failed set is never retried.
    """

    def __init__(
        self,
        load_fn: Callable[[str, int | None], tuple],
        loader_workers: int = 1,
        start_thread: Callable[[Callable[[], None]], None] = _start_daemon_thread,
    ) -> None:
        self._load_fn = load_fn
        self._loader_workers = loader_workers
        self._start_thread = start_thread
        self._lock = threading.Lock()
        self._ready: dict[str, FrameSet] = {}
        self._loading: set[str] = set()
        self._failed: set[str] = set()

    def status(self, set_id: str) -> SetStatus:
        with self._lock:
            return self._status_locked(set_id)

    def get(self, set_id: str, reference: FrameSet, lazy: bool) -> FrameSet | None:
        """Return the ready set; with lazy, start one background load when absent."""
        with self._lock:
            ready = self._ready.get(set_id)
            if ready is not None or not lazy or self._status_locked(set_id) != "absent":
                return ready
            self._loading.add(set_id)
        self._start_thread(lambda: self._load(set_id, reference, self._loader_workers))
        return None

    def preload(self, set_ids: Iterable[str], reference: FrameSet) -> None:
        """Load synchronously with all cores, skipping ids already known."""
        for set_id in set_ids:
            with self._lock:
                if self._status_locked(set_id) != "absent":
                    continue
                self._loading.add(set_id)
            self._load(set_id, reference, None)

    def _status_locked(self, set_id: str) -> SetStatus:
        if set_id in self._ready:
            return "ready"
        if set_id in self._failed:
            return "failed"
        if set_id in self._loading:
            return "loading"
        return "absent"

    def _load(self, set_id: str, reference: FrameSet, max_workers: int | None) -> None:
        loaded: FrameSet | None = None
        try:
            frames, faces, coords = self._load_fn(set_id, max_workers)
            candidate = FrameSet(set_id, frames, faces, coords)
            reason = _validate(candidate, reference)
            if reason is None:
                loaded = candidate
        except Exception as exc:  # any loader failure must mark the set failed, not kill the thread
            reason = f"load failed: {exc!r}"
        with self._lock:
            self._loading.discard(set_id)
            if loaded is None:
                self._failed.add(set_id)
            else:
                self._ready[set_id] = loaded
        if loaded is None:
            logger.error("emotion set %s unusable: %s", set_id, reason)
            return
        height, width = loaded.frames[0].shape[:2]
        logger.info(
            "emotion set %s: %d frames %dx%d, %.1f MB",
            set_id, len(loaded), width, height, loaded.nbytes / 1e6,
        )
