"""Emotion labels, avatar config parsing and the pending-emotion slot."""
import json
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

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
