###############################################################################
#  Streamed replies: PCM reframing, pre-buffer and per-session reply state
#  for the /stream_audio WebSocket (server/routes.py).
###############################################################################

import re
import threading
import time
from collections import deque
from typing import Callable, Literal, Optional

import numpy as np
from numpy.typing import NDArray

from utils.logger import logger

SAMPLE_RATE = 16000
FRAME_SAMPLES = 320                 # 20 ms at 16 kHz, what put_audio_frame expects
FRAME_BYTES = FRAME_SAMPLES * 2     # s16le
PREBUFFER_FRAMES = 15               # 300 ms held before the first frame is fed
STALL_TIMEOUT_S = 5.0               # no message for this long ends the reply
CANCELLED_KEEP = 32                 # cancelled reply ids remembered per session
PLAY_GRACE_S = 10.0                 # a fed reply counts as playing this long past its length
FRAME_S = FRAME_SAMPLES / SAMPLE_RATE

_REPLY_ID_RE = re.compile(r"[A-Za-z0-9_.:-]{1,64}")

Frame = NDArray[np.float32]
Sink = Callable[[Frame, dict], None]
OpenResult = Literal["ok", "busy", "cancelled"]


def valid_reply_id(reply_id: Optional[str]) -> bool:
    return isinstance(reply_id, str) and _REPLY_ID_RE.fullmatch(reply_id) is not None


class PcmFramer:
    """Cuts a 16 kHz mono s16le byte stream into 20 ms float32 frames.

    Messages may split a sample or a frame anywhere; the remainder is kept for the next one.
    """

    def __init__(self) -> None:
        self._rest = bytearray()

    def feed(self, data: bytes) -> list[Frame]:
        self._rest += data
        whole = len(self._rest) // FRAME_BYTES * FRAME_BYTES
        if whole == 0:
            return []
        samples = np.frombuffer(bytes(self._rest[:whole]), dtype="<i2")
        del self._rest[:whole]
        floats = samples.astype(np.float32) / 32768.0
        return list(floats.reshape(-1, FRAME_SAMPLES))

    def flush(self) -> Optional[Frame]:
        """Last partial frame padded with silence; a lone trailing byte is dropped."""
        usable = len(self._rest) // 2 * 2
        rest = bytes(self._rest[:usable])
        self._rest.clear()
        if not rest:
            return None
        frame = np.zeros(FRAME_SAMPLES, dtype=np.float32)
        samples = np.frombuffer(rest, dtype="<i2")
        frame[:len(samples)] = samples.astype(np.float32) / 32768.0
        return frame


class ReplyFeeder:
    """Feeds one reply's frames to the avatar, after a pre-buffer.

    The newest frame is always held back so the last one fed can carry the 'end' event,
    the same events put_audio_file emits for an uploaded clip.
    """

    def __init__(self, sink: Sink, datainfo: dict,
                 prebuffer_frames: int = PREBUFFER_FRAMES,
                 allowed: Callable[[], bool] = lambda: True,
                 clock: Callable[[], float] = time.monotonic) -> None:
        if prebuffer_frames < 0:
            raise ValueError("prebuffer_frames must be >= 0")
        self._sink = sink
        self._datainfo = dict(datainfo)
        self._prebuffer_frames = prebuffer_frames
        self._allowed = allowed
        self._framer = PcmFramer()
        self._pending: list[Frame] = []
        self._released = False
        self._fed_frames = 0
        self._ended = False
        self._clock = clock
        self._opened_at = clock()
        self.received_bytes = 0
        self.messages = 0
        self.first_chunk_s: Optional[float] = None   # seconds after open
        self.released_s: Optional[float] = None      # pre-buffer released, seconds after open
        self.wav_header = False

    @property
    def fed_frames(self) -> int:
        return self._fed_frames

    def feed(self, data: bytes) -> None:
        if self._ended:
            return
        if self.messages == 0:
            self.first_chunk_s = self._clock() - self._opened_at
            # raw PCM is expected; a WAV header here would be played as a click
            self.wav_header = data[:4] == b"RIFF"
        self.messages += 1
        self.received_bytes += len(data)
        self._pending.extend(self._framer.feed(data))
        if not self._released and len(self._pending) <= self._prebuffer_frames:
            return
        if not self._released:
            self.released_s = self._clock() - self._opened_at
        self._released = True
        self._send(self._pending[:-1], last_is_end=False)
        self._pending = self._pending[-1:]

    def end(self) -> None:
        """Feed everything left; the last frame carries 'end'. Safe to call twice."""
        if self._ended:
            return
        self._ended = True
        tail = self._framer.flush()
        if tail is not None:
            self._pending.append(tail)
        if not self._released and self._pending:
            self.released_s = self._clock() - self._opened_at
        self._send(self._pending, last_is_end=True)
        self._pending = []

    def _send(self, frames: list[Frame], last_is_end: bool) -> None:
        for i, frame in enumerate(frames):
            if not self._allowed():
                return
            eventpoint: dict = {}
            if self._fed_frames == 0:
                eventpoint = {"status": "start"}
            if last_is_end and i == len(frames) - 1:
                eventpoint = {"status": "end"}
            eventpoint.update(self._datainfo)
            self._sink(frame, eventpoint)
            self._fed_frames += 1


class ReplyState:
    """Which streamed reply a session has open, and which were cancelled.

    A reply is current from open until its 'end' event has played out (on_event), or until
    it is cancelled. If 'end' never plays (no viewer pulling frames, say), the reply expires
    PLAY_GRACE_S after its audio would have finished, so it cannot block the session.
    notify() can run on an output thread, hence the lock.
    """

    def __init__(self, cancelled_keep: int = CANCELLED_KEEP,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._lock = threading.Lock()
        self._clock = clock
        self._open_id: Optional[str] = None     # socket open, frames still coming
        self._playing_id: Optional[str] = None  # all fed, waiting for 'end' to play
        self._playing_until = 0.0
        self._played_while_open: Optional[str] = None
        self._sent_id: Optional[str] = None     # reply whose frames are being sent
        self._sent_frames = 0
        self._first_sent_at = 0.0
        self._on_cancel: Optional[Callable[[], None]] = None
        self._cancelled: deque[str] = deque(maxlen=cancelled_keep)

    def try_open(self, reply_id: str,
                 on_cancel: Optional[Callable[[], None]] = None) -> OpenResult:
        with self._lock:
            if reply_id in self._cancelled:
                return "cancelled"
            if self._open_id is not None or self._playing() is not None:
                return "busy"
            self._open_id = reply_id
            self._on_cancel = on_cancel
            return "ok"

    def is_feeding(self, reply_id: str) -> bool:
        with self._lock:
            return self._open_id == reply_id

    def finish_feeding(self, reply_id: str, fed_frames: int) -> None:
        """The socket is done; the reply stays current until its audio has played."""
        with self._lock:
            if self._open_id != reply_id:
                return
            self._open_id = None
            self._on_cancel = None
            already_played = self._played_while_open == reply_id
            self._played_while_open = None
            if fed_frames > 0 and not already_played:
                self._playing_id = reply_id
                self._playing_until = self._clock() + fed_frames * FRAME_S + PLAY_GRACE_S

    def _playing(self) -> Optional[str]:
        """Playing reply id, dropping it once expired. Caller holds the lock."""
        if self._playing_id is not None and self._clock() >= self._playing_until:
            self._playing_id = None
        return self._playing_id

    def on_event(self, eventpoint: dict) -> None:
        """Avatar event hook, called for each audio frame sent; 'end' means the reply played."""
        reply_id = eventpoint.get("reply_id")
        if reply_id is not None:
            self._count_sent(reply_id, eventpoint.get("status"))
        if eventpoint.get("status") != "end":
            return
        with self._lock:
            if reply_id == self._playing_id:
                self._playing_id = None
            elif reply_id is not None and reply_id == self._open_id:
                # 'end' played before finish_feeding ran; it must not start playing then
                self._played_while_open = reply_id

    def _count_sent(self, reply_id: str, status: Optional[str]) -> None:
        with self._lock:
            if reply_id != self._sent_id:
                self._sent_id = reply_id
                self._sent_frames = 0
                self._first_sent_at = self._clock()
            self._sent_frames += 1
            sent_s = self._sent_frames * FRAME_S
            elapsed = self._clock() - self._first_sent_at
        if status == "start":
            logger.info("stream_audio reply %s first frame sent", reply_id)
        elif status == "end":
            logger.info("stream_audio reply %s sent %.2f s of audio in %.2f s",
                        reply_id, sent_s, elapsed)

    def cancel(self) -> Optional[str]:
        """Cancel the current reply (interrupt); returns its id, or None if there was none."""
        with self._lock:
            reply_id = self._open_id or self._playing()
            on_cancel = self._on_cancel
            self._open_id = None
            self._playing_id = None
            self._played_while_open = None
            self._on_cancel = None
            if reply_id is not None:
                self._cancelled.append(reply_id)
        if on_cancel is not None:
            on_cancel()
        return reply_id

    def current(self) -> Optional[str]:
        with self._lock:
            return self._open_id or self._playing()


_state_guard = threading.Lock()


def reply_state_for(avatar_session) -> ReplyState:
    """The session's ReplyState, created on first use and hooked to its avatar events."""
    state = getattr(avatar_session, "_reply_state", None)
    if state is not None:
        return state
    with _state_guard:
        state = getattr(avatar_session, "_reply_state", None)
        if state is None:
            state = ReplyState()
            hooks = getattr(avatar_session, "event_hooks", None)
            if hooks is not None:
                hooks.append(state.on_event)
            avatar_session._reply_state = state
    return state
