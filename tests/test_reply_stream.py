import unittest

import numpy as np

from server.reply_stream import (
    FRAME_SAMPLES,
    PLAY_GRACE_S,
    PcmFramer,
    ReplyFeeder,
    ReplyState,
    valid_reply_id,
)


def pcm(*samples: int) -> bytes:
    return np.array(samples, dtype="<i2").tobytes()


def frame_bytes(value: int) -> bytes:
    return pcm(*([value] * FRAME_SAMPLES))


class PcmFramerTest(unittest.TestCase):
    def test_whole_frame_becomes_float32_frame(self):
        frames = PcmFramer().feed(frame_bytes(16384))
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0].dtype, np.float32)
        self.assertEqual(frames[0].shape, (FRAME_SAMPLES,))
        self.assertAlmostEqual(float(frames[0][0]), 0.5)

    def test_partial_frame_is_held_until_complete(self):
        framer = PcmFramer()
        data = frame_bytes(100)
        self.assertEqual(framer.feed(data[:300]), [])
        self.assertEqual(len(framer.feed(data[300:])), 1)

    def test_sample_split_across_messages_is_carried_over(self):
        framer = PcmFramer()
        data = frame_bytes(-2)  # 0xFFFE: wrong if bytes are realigned
        frames = framer.feed(data[:1]) + framer.feed(data[1:])
        self.assertEqual(len(frames), 1)
        self.assertTrue(np.all(frames[0] == np.float32(-2 / 32768)))

    def test_flush_pads_remainder_with_silence(self):
        framer = PcmFramer()
        framer.feed(pcm(32767, 32767, 32767))
        last = framer.flush()
        self.assertEqual(last.shape, (FRAME_SAMPLES,))
        self.assertGreater(float(last[2]), 0.99)
        self.assertEqual(float(last[3]), 0.0)

    def test_flush_drops_lone_odd_byte(self):
        framer = PcmFramer()
        framer.feed(b"\x01")
        self.assertIsNone(framer.flush())

    def test_flush_with_nothing_buffered_returns_none(self):
        self.assertIsNone(PcmFramer().flush())

    def test_many_frames_in_one_message(self):
        frames = PcmFramer().feed(frame_bytes(1) * 3 + b"\x00")
        self.assertEqual(len(frames), 3)


class FakeSink:
    def __init__(self):
        self.frames = []

    def __call__(self, frame, eventpoint):
        self.frames.append((frame, dict(eventpoint)))

    def statuses(self):
        return [e.get("status") for _, e in self.frames]


def make_feeder(sink, prebuffer_frames=3, allowed=lambda: True):
    return ReplyFeeder(sink, {"reply_id": "r1", "emotion": "happy"},
                       prebuffer_frames=prebuffer_frames, allowed=allowed)


class ReplyFeederTest(unittest.TestCase):
    def test_nothing_is_fed_before_prebuffer_fills(self):
        sink = FakeSink()
        feeder = make_feeder(sink, prebuffer_frames=3)
        feeder.feed(frame_bytes(1) * 3)
        self.assertEqual(sink.frames, [])

    def test_prebuffer_released_once_exceeded_keeping_last_frame_back(self):
        sink = FakeSink()
        feeder = make_feeder(sink, prebuffer_frames=3)
        feeder.feed(frame_bytes(1) * 4)
        self.assertEqual(sink.statuses(), ["start", None, None])

    def test_after_prebuffer_frames_flow_with_one_held_back(self):
        sink = FakeSink()
        feeder = make_feeder(sink, prebuffer_frames=1)
        feeder.feed(frame_bytes(1) * 2)
        feeder.feed(frame_bytes(1))
        self.assertEqual(len(sink.frames), 2)

    def test_end_feeds_everything_and_marks_last_frame(self):
        sink = FakeSink()
        feeder = make_feeder(sink, prebuffer_frames=10)
        feeder.feed(frame_bytes(1) * 3 + pcm(5))
        feeder.end()
        self.assertEqual(sink.statuses(), ["start", None, None, "end"])

    def test_single_frame_reply_is_marked_end(self):
        # same as put_audio_file: on a one-frame clip 'end' replaces 'start'
        sink = FakeSink()
        feeder = make_feeder(sink)
        feeder.feed(frame_bytes(1))
        feeder.end()
        self.assertEqual(sink.statuses(), ["end"])

    def test_every_frame_carries_reply_id_and_emotion(self):
        sink = FakeSink()
        feeder = make_feeder(sink, prebuffer_frames=0)
        feeder.feed(frame_bytes(1) * 2)
        feeder.end()
        for _, eventpoint in sink.frames:
            self.assertEqual(eventpoint["reply_id"], "r1")
            self.assertEqual(eventpoint["emotion"], "happy")

    def test_end_with_no_audio_feeds_nothing(self):
        sink = FakeSink()
        make_feeder(sink).end()
        self.assertEqual(sink.frames, [])

    def test_end_twice_feeds_once(self):
        sink = FakeSink()
        feeder = make_feeder(sink)
        feeder.feed(frame_bytes(1))
        feeder.end()
        feeder.end()
        self.assertEqual(len(sink.frames), 1)

    def test_nothing_fed_when_not_allowed(self):
        sink = FakeSink()
        feeder = make_feeder(sink, prebuffer_frames=0, allowed=lambda: False)
        feeder.feed(frame_bytes(1) * 3)
        feeder.end()
        self.assertEqual(sink.frames, [])


class ReplyStateTest(unittest.TestCase):
    def test_open_sets_current(self):
        state = ReplyState()
        self.assertEqual(state.try_open("a"), "ok")
        self.assertEqual(state.current(), "a")

    def test_second_open_while_open_is_busy(self):
        state = ReplyState()
        state.try_open("a")
        self.assertEqual(state.try_open("b"), "busy")

    def test_open_while_previous_still_playing_is_busy(self):
        state = ReplyState()
        state.try_open("a")
        state.finish_feeding("a", 50)
        self.assertEqual(state.try_open("b"), "busy")

    def test_played_end_event_clears_current(self):
        state = ReplyState()
        state.try_open("a")
        state.finish_feeding("a", 50)
        state.on_event({"status": "end", "reply_id": "a"})
        self.assertIsNone(state.current())
        self.assertEqual(state.try_open("b"), "ok")

    def test_end_event_for_other_reply_is_ignored(self):
        state = ReplyState()
        state.try_open("a")
        state.finish_feeding("a", 50)
        state.on_event({"status": "end", "reply_id": "x"})
        self.assertEqual(state.current(), "a")

    def test_end_event_while_socket_still_open_keeps_reply(self):
        state = ReplyState()
        state.try_open("a")
        state.on_event({"status": "end", "reply_id": "a"})
        self.assertEqual(state.current(), "a")

    def test_regression_end_played_before_finish_feeding_clears_reply(self):
        state = ReplyState()
        state.try_open("a")
        state.on_event({"status": "end", "reply_id": "a"})
        state.finish_feeding("a", 50)
        self.assertIsNone(state.current())

    def test_finish_feeding_with_nothing_fed_clears_at_once(self):
        state = ReplyState()
        state.try_open("a")
        state.finish_feeding("a", 0)
        self.assertIsNone(state.current())

    def test_cancel_clears_and_blocks_reopen(self):
        state = ReplyState()
        state.try_open("a")
        self.assertEqual(state.cancel(), "a")
        self.assertIsNone(state.current())
        self.assertFalse(state.is_feeding("a"))
        self.assertEqual(state.try_open("a"), "cancelled")

    def test_cancel_runs_close_callback(self):
        state = ReplyState()
        closed = []
        state.try_open("a", on_cancel=lambda: closed.append(True))
        state.cancel()
        self.assertEqual(closed, [True])

    def test_cancel_with_nothing_open_returns_none(self):
        self.assertIsNone(ReplyState().cancel())

    def test_cancelled_ids_are_bounded(self):
        state = ReplyState(cancelled_keep=2)
        for rid in ("a", "b", "c"):
            state.try_open(rid)
            state.cancel()
        self.assertEqual(state.try_open("a"), "ok")

    def test_playing_reply_expires_after_its_length_plus_grace(self):
        now = [100.0]
        state = ReplyState(clock=lambda: now[0])
        state.try_open("a")
        state.finish_feeding("a", 50)  # 50 frames = 1 s of audio
        now[0] += 1.0 + PLAY_GRACE_S - 0.01
        self.assertEqual(state.current(), "a")
        now[0] += 0.02
        self.assertIsNone(state.current())
        self.assertEqual(state.try_open("b"), "ok")

    def test_feeder_counts_fed_frames(self):
        sink = FakeSink()
        feeder = make_feeder(sink, prebuffer_frames=0)
        feeder.feed(frame_bytes(1) * 3)
        feeder.end()
        self.assertEqual(feeder.fed_frames, 3)

    def test_finish_feeding_after_cancel_does_not_reopen(self):
        state = ReplyState()
        state.try_open("a")
        state.cancel()
        state.finish_feeding("a", 50)
        self.assertIsNone(state.current())


class ValidReplyIdTest(unittest.TestCase):
    def test_accepts_allowed_characters(self):
        self.assertTrue(valid_reply_id("reply_01.a:b-c"))

    def test_rejects_empty_long_and_odd_characters(self):
        for bad in ("", "a" * 65, "a b", "a/b", None):
            self.assertFalse(valid_reply_id(bad), bad)


if __name__ == "__main__":
    unittest.main()
