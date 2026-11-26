import unittest

from avatars.emotion import (
    NEUTRAL,
    EmotionConfig,
    EmotionController,
    EmotionSetStore,
)
from test_emotion_store import Capture, FakeLoader, make_set

LABELS = ("neutral", "happy", "sad", "angry")
NEUTRAL_SET = make_set(name="neutral")


class SteppingClock:
    """Advances by `step` seconds on every reading."""

    def __init__(self, step=0.0):
        self.now = 0.0
        self.step = step

    def __call__(self):
        value = self.now
        self.now += self.step
        return value


def make_controller(preload=False, store="default", clock=None, ready=True):
    capture = Capture()
    loader = FakeLoader()
    if store == "default":
        store = EmotionSetStore(loader, start_thread=capture)
    config = EmotionConfig({"happy": "h", "sad": "s"}, preload)
    kwargs = {"clock": clock} if clock else {}
    controller = EmotionController(NEUTRAL_SET, config, store, LABELS, 30.0, **kwargs)
    if store is not None and ready:
        store.preload(["h", "s"], NEUTRAL_SET)
    return controller, store, capture


class EmotionControllerTest(unittest.TestCase):
    def test_field_wins_and_clears_pending(self):
        c, _, _ = make_controller()
        c.set_pending("sad")
        self.assertEqual(c.emotion_for_upload("happy"), "happy")
        self.assertIsNone(c.pending_name)

    def test_empty_field_falls_through_to_pending(self):
        c, _, _ = make_controller()
        c.set_pending("sad")
        self.assertEqual(c.emotion_for_upload(""), "sad")

    def test_no_field_no_pending_is_neutral(self):
        c, _, _ = make_controller()
        self.assertEqual(c.emotion_for_upload(None), NEUTRAL)

    def test_expired_pending_is_neutral(self):
        clock = SteppingClock(step=31.0)
        c, _, _ = make_controller(clock=clock)
        c.set_pending("sad")
        self.assertEqual(c.emotion_for_upload(None), NEUTRAL)

    def test_emotion_without_set_is_neutral(self):
        c, _, _ = make_controller()
        self.assertEqual(c.emotion_for_upload("angry"), NEUTRAL)

    def test_set_pending_returns_resolved_label(self):
        c, _, _ = make_controller()
        self.assertEqual(c.set_pending(" Sad "), "sad")
        self.assertEqual(c.set_pending("bogus"), NEUTRAL)

    def test_lazy_first_upload_neutral_second_emotion(self):
        c, _, capture = make_controller(preload=False, ready=False)
        self.assertEqual(c.emotion_for_upload("happy"), NEUTRAL)
        capture.fns[0]()
        self.assertEqual(c.emotion_for_upload("happy"), "happy")

    def test_preload_mode_unloaded_set_is_neutral_and_starts_no_load(self):
        c, _, capture = make_controller(preload=True, ready=False)
        self.assertEqual(c.emotion_for_upload("happy"), NEUTRAL)
        self.assertEqual(capture.fns, [])

    def test_frame_set_ready_returns_loaded_set(self):
        c, store, _ = make_controller()
        self.assertIs(c.frame_set("happy"), store.get("h", NEUTRAL_SET, lazy=False))

    def test_frame_set_never_starts_load(self):
        c, _, capture = make_controller(ready=False)
        self.assertIs(c.frame_set("happy"), NEUTRAL_SET)
        self.assertEqual(capture.fns, [])

    def test_frame_set_unknown_returns_neutral_object(self):
        c, _, _ = make_controller()
        self.assertIs(c.frame_set("bogus"), NEUTRAL_SET)

    def test_frame_set_neutral_and_none_return_neutral_object(self):
        c, _, _ = make_controller()
        self.assertIs(c.frame_set(NEUTRAL), NEUTRAL_SET)
        self.assertIs(c.frame_set(None), NEUTRAL_SET)

    def test_store_none_always_neutral(self):
        c, _, _ = make_controller(store=None)
        c.set_pending("sad")
        self.assertEqual(c.emotion_for_upload("happy"), NEUTRAL)
        self.assertIs(c.frame_set("happy"), NEUTRAL_SET)

    def test_preload_loads_all_when_enabled(self):
        c, store, _ = make_controller(preload=True, ready=False)
        c.preload()
        self.assertEqual(store.status("h"), "ready")
        self.assertEqual(store.status("s"), "ready")

    def test_preload_noop_when_disabled(self):
        c, store, _ = make_controller(preload=False, ready=False)
        c.preload()
        self.assertEqual(store.status("h"), "absent")

    def test_active_name_defaults_to_neutral(self):
        c, _, _ = make_controller()
        self.assertEqual(c.active_name, NEUTRAL)


if __name__ == "__main__":
    unittest.main()
