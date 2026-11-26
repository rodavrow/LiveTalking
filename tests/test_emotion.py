import json
import tempfile
import unittest
from pathlib import Path

from avatars import emotion as emotion_module
from avatars.emotion import (
    NEUTRAL,
    EmotionConfig,
    choose_custom,
    PendingEmotion,
    parse_custom_config,
    parse_labels,
    read_avatar_config,
    resolve_label,
)
from utils.logger import logger

LABELS = ("neutral", "happy", "sad")


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class LabelTests(unittest.TestCase):
    def test_parse_labels_adds_neutral_and_normalises(self):
        self.assertEqual(parse_labels(" Happy,sad,happy "), ("neutral", "happy", "sad"))

    def test_parse_labels_empty_raises(self):
        with self.assertRaises(ValueError):
            parse_labels(" , ")

    def test_resolve_label_case_and_unknown(self):
        self.assertEqual(resolve_label(" HAPPY ", LABELS), "happy")
        with self.assertLogs(logger, "WARNING"):
            self.assertEqual(resolve_label("furious", LABELS), NEUTRAL)
        with self.assertNoLogs(logger, "WARNING"):
            self.assertEqual(resolve_label(None, LABELS), NEUTRAL)
            self.assertEqual(resolve_label("", LABELS), NEUTRAL)


class ParseCustomConfigTests(unittest.TestCase):
    def setUp(self):
        emotion_module._warned_keys.clear()

    def test_preload_must_be_a_real_bool(self):
        for value in ("false", 0, 1, None, "true"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "emotion_preload"):
                parse_custom_config({"emotion_preload": value}, LABELS)

    def test_preload_bool_accepted(self):
        _, cfg = parse_custom_config({"emotion_preload": False}, LABELS)
        self.assertIs(cfg.preload, False)

    def test_ignored_key_warned_once_per_source(self):
        raw = {"emotions": {"neutral": "x", "bogus": "y"}}
        with self.assertLogs(logger, "WARNING") as logs:
            for _ in range(3):
                parse_custom_config(raw, LABELS, "a.json")
            parse_custom_config(raw, LABELS, "b.json")
        self.assertEqual(len(logs.records), 4)

    def test_plain_list_is_custom_only(self):
        custom = [{"audiotype": 1, "imgpath": "x", "audiopath": ""}]
        self.assertEqual(parse_custom_config(custom, LABELS), (custom, EmotionConfig({}, True)))

    def test_object_form(self):
        raw = {
            "custom": [{"audiotype": 1, "imgpath": "x", "audiopath": ""}],
            "emotions": {"Happy": "a_happy", "neutral": "x", "bogus": "y"},
            "emotion_preload": False,
        }
        with self.assertLogs(logger, "WARNING") as logs:
            custom, emotions = parse_custom_config(raw, LABELS)
        self.assertEqual(custom, raw["custom"])
        self.assertEqual(emotions.emotions, {"happy": "a_happy"})
        self.assertIs(emotions.preload, False)
        text = "\n".join(logs.output)
        self.assertIn("neutral", text)
        self.assertIn("bogus", text)
        self.assertEqual(len(logs.records), 2)

    def test_object_form_missing_custom_defaults_empty(self):
        custom, emotions = parse_custom_config({"emotions": {}}, LABELS)
        self.assertEqual(custom, [])
        self.assertEqual(emotions, EmotionConfig({}, True))

    def test_bad_shape_raises(self):
        for raw in ("str", {"custom": 3}, {"emotions": []}):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_custom_config(raw, LABELS)


class ReadAvatarConfigTests(unittest.TestCase):
    def test_read_avatar_config_missing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(read_avatar_config(tmp, LABELS), (None, EmotionConfig.empty()))

    def test_read_avatar_config_bad_json_names_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "custom_config.json"
            path.write_text("{nope", encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                read_avatar_config(tmp, LABELS)
            self.assertIn(str(path), str(ctx.exception))

    def test_read_avatar_config_reads_object_form(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = {"custom": [], "emotions": {"sad": "a_sad"}}
            (Path(tmp) / "custom_config.json").write_text(json.dumps(raw), encoding="utf-8")
            self.assertEqual(
                read_avatar_config(tmp, LABELS), ([], EmotionConfig({"sad": "a_sad"}, True))
            )


class PendingEmotionTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.pending = PendingEmotion(30.0, clock=self.clock)

    def test_pending_take_within_ttl_then_cleared(self):
        self.pending.set("happy")
        self.clock.now = 29.9
        self.assertEqual(self.pending.take(), "happy")
        self.assertIsNone(self.pending.take())

    def test_pending_expired(self):
        self.pending.set("happy")
        self.clock.now = 30.1
        self.assertIsNone(self.pending.take())

    def test_pending_set_replaces(self):
        self.pending.set("happy")
        self.pending.set("sad")
        self.assertEqual(self.pending.take(), "sad")

    def test_peek_does_not_clear_and_respects_ttl(self):
        self.pending.set("happy")
        self.assertEqual(self.pending.peek(), "happy")
        self.assertEqual(self.pending.peek(), "happy")
        self.clock.now = 30.1
        self.assertIsNone(self.pending.peek())


class ChooseCustomTests(unittest.TestCase):
    FILE = [{"audiotype": 2}]
    CLI = [{"audiotype": 3}]

    def test_request_wins_over_file_and_cli(self):
        result = choose_custom('[{"audiotype": 1}]', self.FILE, self.CLI, LABELS)
        self.assertEqual(result, [{"audiotype": 1}])

    def test_file_used_when_no_request(self):
        self.assertEqual(choose_custom(None, self.FILE, self.CLI, LABELS), self.FILE)
        self.assertEqual(choose_custom("", self.FILE, self.CLI, LABELS), self.FILE)

    def test_empty_file_list_still_beats_cli(self):
        self.assertEqual(choose_custom(None, [], self.CLI, LABELS), [])

    def test_cli_used_last(self):
        self.assertEqual(choose_custom(None, None, self.CLI, LABELS), self.CLI)

    def test_request_emotions_warned_and_ignored(self):
        raw = json.dumps({"custom": [{"audiotype": 1}], "emotions": {"happy": "x"}})
        with self.assertLogs(logger, "WARNING"):
            result = choose_custom(raw, None, self.CLI, LABELS)
        self.assertEqual(result, [{"audiotype": 1}])

    def test_request_without_emotions_does_not_warn(self):
        with self.assertNoLogs(logger, "WARNING"):
            choose_custom("[]", None, self.CLI, LABELS)

    def test_bad_request_json_raises(self):
        with self.assertRaises(ValueError):
            choose_custom("{not json", None, self.CLI, LABELS)


if __name__ == "__main__":
    unittest.main()
