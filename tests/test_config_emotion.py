import argparse
import unittest
from types import SimpleNamespace

from config import parse_bool, validate_emotion_opts


class ParseBoolTests(unittest.TestCase):
    def test_truthy_spellings(self):
        for value in ("1", "true", "YES", " On ", True):
            self.assertIs(parse_bool(value), True, value)

    def test_falsy_spellings(self):
        for value in ("0", "false", "No", "OFF", False):
            self.assertIs(parse_bool(value), False, value)

    def test_unrecognised_raises(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_bool("maybe")


class ValidateEmotionOptsTests(unittest.TestCase):
    def opts(self, **kw):
        base = {"emotion_labels": "neutral,happy", "emotion_ttl_s": 30.0}
        return SimpleNamespace(**{**base, **kw})

    def test_valid_returns_none(self):
        self.assertIsNone(validate_emotion_opts(self.opts()))

    def test_non_positive_ttl_rejected(self):
        for ttl in (0, -1.5):
            self.assertIn("emotion_ttl_s", validate_emotion_opts(self.opts(emotion_ttl_s=ttl)))

    def test_non_string_labels_rejected(self):
        for labels in (None, ["happy"], 3):
            self.assertIn("emotion_labels", validate_emotion_opts(self.opts(emotion_labels=labels)))


if __name__ == "__main__":
    unittest.main()
