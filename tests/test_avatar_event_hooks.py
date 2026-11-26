import unittest

from test_base_avatar_emotion import base


def bare_avatar():
    """BaseAvatar without __init__ (which loads models); only what notify() touches."""
    avatar = base.BaseAvatar.__new__(base.BaseAvatar)
    avatar.msgqueues = []
    avatar.event_hooks = []
    return avatar


class NotifyHookTest(unittest.TestCase):
    def test_hooks_get_status_events(self):
        avatar = bare_avatar()
        seen = []
        avatar.event_hooks.append(seen.append)
        avatar.notify({"status": "end", "reply_id": "r1"})
        self.assertEqual(seen, [{"status": "end", "reply_id": "r1"}])

    def test_events_without_status_skip_hooks(self):
        avatar = bare_avatar()
        seen = []
        avatar.event_hooks.append(seen.append)
        avatar.notify({"emotion": "happy"})
        self.assertEqual(seen, [])

    def test_failing_hook_does_not_stop_later_hooks(self):
        avatar = bare_avatar()
        seen = []

        def broken(eventpoint):
            raise RuntimeError("boom")

        avatar.event_hooks.extend([broken, seen.append])
        with self.assertLogs(level="ERROR"):
            avatar.notify({"status": "start"})
        self.assertEqual(len(seen), 1)


if __name__ == "__main__":
    unittest.main()
