import threading
import unittest

import numpy as np

from avatars.emotion import EmotionSetStore, FrameSet
from utils.logger import logger


def make_set(n=3, frame_shape=(256, 256, 3), face_shape=(96, 96, 3), name="s"):
    frames = [np.zeros(frame_shape, np.uint8) for _ in range(n)]
    faces = [np.zeros(face_shape, np.uint8) for _ in range(n)]
    return FrameSet(name, frames, faces, [(0, 0, 8, 8)] * n)


REFERENCE = make_set(name="ref")


class FakeLoader:
    def __init__(self, result=None, error=None):
        self.result = result if result is not None else make_set()
        self.error = error
        self.calls = []

    def __call__(self, avatar_id, max_workers):
        self.calls.append((avatar_id, max_workers))
        if self.error:
            raise self.error
        return self.result.frames, self.result.faces, self.result.coords


class Capture:
    def __init__(self):
        self.fns = []

    def __call__(self, fn):
        self.fns.append(fn)


class EmotionSetStoreTest(unittest.TestCase):
    def test_preload_then_get_returns_same_object(self):
        store = EmotionSetStore(FakeLoader())
        store.preload(["happy"], REFERENCE)
        first = store.get("happy", REFERENCE, lazy=False)
        self.assertIsNotNone(first)
        self.assertIs(first, store.get("happy", REFERENCE, lazy=True))

    def test_lazy_get_returns_none_then_ready(self):
        capture = Capture()
        store = EmotionSetStore(FakeLoader(), start_thread=capture)
        self.assertIsNone(store.get("happy", REFERENCE, lazy=True))
        self.assertEqual(store.status("happy"), "loading")
        capture.fns[0]()
        self.assertEqual(store.status("happy"), "ready")
        self.assertIsNotNone(store.get("happy", REFERENCE, lazy=True))

    def test_concurrent_get_starts_single_load(self):
        capture = Capture()
        store = EmotionSetStore(FakeLoader(), start_thread=capture)
        barrier = threading.Barrier(8)

        def worker():
            barrier.wait()
            store.get("happy", REFERENCE, lazy=True)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(capture.fns), 1)

    def test_lazy_uses_loader_workers_preload_uses_all(self):
        loader = FakeLoader()
        store = EmotionSetStore(loader, start_thread=lambda fn: fn())
        store.get("a", REFERENCE, lazy=True)
        store.preload(["b"], REFERENCE)
        self.assertEqual(loader.calls, [("a", 1), ("b", None)])

    def test_preload_skips_known_ids(self):
        loader = FakeLoader()
        store = EmotionSetStore(loader)
        store.preload(["a", "a"], REFERENCE)
        store.preload(["a"], REFERENCE)
        self.assertEqual(len(loader.calls), 1)

    def test_mismatches_marked_failed(self):
        cases = {
            "face": make_set(face_shape=(128, 128, 3)),
            "frame": make_set(frame_shape=(128, 128, 3)),
            "length": FrameSet("x", make_set().frames, make_set().faces, [(0, 0, 1, 1)]),
            "empty": FrameSet("x", [], [], []),
        }
        for name, bad in cases.items():
            with self.subTest(name):
                store = EmotionSetStore(FakeLoader(bad), start_thread=lambda fn: fn())
                with self.assertLogs(logger, level="ERROR") as logs:
                    store.preload(["bad"], REFERENCE)
                self.assertEqual(len(logs.records), 1)
                self.assertIn("bad", logs.output[0])
                self.assertIsNone(store.get("bad", REFERENCE, lazy=False))
                self.assertEqual(store.status("bad"), "failed")

    def test_bad_frame_or_face_in_middle_marked_failed_once(self):
        def with_frame(index, value):
            good = make_set(n=4)
            frames = list(good.frames)
            frames[index] = value
            return FrameSet("x", frames, good.faces, good.coords)

        def with_face(index, value):
            good = make_set(n=4)
            faces = list(good.faces)
            faces[index] = value
            return FrameSet("x", good.frames, faces, good.coords)

        odd_face = np.zeros((64, 64, 3), np.uint8)
        odd_frame = np.zeros((128, 128, 3), np.uint8)
        cases = {
            "none frame": with_frame(2, None),
            "odd frame": with_frame(2, odd_frame),
            "none face": with_face(1, None),
            "odd face": with_face(2, odd_face),
        }
        for name, bad in cases.items():
            with self.subTest(name):
                store = EmotionSetStore(FakeLoader(bad), start_thread=lambda fn: fn())
                with self.assertLogs(logger, level="ERROR") as logs:
                    store.preload(["bad"], REFERENCE)
                self.assertEqual(len(logs.records), 1)
                self.assertEqual(store.status("bad"), "failed")
                self.assertIsNone(store.get("bad", REFERENCE, lazy=False))

    def test_missing_set_marked_failed_once(self):
        loader = FakeLoader(error=FileNotFoundError("nope"))
        store = EmotionSetStore(loader, start_thread=lambda fn: fn())
        with self.assertLogs(logger, level="ERROR") as logs:
            for _ in range(3):
                self.assertIsNone(store.get("gone", REFERENCE, lazy=True))
        self.assertEqual(len(logs.records), 1)
        self.assertIn("nope", logs.output[0])
        self.assertEqual(len(loader.calls), 1)
        self.assertEqual(store.status("gone"), "failed")

    def test_load_logs_ram(self):
        store = EmotionSetStore(FakeLoader())
        with self.assertLogs(logger, level="INFO") as logs:
            store.preload(["happy"], REFERENCE)
        ram = [m for m in logs.output if "MB" in m]
        self.assertEqual(len(ram), 1)
        self.assertIn("happy", ram[0])
        self.assertIn("3 frames", ram[0])

    def test_get_without_lazy_does_not_start_load(self):
        capture = Capture()
        store = EmotionSetStore(FakeLoader(), start_thread=capture)
        self.assertIsNone(store.get("happy", REFERENCE, lazy=False))
        self.assertEqual(capture.fns, [])
        self.assertEqual(store.status("happy"), "absent")

    def test_frameset_len_and_nbytes(self):
        fs = make_set(n=2)
        self.assertEqual(len(fs), 2)
        self.assertEqual(fs.nbytes, 2 * (256 * 256 * 3 + 96 * 96 * 3))


if __name__ == "__main__":
    unittest.main()
