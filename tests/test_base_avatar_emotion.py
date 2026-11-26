"""Per-frame emotion set selection in BaseAvatar.inference() and process_frames()."""
import importlib
import queue
import sys
import threading
import types
import unittest
from unittest.mock import patch

import numpy as np

# Pure modules are imported for real first, so they stay cached once the stubs are removed.
from avatars.emotion import NEUTRAL, FrameSet
import utils.crossfade  # noqa: F401
import utils.logger  # noqa: F401


def _load_base_avatar():
    fake_torch = types.ModuleType("torch")
    fake_mp = types.ModuleType("torch.multiprocessing")
    fake_mp.Event = threading.Event
    fake_torch.multiprocessing = fake_mp

    fake_cv2 = types.ModuleType("cv2")
    fake_cv2.FONT_HERSHEY_SIMPLEX = 0
    fake_cv2.addWeighted = lambda a, wa, b, wb, gamma: (a * wa + b * wb + gamma).astype(a.dtype)
    fake_cv2.putText = lambda *args, **kwargs: None

    fake_av = types.ModuleType("av")
    fake_av.AudioFrame = object
    fake_av.VideoFrame = object

    fake_tqdm = types.ModuleType("tqdm")
    fake_tqdm.tqdm = lambda iterable, **kwargs: iterable

    stubs = {
        "torch": fake_torch,
        "torch.multiprocessing": fake_mp,
        "cv2": fake_cv2,
        "av": fake_av,
        "resampy": types.ModuleType("resampy"),
        "soundfile": types.ModuleType("soundfile"),
        "registry": types.ModuleType("registry"),
        "tqdm": fake_tqdm,
    }
    # patch.dict drops every module imported inside the block, so no stub leaks into other tests.
    with patch.dict(sys.modules, stubs):
        sys.modules.pop("avatars.base_avatar", None)
        sys.modules.pop("utils.image", None)
        return importlib.import_module("avatars.base_avatar")


base = _load_base_avatar()
mirror_index = base.mirror_index
AudioFrameData = base.AudioFrameData

BATCH = 4


def make_set(name, length):
    frames = [np.full((4, 4, 3), i, dtype=np.uint8) for i in range(length)]
    faces = [np.full((2, 2, 3), i, dtype=np.uint8) for i in range(length)]
    coords = [(0, 2, 0, 2)] * length
    return FrameSet(name, frames, faces, coords)


def speech(emotion=None):
    userdata = {} if emotion is None else {"emotion": emotion}
    return AudioFrameData(np.zeros(320, dtype=np.float32), 0, userdata)


def silence():
    return AudioFrameData(np.zeros(320, dtype=np.float32), 1, {})


def video_frames(n, make):
    """Audio frames for n video frames (two audio frames each)."""
    return [make() for _ in range(2 * n)]


class FakeController:
    """Maps labels to sets the way EmotionController does: unknown or not ready -> neutral."""

    def __init__(self, neutral, sets):
        self.neutral = neutral
        self.sets = sets
        self.active_name = NEUTRAL

    def frame_set(self, name):
        return self.sets.get(name, self.neutral)


class FeatQueue:
    """Hands out the preloaded batches, then stops the loop it feeds."""

    def __init__(self, n_batches, quit_event):
        self.remaining = n_batches
        self.quit_event = quit_event

    def get(self, block=True, timeout=None):
        if self.remaining == 0:
            self.quit_event.set()
            raise queue.Empty
        self.remaining -= 1
        return [np.zeros((80, 16))] * BATCH


class ListSink:
    def __init__(self):
        self.items = []

    def put(self, item, *args, **kwargs):
        self.items.append(item)


class RecordingAvatar(base.BaseAvatar):
    """BaseAvatar with infer_frames recording the refs it was handed."""

    def infer_frames(self, refs, start_index, audiofeat_batch):
        self.infer_calls.append((list(refs), start_index))
        return [np.zeros((2, 2, 3)) for _ in refs]


class LegacyAvatar(base.BaseAvatar):
    """Overrides only inference_batch, like musetalk/ultralight do."""

    def inference_batch(self, index, audiofeat_batch):
        self.batch_calls.append(index)
        return [np.zeros((2, 2, 3)) for _ in range(self.batch_size)]


def make_avatar(audio, neutral_len=10, sets=None, cls=RecordingAvatar, custom_index=None):
    avatar = object.__new__(cls)
    avatar.frame_list_cycle = [np.full((4, 4, 3), i, dtype=np.uint8) for i in range(neutral_len)]
    avatar.face_list_cycle = [np.full((2, 2, 3), i, dtype=np.uint8) for i in range(neutral_len)]
    avatar.coord_list_cycle = [(0, 2, 0, 2)] * neutral_len
    avatar.batch_size = BATCH
    avatar.custom_index = {} if custom_index is None else custom_index
    avatar.res_frame_queue = ListSink()
    avatar.infer_calls = []
    avatar.batch_calls = []
    avatar.emotion = None
    if sets is not None:
        avatar.attach_emotions(FakeController(avatar.neutral_set, sets))
    assert len(audio) % (2 * BATCH) == 0, "audio must fill whole batches"
    quit_event = threading.Event()
    output_queue = queue.Queue()
    for frame in audio:
        output_queue.put(frame)
    avatar.asr = types.SimpleNamespace(
        feat_queue=FeatQueue(len(audio) // (2 * BATCH), quit_event),
        output_queue=output_queue,
    )
    return avatar, quit_event


def run_inference(audio, **kwargs):
    avatar, quit_event = make_avatar(audio, **kwargs)
    avatar.inference(quit_event)
    return avatar, avatar.res_frame_queue.items


class InferenceSelectionTestCase(unittest.TestCase):
    def setUp(self):
        self.happy = make_set("avatar_happy", 10)
        self.sad = make_set("avatar_sad", 10)
        self.sets = {"happy": self.happy, "sad": self.sad}
        self.audio = video_frames(6, lambda: speech("happy")) + video_frames(6, lambda: speech("sad"))

    def test_each_item_carries_set_of_its_face_crop(self):
        avatar, items = run_inference(self.audio, sets=self.sets)
        refs = [ref for call_refs, _ in avatar.infer_calls for ref in call_refs]
        self.assertEqual(len(items), 12)
        for (res, _, idx, frame_set), (ref_set, ref_idx) in zip(items, refs):
            self.assertIs(frame_set, ref_set)
            self.assertEqual(idx, ref_idx)
            self.assertLess(idx, len(frame_set))
        self.assertEqual([item[3] for item in items], [self.happy] * 6 + [self.sad] * 6)

    def test_batch_straddling_set_change(self):
        avatar, _ = run_inference(self.audio, sets=self.sets)
        second_batch_refs, _ = avatar.infer_calls[1]
        self.assertEqual([fs for fs, _ in second_batch_refs], [self.happy] * 2 + [self.sad] * 2)

    def test_index_restarts_on_set_change(self):
        _, items = run_inference(self.audio, sets=self.sets)
        self.assertEqual([item[2] for item in items], [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5])

    def test_silence_uses_neutral(self):
        avatar, items = run_inference(video_frames(BATCH, silence), sets=self.sets)
        self.assertEqual(len(items), BATCH)
        for res, _, _, frame_set in items:
            self.assertIsNone(res)
            self.assertIs(frame_set, avatar.neutral_set)
        self.assertEqual(avatar.infer_calls, [])

    def test_no_controller_behaves_as_today(self):
        avatar, items = run_inference(video_frames(12, speech), neutral_len=5, cls=LegacyAvatar)
        neutral = avatar.neutral_set
        for _, _, _, frame_set in items:
            self.assertIs(frame_set, neutral)
        self.assertEqual([item[2] for item in items], [mirror_index(5, k) for k in range(12)])
        self.assertEqual(avatar.batch_calls, [0, 4, 8])

    def test_no_controller_silence_then_speech_with_custom_clip_restarts(self):
        audio = video_frames(BATCH, silence) + video_frames(BATCH, speech)
        avatar, items = run_inference(audio, neutral_len=5, cls=LegacyAvatar, custom_index={1: 0})
        self.assertEqual([item[2] for item in items], [0, 1, 2, 3, 0, 1, 2, 3])
        self.assertEqual(avatar.batch_calls, [0])

    def test_sets_of_different_lengths_mirror_within_own_length(self):
        sets = {"happy": make_set("avatar_happy", 3)}
        _, items = run_inference(video_frames(8, lambda: speech("happy")), sets=sets)
        self.assertEqual([item[2] for item in items], [0, 1, 2, 2, 1, 0, 0, 1])

    def test_short_pause_inside_speech_batch_keeps_set(self):
        audio = (
            video_frames(2, lambda: speech("happy"))
            + video_frames(1, silence)
            + video_frames(1, lambda: speech("happy"))
        )
        _, items = run_inference(audio, sets=self.sets)
        self.assertEqual([item[3] for item in items], [self.happy] * 4)
        self.assertEqual([item[2] for item in items], [0, 1, 2, 3])

    def test_revert_to_neutral_after_silent_batch(self):
        audio = video_frames(BATCH, lambda: speech("happy")) + video_frames(BATCH, silence)
        avatar, items = run_inference(audio, sets=self.sets)
        self.assertEqual([item[3] for item in items[BATCH:]], [avatar.neutral_set] * BATCH)
        self.assertEqual([item[2] for item in items[BATCH:]], [0, 1, 2, 3])

    def test_revert_to_neutral_resets_active_name(self):
        audio = video_frames(BATCH, lambda: speech("happy")) + video_frames(BATCH, silence)
        avatar, _ = run_inference(audio, sets=self.sets)
        self.assertEqual(avatar.emotion.active_name, NEUTRAL)

    def test_speech_pair_takes_emotion_of_its_first_speech_frame(self):
        audio = [silence(), speech("sad")] + video_frames(BATCH - 1, lambda: speech("sad"))
        _, items = run_inference(audio, sets=self.sets)
        self.assertEqual([item[3] for item in items], [self.sad] * BATCH)

    def test_active_name_follows_speech_label(self):
        avatar, _ = run_inference(video_frames(BATCH, lambda: speech("happy")), sets=self.sets)
        self.assertEqual(avatar.emotion.active_name, "happy")

    def test_active_name_is_neutral_when_set_not_ready(self):
        avatar, items = run_inference(video_frames(BATCH, lambda: speech("angry")), sets=self.sets)
        self.assertEqual(avatar.emotion.active_name, NEUTRAL)
        self.assertEqual([item[3] for item in items], [avatar.neutral_set] * BATCH)


class NeutralSetTestCase(unittest.TestCase):
    def test_neutral_set_is_built_once_from_avatar_lists(self):
        avatar, _ = make_avatar(video_frames(BATCH, silence))
        neutral = avatar.neutral_set
        self.assertIs(avatar.neutral_set, neutral)
        self.assertEqual(neutral.name, NEUTRAL)
        self.assertIs(neutral.frames, avatar.frame_list_cycle)
        self.assertIs(neutral.faces, avatar.face_list_cycle)
        self.assertIs(neutral.coords, avatar.coord_list_cycle)

    def test_avatar_length_follows_active_set(self):
        avatar, _ = make_avatar(video_frames(BATCH, silence), sets={"happy": make_set("h", 3)})
        self.assertEqual(avatar.get_avatar_length(), 10)
        avatar.emotion.active_name = "happy"
        self.assertEqual(avatar.get_avatar_length(), 3)


class FakeCrossFade:
    instances = []

    def __init__(self, *args, **kwargs):
        self.applied = 0
        self.started_at = []
        FakeCrossFade.instances.append(self)

    def start(self, frame):
        self.started_at.append(self.applied)

    def apply(self, target):
        self.applied += 1
        return target


class ItemQueue:
    def __init__(self, items, quit_event):
        self.items = list(items)
        self.quit_event = quit_event

    def get(self, block=True, timeout=None):
        if not self.items:
            self.quit_event.set()
            raise queue.Empty
        return self.items.pop(0)


class FakeOutput:
    def __init__(self):
        self.video = []

    def start(self):
        pass

    def stop(self):
        pass

    def push_video_frame(self, frame):
        self.video.append(frame)

    def push_audio_frame(self, frame, userdata):
        pass


class PastingAvatar(base.BaseAvatar):
    def paste_frame(self, res_frame, idx, frame_set):
        self.pasted.append((idx, frame_set))
        return frame_set.frames[idx].copy()


class ProcessFramesTestCase(unittest.TestCase):
    def run_process_frames(self, items):
        avatar = object.__new__(PastingAvatar)
        avatar.custom_index = {}
        avatar.custom_img_cycle = {}
        avatar.width = avatar.height = 0
        avatar.recording = False
        avatar.speaking = False
        avatar.pasted = []
        avatar.output = FakeOutput()
        quit_event = threading.Event()
        avatar.res_frame_queue = ItemQueue(items, quit_event)
        FakeCrossFade.instances = []
        with patch.object(base, "CrossFade", FakeCrossFade):
            avatar.process_frames(quit_event)
        return avatar, FakeCrossFade.instances[0]

    def test_process_frames_fades_on_talking_set_change(self):
        happy, sad = make_set("avatar_happy", 4), make_set("avatar_sad", 4)
        res = np.zeros((2, 2, 3))
        items = [
            (res, [speech("happy")] * 2, 0, happy),
            (res, [speech("happy")] * 2, 1, happy),
            (res, [speech("sad")] * 2, 0, sad),
            (res, [speech("sad")] * 2, 1, sad),
        ]
        avatar, crossfade = self.run_process_frames(items)
        # Item 0 fades in from listening; item 2 is the happy -> sad boundary.
        self.assertEqual(crossfade.started_at, [0, 2])
        self.assertEqual(avatar.pasted, [(0, happy), (1, happy), (0, sad), (1, sad)])

    def test_process_frames_silent_frame_uses_item_set(self):
        neutral = make_set(NEUTRAL, 4)
        items = [(None, [silence()] * 2, 3, neutral)]
        avatar, crossfade = self.run_process_frames(items)
        np.testing.assert_array_equal(avatar.output.video[0], neutral.frames[3])
        self.assertEqual(crossfade.started_at, [])


if __name__ == "__main__":
    unittest.main()
