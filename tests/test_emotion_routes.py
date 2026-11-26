import importlib
import io
import warnings
import json
import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock

import aiohttp
from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase

from avatars.emotion import (
    EmotionConfig,
    EmotionController,
    EmotionSetStore,
)
from test_emotion_store import FakeLoader, make_set

LABELS = ("neutral", "happy", "sad")
NEUTRAL_SET = make_set(name="neutral")
_STUBBED = ("avatars.base_avatar", "server.avatar_routes", "server.admin_routes")

routes = None
_saved_modules: dict = {}
_had_attr = False


def setUpModule():
    """Import server.routes with its heavy imports stubbed, undoing it in tearDownModule."""
    global routes, _had_attr
    for name in _STUBBED + ("server.routes", "server.session_manager"):
        _saved_modules[name] = sys.modules.pop(name, None)
    base = types.ModuleType("avatars.base_avatar")
    base.BaseAvatar = type("BaseAvatar", (), {})
    sys.modules["avatars.base_avatar"] = base
    for name, fn in (("server.avatar_routes", "setup_avatar_routes"),
                     ("server.admin_routes", "setup_admin_routes")):
        stub = types.ModuleType(name)
        setattr(stub, fn, lambda app: None)
        sys.modules[name] = stub
    routes = importlib.import_module("server.routes")


def tearDownModule():
    for name, mod in _saved_modules.items():
        if mod is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = mod
    server_pkg = sys.modules.get("server")
    for attr in ("routes", "session_manager"):
        if server_pkg is not None and hasattr(server_pkg, attr):
            delattr(server_pkg, attr)


def make_controller():
    loader = FakeLoader()
    store = EmotionSetStore(loader, start_thread=lambda fn: None)
    controller = EmotionController(
        NEUTRAL_SET, EmotionConfig({"happy": "h"}, True), store, LABELS, 30.0)
    store.preload(["h"], NEUTRAL_SET)
    return controller


class FakeSession:
    def __init__(self, avatar_id, with_emotion=True):
        self.opt = SimpleNamespace(avatar_id=avatar_id, model="wav2lip", REF_FILE="",
                                   transport="webrtc", batch_size=16, customopt=[])
        self.emotion = make_controller() if with_emotion else None
        self.datainfos = []
        self.flushed = 0

    def put_audio_file(self, filebytes, datainfo):
        self.datainfos.append(dict(datainfo))

    def put_msg_txt(self, text, datainfo):
        self.datainfos.append(dict(datainfo))

    def flush_talk(self):
        self.flushed += 1

    def is_speaking(self):
        return False


class RouteTestCase(AioHTTPTestCase):
    async def get_application(self):
        app = web.Application()
        with warnings.catch_warnings():
            # production stores opt under a plain string key, and the routes read it that way
            warnings.simplefilter("ignore", web.NotAppKeyWarning)
            app["opt"] = SimpleNamespace(emotion_labels_parsed=LABELS)
        app.router.add_post("/set_emotion", routes.set_emotion)
        app.router.add_post("/humanaudio", routes.humanaudio)
        app.router.add_post("/human", routes.human)
        app.router.add_post("/interrupt_talk", routes.interrupt_talk)
        app.router.add_get("/api/admin/sessions", routes.admin_sessions)
        return app

    def use_sessions(self, sessions):
        patcher = mock.patch.object(routes.session_manager, "sessions", sessions)
        patcher.start()
        self.addCleanup(patcher.stop)
        return sessions

    async def post_json(self, path, body):
        resp = await self.client.post(path, data=json.dumps(body))
        return await resp.json()

    async def upload(self, sessionid, emotion=None):
        form = aiohttp.FormData()
        form.add_field("sessionid", sessionid)
        form.add_field("file", io.BytesIO(b"RIFF"), filename="a.wav")
        if emotion is not None:
            form.add_field("emotion", emotion)
        resp = await self.client.post("/humanaudio", data=form)
        return await resp.json()


class SetEmotionTests(RouteTestCase):
    async def test_set_emotion_by_avatar_id_targets_all_matching(self):
        a, b, c = FakeSession("sylvia_3"), FakeSession("sylvia_3"), FakeSession("other")
        self.use_sessions({"1": a, "2": b, "3": c})
        body = await self.post_json("/set_emotion", {"avatar_id": "sylvia_3", "emotion": "happy"})
        self.assertEqual(body["data"], {"emotion": "happy", "sessions": 2})
        self.assertEqual((a.emotion.pending_name, b.emotion.pending_name), ("happy", "happy"))
        self.assertIsNone(c.emotion.pending_name)

    async def test_set_emotion_by_session(self):
        a, b = FakeSession("x"), FakeSession("x")
        self.use_sessions({"1": a, "2": b})
        body = await self.post_json("/set_emotion", {"sessionid": "2", "emotion": "sad"})
        self.assertEqual(body["data"], {"emotion": "sad", "sessions": 1})
        self.assertIsNone(a.emotion.pending_name)
        self.assertEqual(b.emotion.pending_name, "sad")

    async def test_set_emotion_unknown_session_errors(self):
        self.use_sessions({"1": FakeSession("x")})
        body = await self.post_json("/set_emotion", {"sessionid": "nope", "emotion": "sad"})
        self.assertEqual((body["code"], body["msg"]), (-1, "session not found"))

    async def test_set_emotion_invalid_name_resolves_neutral(self):
        a = FakeSession("x")
        self.use_sessions({"1": a})
        body = await self.post_json("/set_emotion", {"sessionid": "1", "emotion": "furious"})
        self.assertEqual(body["data"], {"emotion": "neutral", "sessions": 1})
        self.assertEqual(a.emotion.pending_name, "neutral")

    async def test_set_emotion_requires_target(self):
        a = FakeSession("x")
        self.use_sessions({"1": a})
        body = await self.post_json("/set_emotion", {"emotion": "happy"})
        self.assertEqual(body["code"], -1)
        resp = await self.client.post("/set_emotion", data="not json")
        self.assertEqual((await resp.json())["code"], -1)
        self.assertIsNone(a.emotion.pending_name)

    async def test_set_emotion_skips_sessions_without_controller(self):
        a, b = FakeSession("x", with_emotion=False), FakeSession("x")
        self.use_sessions({"1": a, "2": b, "3": None})
        body = await self.post_json("/set_emotion", {"avatar_id": "x", "emotion": "happy"})
        self.assertEqual(body["data"]["sessions"], 1)


class UploadTests(RouteTestCase):
    async def test_humanaudio_field_tags_datainfo(self):
        a = FakeSession("x")
        self.use_sessions({"1": a})
        await self.upload("1", emotion="happy")
        self.assertEqual(a.datainfos[0]["emotion"], "happy")

    async def test_humanaudio_without_field_uses_pending(self):
        a = FakeSession("x")
        self.use_sessions({"1": a})
        await self.post_json("/set_emotion", {"sessionid": "1", "emotion": "happy"})
        await self.upload("1")
        self.assertEqual(a.datainfos[0]["emotion"], "happy")

    async def test_interrupt_keeps_pending(self):
        a = FakeSession("x")
        self.use_sessions({"1": a})
        await self.post_json("/set_emotion", {"sessionid": "1", "emotion": "happy"})
        await self.post_json("/interrupt_talk", {"sessionid": "1"})
        await self.upload("1")
        self.assertEqual(a.flushed, 1)
        self.assertEqual(a.datainfos[0]["emotion"], "happy")

    async def test_humanaudio_without_controller_adds_no_emotion(self):
        a = FakeSession("x", with_emotion=False)
        self.use_sessions({"1": a})
        await self.upload("1", emotion="happy")
        self.assertNotIn("emotion", a.datainfos[0])

    async def test_human_echo_tags_datainfo(self):
        a = FakeSession("x")
        self.use_sessions({"1": a})
        await self.post_json("/human", {"sessionid": "1", "type": "echo", "text": "hi",
                                        "emotion": "happy"})
        self.assertEqual(a.datainfos[0]["emotion"], "happy")

    async def test_human_bad_type_keeps_pending(self):
        a = FakeSession("x")
        self.use_sessions({"1": a})
        await self.post_json("/set_emotion", {"sessionid": "1", "emotion": "happy"})
        await self.post_json("/human", {"sessionid": "1", "type": "bogus", "text": "hi"})
        self.assertEqual(a.emotion.pending_name, "happy")

    async def test_human_interrupt_keeps_pending(self):
        a = FakeSession("x")
        self.use_sessions({"1": a})
        await self.post_json("/set_emotion", {"sessionid": "1", "emotion": "happy"})
        await self.post_json("/human", {"sessionid": "1", "type": "echo", "text": "hi",
                                        "interrupt": True})
        self.assertEqual(a.flushed, 1)
        self.assertEqual(a.datainfos[0]["emotion"], "happy")


class AdminTests(RouteTestCase):
    async def test_admin_sessions_reports_emotion(self):
        a, b = FakeSession("x"), FakeSession("y", with_emotion=False)
        a.emotion.active_name = "sad"
        a.emotion.set_pending("happy")
        self.use_sessions({"1": a, "2": b})
        resp = await self.client.get("/api/admin/sessions")
        by_id = {s["sessionid"]: s for s in (await resp.json())["data"]["sessions"]}
        self.assertEqual((by_id["1"]["emotion_active"], by_id["1"]["emotion_pending"]),
                         ("sad", "happy"))
        self.assertEqual((by_id["2"]["emotion_active"], by_id["2"]["emotion_pending"]),
                         (None, None))


if __name__ == "__main__":
    unittest.main()
