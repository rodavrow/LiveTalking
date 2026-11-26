import asyncio
import json
import unittest
import warnings
from types import SimpleNamespace
from unittest import mock

import aiohttp
import numpy as np
from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase

import test_emotion_routes as emotion_routes
from test_emotion_routes import LABELS, FakeSession

FRAME = np.full(320, 1000, dtype="<i2").tobytes()


def setUpModule():
    emotion_routes.setUpModule()


def tearDownModule():
    emotion_routes.tearDownModule()


class StreamSession(FakeSession):
    def __init__(self, avatar_id="sylvia_4", with_emotion=True):
        super().__init__(avatar_id, with_emotion)
        self.frames = []
        self.event_hooks = []

    def put_audio_frame(self, frame, eventpoint):
        self.frames.append((frame, dict(eventpoint)))

    def statuses(self):
        return [e.get("status") for _, e in self.frames]

    def play_out(self):
        """What the output track does as frames are sent: notify() runs the hooks."""
        for _, eventpoint in self.frames:
            if eventpoint.get("status"):
                for hook in self.event_hooks:
                    hook(eventpoint)


class StreamAudioTests(AioHTTPTestCase):
    async def get_application(self):
        routes = emotion_routes.routes
        app = web.Application()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", web.NotAppKeyWarning)
            app["opt"] = SimpleNamespace(emotion_labels_parsed=LABELS)
        app.router.add_get("/stream_audio", routes.stream_audio)
        app.router.add_post("/interrupt_talk", routes.interrupt_talk)
        app.router.add_post("/is_speaking", routes.is_speaking)
        app.router.add_post("/set_emotion", routes.set_emotion)
        return app

    def use_session(self, session, sessionid="s1"):
        patcher = mock.patch.object(emotion_routes.routes.session_manager, "sessions",
                                    {sessionid: session})
        patcher.start()
        self.addCleanup(patcher.stop)
        return session

    async def open(self, reply_id="r1", sessionid="s1", emotion=None):
        params = {"sessionid": sessionid, "reply_id": reply_id}
        if emotion is not None:
            params["emotion"] = emotion
        ws = await self.client.ws_connect("/stream_audio", params=params)
        ready = await ws.receive_json(timeout=2)
        self.assertEqual(ready, {"type": "ready", "reply_id": reply_id})
        return ws

    async def handshake_status(self, **params):
        with self.assertRaises(aiohttp.WSServerHandshakeError) as caught:
            await self.client.ws_connect("/stream_audio", params=params)
        return caught.exception.status

    async def finish(self, ws):
        await ws.send_str(json.dumps({"type": "end"}))
        msg = await ws.receive(timeout=2)
        self.assertEqual(msg.type, aiohttp.WSMsgType.CLOSE)
        return ws.close_code

    async def speaking(self, sessionid="s1"):
        resp = await self.client.post("/is_speaking", data=json.dumps({"sessionid": sessionid}))
        return await resp.json()

    async def test_unknown_session_is_404(self):
        self.use_session(StreamSession())
        self.assertEqual(await self.handshake_status(sessionid="nope", reply_id="r1"), 404)

    async def test_bad_reply_id_is_400(self):
        self.use_session(StreamSession())
        self.assertEqual(await self.handshake_status(sessionid="s1", reply_id="a b"), 400)

    async def test_streamed_reply_is_fed_with_start_and_end(self):
        session = self.use_session(StreamSession())
        ws = await self.open(emotion="happy")
        await ws.send_bytes(FRAME * 20)
        self.assertEqual(await self.finish(ws), 1000)
        self.assertEqual(len(session.frames), 20)
        self.assertEqual(session.statuses()[0], "start")
        self.assertEqual(session.statuses()[-1], "end")
        self.assertEqual(session.frames[0][1]["reply_id"], "r1")
        self.assertEqual(session.frames[0][1]["emotion"], "happy")

    async def test_logs_received_and_fed_seconds(self):
        self.use_session(StreamSession())
        ws = await self.open()
        with self.assertLogs(level="INFO") as logs:
            await ws.send_bytes(FRAME * 50)
            await self.finish(ws)
        line = next(l for l in logs.output if "received" in l)
        self.assertIn("received 1.00 s of audio in 1 messages", line)
        self.assertIn("fed 1.00 s", line)

    async def test_split_sample_is_reassembled(self):
        session = self.use_session(StreamSession())
        ws = await self.open()
        data = FRAME * 2
        await ws.send_bytes(data[:321])
        await ws.send_bytes(data[321:])
        await self.finish(ws)
        self.assertEqual(len(session.frames), 2)
        self.assertTrue(np.allclose(session.frames[1][0], 1000 / 32768))

    async def test_missing_emotion_uses_pending_set_emotion(self):
        session = self.use_session(StreamSession())
        await self.client.post("/set_emotion",
                               data=json.dumps({"sessionid": "s1", "emotion": "happy"}))
        ws = await self.open()
        await ws.send_bytes(FRAME)
        await self.finish(ws)
        self.assertEqual(session.frames[0][1]["emotion"], "happy")

    async def test_session_without_emotions_sends_no_emotion(self):
        session = self.use_session(StreamSession(with_emotion=False))
        ws = await self.open(emotion="happy")
        await ws.send_bytes(FRAME)
        await self.finish(ws)
        self.assertNotIn("emotion", session.frames[0][1])

    async def test_second_reply_while_one_is_open_is_409(self):
        self.use_session(StreamSession())
        ws = await self.open("r1")
        self.assertEqual(await self.handshake_status(sessionid="s1", reply_id="r2"), 409)
        await self.finish(ws)

    async def test_next_reply_opens_after_previous_has_played(self):
        session = self.use_session(StreamSession())
        ws = await self.open("r1")
        await ws.send_bytes(FRAME)
        await self.finish(ws)
        self.assertEqual(await self.handshake_status(sessionid="s1", reply_id="r2"), 409)
        session.play_out()
        ws2 = await self.open("r2")
        await self.finish(ws2)

    async def test_is_speaking_reports_reply_until_played(self):
        session = self.use_session(StreamSession())
        ws = await self.open("r1")
        self.assertEqual((await self.speaking())["reply_id"], "r1")
        await ws.send_bytes(FRAME)
        await self.finish(ws)
        self.assertEqual((await self.speaking())["reply_id"], "r1")
        session.play_out()
        body = await self.speaking()
        self.assertIsNone(body["reply_id"])
        self.assertIs(body["data"], False)

    async def test_interrupt_cancels_stream_and_refuses_reopen(self):
        session = self.use_session(StreamSession())
        ws = await self.open("r1")
        await ws.send_bytes(FRAME * 20)
        await self.client.post("/interrupt_talk", data=json.dumps({"sessionid": "s1"}))
        self.assertEqual(await ws.receive_json(timeout=2), {"type": "cancelled"})
        msg = await ws.receive(timeout=2)
        self.assertEqual(msg.type, aiohttp.WSMsgType.CLOSE)
        self.assertEqual(ws.close_code, 4410)
        self.assertEqual(session.flushed, 1)
        self.assertNotIn("end", session.statuses())
        self.assertIsNone((await self.speaking())["reply_id"])
        self.assertEqual(await self.handshake_status(sessionid="s1", reply_id="r1"), 410)

    async def test_chunks_after_interrupt_are_not_fed(self):
        session = self.use_session(StreamSession())
        ws = await self.open("r1")
        await ws.send_bytes(FRAME * 20)
        await self.speaking()  # round trip, so the server has fed the 20 frames
        fed = len(session.frames)
        await self.client.post("/interrupt_talk", data=json.dumps({"sessionid": "s1"}))
        try:
            await ws.send_bytes(FRAME * 5)
        except ConnectionResetError:
            pass  # the server may already have closed the socket
        await ws.receive(timeout=2)
        await self.speaking()
        self.assertEqual(len(session.frames), fed)

    async def test_interrupt_with_no_stream_still_flushes(self):
        session = self.use_session(StreamSession())
        await self.client.post("/interrupt_talk", data=json.dumps({"sessionid": "s1"}))
        self.assertEqual(session.flushed, 1)

    async def test_close_without_end_plays_what_was_received(self):
        session = self.use_session(StreamSession())
        ws = await self.open("r1")
        await ws.send_bytes(FRAME * 3)
        await ws.close()
        await self.speaking()  # round trip, so the server has seen the close
        self.assertEqual(session.statuses(), ["start", None, "end"])
        session.play_out()
        await self.finish(await self.open("r2"))

    async def test_stall_ends_reply_and_closes(self):
        session = self.use_session(StreamSession())
        with mock.patch.object(emotion_routes.routes.reply_stream, "STALL_TIMEOUT_S", 0.05):
            ws = await self.open("r1")
            await ws.send_bytes(FRAME * 2)
            msg = await ws.receive(timeout=2)
        self.assertEqual(msg.type, aiohttp.WSMsgType.CLOSE)
        self.assertEqual(ws.close_code, 1000)
        self.assertEqual(session.statuses(), ["start", "end"])

    async def test_regression_failing_feed_does_not_block_session(self):
        session = self.use_session(StreamSession())

        def broken(frame, eventpoint):
            raise RuntimeError("queue gone")

        session.put_audio_frame = broken
        ws = await self.open("r1")
        with self.assertLogs(level="ERROR"):
            await ws.send_bytes(FRAME * 20)
            await ws.receive(timeout=2)
        self.assertIsNone((await self.speaking())["reply_id"])
        del session.put_audio_frame
        await self.finish(await self.open("r2"))

    async def test_regression_plain_get_is_400_and_keeps_pending_emotion(self):
        session = self.use_session(StreamSession())
        await self.client.post("/set_emotion",
                               data=json.dumps({"sessionid": "s1", "emotion": "happy"}))
        resp = await self.client.get("/stream_audio",
                                     params={"sessionid": "s1", "reply_id": "r1"})
        self.assertEqual(resp.status, 400)
        self.assertEqual(session.emotion.pending_name, "happy")
        self.assertIsNone((await self.speaking())["reply_id"])

    async def test_text_other_than_end_is_ignored(self):
        session = self.use_session(StreamSession())
        ws = await self.open("r1")
        await ws.send_str("hello")
        await ws.send_bytes(FRAME)
        await self.finish(ws)
        self.assertEqual(session.statuses(), ["end"])


class FakeWs:
    def __init__(self, prepared):
        self.prepared = prepared
        self.closed = False
        self.sent = []

    async def send_json(self, data):
        self.sent.append(data)

    async def close(self, code, message=b""):
        self.closed = True
        self.sent.append(code)


class CancelCloserTest(unittest.IsolatedAsyncioTestCase):
    async def test_regression_interrupt_during_upgrade_still_sends_cancelled(self):
        ws = FakeWs(prepared=False)
        closer = emotion_routes.routes._CancelCloser(ws)
        closer.start()           # interrupt lands while prepare() is still running
        await asyncio.sleep(0)   # let anything start() scheduled run before the upgrade ends
        ws.prepared = True       # upgrade finishes
        await closer.wait()
        self.assertEqual(ws.sent, [{"type": "cancelled"}, 4410])

    async def test_cancelled_is_sent_once_when_both_sides_close(self):
        ws = FakeWs(prepared=True)
        closer = emotion_routes.routes._CancelCloser(ws)
        closer.start()
        closer.start()
        await closer.wait()
        self.assertEqual(ws.sent, [{"type": "cancelled"}, 4410])


if __name__ == "__main__":
    unittest.main()
