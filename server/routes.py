###############################################################################
#  服务器路由 — 统一异常处理的 API 路由
###############################################################################

import json
import asyncio
import threading
from aiohttp import web

from avatars.emotion import resolve_label
from server import reply_stream
from server.reply_stream import ReplyFeeder, reply_state_for, valid_reply_id
from utils.logger import logger


# ─── 路由工具函数 ──────────────────────────────────────────────────────────

def json_ok(data=None):
    """返回成功 JSON 响应"""
    body = {"code": 0, "msg": "ok"}
    if data is not None:
        body["data"] = data
    return web.Response(
        content_type="application/json",
        text=json.dumps(body),
    )


def json_error(msg: str, code: int = -1):
    """返回错误 JSON 响应"""
    return web.Response(
        content_type="application/json",
        text=json.dumps({"code": code, "msg": str(msg)}),
    )


from server.session_manager import session_manager
from server.avatar_routes import setup_avatar_routes
from server.admin_routes import setup_admin_routes

def get_session(request, sessionid: str):
    """从 app 中获取 session 实例"""
    return session_manager.get_session(sessionid)


# 音频上传解码/重采样是 CPU 密集操作，放到默认线程池执行，避免阻塞事件循环
# （WebRTC 信令与所有会话共用该 loop）；同一会话串行、跨会话并行
_humanaudio_lock_guard = threading.Lock()


def _get_humanaudio_lock(avatar_session):
    """同一会话的上传需要串行（分块入队不能交错），不同会话之间可以并行"""
    lock = getattr(avatar_session, '_humanaudio_lock', None)
    if lock is None:
        with _humanaudio_lock_guard:
            lock = getattr(avatar_session, '_humanaudio_lock', None)
            if lock is None:
                lock = threading.Lock()
                avatar_session._humanaudio_lock = lock
    return lock


def _put_audio_file_serialized(avatar_session, filebytes, datainfo):
    with _get_humanaudio_lock(avatar_session):
        avatar_session.put_audio_file(filebytes, datainfo)


def _interrupt(avatar_session):
    """Stop talking: cancel any streamed reply first, so none of its frames follow the flush."""
    cancelled = reply_state_for(avatar_session).cancel()
    if cancelled is not None:
        logger.info('stream_audio reply %s cancelled by interrupt', cancelled)
    avatar_session.flush_talk()


def _stream_refusal(status: int, msg: str):
    return web.json_response({"code": -1, "msg": msg}, status=status)


class _CancelCloser:
    """Sends "cancelled" and closes with 4410 exactly once, from whichever side gets there first."""

    def __init__(self, ws):
        self.ws = ws
        self.requested = False
        self._task = None

    def start(self):
        if self._task is None:
            self._task = asyncio.ensure_future(self._close())

    async def wait(self):
        self.start()
        await self._task

    async def _close(self):
        if not self.ws.prepared or self.ws.closed:
            return
        try:
            await self.ws.send_json({"type": "cancelled"})
            await self.ws.close(code=4410, message=b"cancelled")
        except (ConnectionResetError, RuntimeError):
            pass  # client already gone; nothing left to tell it


async def stream_audio(request):
    """WebSocket: one streamed reply of 16 kHz mono s16le PCM, ended by {"type":"end"}."""
    sessionid = request.query.get('sessionid', '')
    reply_id = request.query.get('reply_id')
    avatar_session = get_session(request, sessionid)
    if avatar_session is None:
        return _stream_refusal(404, "session not found")
    if not valid_reply_id(reply_id):
        return _stream_refusal(400, "reply_id must be 1-64 chars of A-Z a-z 0-9 _ . : -")

    ws = web.WebSocketResponse()
    if not ws.can_prepare(request).ok:
        return _stream_refusal(400, "WebSocket upgrade required")

    state = reply_state_for(avatar_session)
    loop = asyncio.get_running_loop()
    closer = _CancelCloser(ws)

    def on_cancel():
        # Closing the socket is what wakes the receive loop below. cancel() is plain sync
        # code that may run off the event loop, hence call_soon_threadsafe.
        closer.requested = True
        loop.call_soon_threadsafe(closer.start)

    opened = state.try_open(reply_id, on_cancel=on_cancel)
    if opened == "cancelled":
        return _stream_refusal(410, "reply cancelled")
    if opened == "busy":
        return _stream_refusal(409, "another reply is open on this session")

    feeder = None
    ended = False
    try:
        await ws.prepare(request)
        datainfo = {'reply_id': reply_id}
        if avatar_session.emotion is not None:
            # after prepare, so a failed upgrade does not use up the pending emotion
            datainfo['emotion'] = avatar_session.emotion.emotion_for_upload(
                request.query.get('emotion'))
        feeder = ReplyFeeder(avatar_session.put_audio_frame, datainfo,
                             allowed=lambda: state.is_feeding(reply_id))
        await ws.send_json({"type": "ready", "reply_id": reply_id})
        logger.info('stream_audio reply %s open on session %s', reply_id, sessionid)

        while state.is_feeding(reply_id):
            try:
                msg = await ws.receive(timeout=reply_stream.STALL_TIMEOUT_S)
            except asyncio.TimeoutError:
                logger.warning('stream_audio reply %s stalled, ending it', reply_id)
                break
            if msg.type == web.WSMsgType.BINARY:
                feeder.feed(msg.data)
            elif msg.type == web.WSMsgType.TEXT:
                if _is_end_message(msg.data):
                    ended = True
                    break
                logger.warning('stream_audio reply %s: ignored text message', reply_id)
            else:
                break  # closed or errored without an end marker
    except Exception:
        logger.exception('stream_audio reply %s failed', reply_id)
    finally:
        try:
            # Stall, close or error without an end marker: play what was received.
            if feeder is not None and state.is_feeding(reply_id):
                feeder.end()
        except Exception:
            logger.exception('stream_audio reply %s: feeding the rest failed', reply_id)
        finally:
            # Always release the reply, or the session would refuse new ones with 409.
            state.finish_feeding(reply_id, feeder.fed_frames if feeder is not None else 0)
        if closer.requested:
            await closer.wait()
        else:
            if not ended:
                logger.info('stream_audio reply %s ended without end marker', reply_id)
            if ws.prepared and not ws.closed:
                await ws.close(code=1000)
    if not ws.prepared:
        return _stream_refusal(500, "stream setup failed")
    return ws


def _is_end_message(text: str) -> bool:
    try:
        message = json.loads(text)
    except ValueError:
        return False
    return isinstance(message, dict) and message.get('type') == 'end'


# ─── 路由处理函数 ──────────────────────────────────────────────────────────

async def human(request):
    """文本输入（echo/chat 模式），支持 voice/emotion 参数"""
    try:
        params: dict = await request.json()

        sessionid: str = params.get('sessionid', '')
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")

        if params.get('interrupt'):
            _interrupt(avatar_session)

        datainfo = {}
        if params.get('tts'):  # tts 参数透传（voice, emotion 等）
            datainfo['tts'] = params.get('tts')

        if params['type'] in ('echo', 'chat') and avatar_session.emotion is not None:
            # after the type check, so a malformed request does not consume the pending emotion
            emotion = params.get('emotion')
            datainfo['emotion'] = avatar_session.emotion.emotion_for_upload(
                str(emotion) if emotion is not None else None)

        if params['type'] == 'echo':
            avatar_session.put_msg_txt(params['text'], datainfo)
        elif params['type'] == 'chat':
            llm_response = request.app.get("llm_response")
            if llm_response:
                asyncio.get_event_loop().run_in_executor(
                    None, llm_response, params['text'], avatar_session, datainfo
                )

        return json_ok()
    except Exception as e:
        logger.exception('human route exception:')
        return json_error(str(e))


async def interrupt_talk(request):
    """打断当前说话"""
    try:
        params = await request.json()
        sessionid = params.get('sessionid', '')
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")
        _interrupt(avatar_session)
        return json_ok()
    except Exception as e:
        logger.exception('interrupt_talk exception:')
        return json_error(str(e))


async def humanaudio(request):
    """上传音频文件"""
    try:
        form = await request.post()
        sessionid = str(form.get('sessionid', ''))
        fileobj = form["file"]
        filebytes = fileobj.file.read()

        datainfo = {}

        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")
        if avatar_session.emotion is not None:
            emotion = form.get('emotion')
            datainfo['emotion'] = avatar_session.emotion.emotion_for_upload(
                str(emotion) if emotion is not None else None)
        # 解码/重采样耗时与音频时长相关，放到默认线程池执行，避免阻塞事件循环；
        # 同一会话串行、跨会话并行
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _put_audio_file_serialized, avatar_session, filebytes, datainfo)
        return json_ok()
    except Exception as e:
        logger.exception('humanaudio exception:')
        return json_error(str(e))


async def set_emotion(request):
    """Set the pending emotion for one session or every session of an avatar"""
    try:
        try:
            params = await request.json()
        except ValueError:
            return json_error("invalid json")
        if not isinstance(params, dict):
            return json_error("invalid json")
        sessionid = params.get('sessionid')
        avatar_id = params.get('avatar_id')
        if not sessionid and not avatar_id:
            return json_error("sessionid or avatar_id required")
        if sessionid:
            session = get_session(request, str(sessionid))
            if session is None:
                return json_error("session not found")
            targets = [session]
        else:
            targets = [s for s in list(session_manager.sessions.values())
                       if s is not None
                       and getattr(getattr(s, 'opt', None), 'avatar_id', None) == avatar_id]
        emotion = params.get('emotion')
        resolved = resolve_label(str(emotion) if emotion is not None else None,
                                 request.app['opt'].emotion_labels_parsed)
        count = 0
        for session in targets:
            if session.emotion is not None:
                session.emotion.set_pending(resolved)
                count += 1
        return json_ok({"emotion": resolved, "sessions": count})
    except Exception as e:
        logger.exception('set_emotion exception:')
        return json_error(str(e))


async def set_audiotype(request):
    """设置自定义状态（动作编排）"""
    try:
        params = await request.json()
        sessionid = params.get('sessionid', '')
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")
        avatar_session.set_custom_state(params['audiotype'])
        return json_ok()
    except Exception as e:
        logger.exception('set_audiotype exception:')
        return json_error(str(e))


async def record(request):
    """录制控制"""
    try:
        params = await request.json()
        sessionid = params.get('sessionid', '')
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")
        if params['type'] == 'start_record':
            avatar_session.start_recording()
        elif params['type'] == 'end_record':
            avatar_session.stop_recording()
        return json_ok()
    except Exception as e:
        logger.exception('record exception:')
        return json_error(str(e))


async def is_speaking(request):
    """查询是否正在说话"""
    params = await request.json()
    sessionid = params.get('sessionid', '')
    avatar_session = get_session(request, sessionid)
    if avatar_session is None:
        return json_error("session not found")
    # reply_id sits beside data so clients reading data as a bool keep working
    return web.json_response({"code": 0, "msg": "ok",
                              "data": avatar_session.is_speaking(),
                              "reply_id": reply_state_for(avatar_session).current()})

async def sse_handler(request):
    """SSE 事件流，推送服务器状态更新到客户端"""
    sessionid = request.query.get('sessionid', '')
    avatar_session = session_manager.get_session(sessionid)
    if avatar_session is None:
        return json_error("session not found")

    response = web.StreamResponse(
        status=200,
        reason='OK',
        headers={
            'Content-Type': 'text/event-stream',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Access-Control-Allow-Origin': '*',
        }
    )
    await response.prepare(request)

    import queue
    msgqueue = queue.Queue()
    avatar_session.add_msgqueue(msgqueue)

    try:
        while True:
            try:
                msg = msgqueue.get_nowait()
                await response.write(f"data: {msg}\n\n".encode('utf-8'))
            except queue.Empty:
                await asyncio.sleep(0.01)
    except (asyncio.CancelledError, ConnectionResetError):
        logger.info('SSE connection closed for session: %s', sessionid)
    finally:
        if msgqueue in avatar_session.msgqueues:
            avatar_session.msgqueues.remove(msgqueue)

    return response


async def admin_config(request):
    """Admin: 获取全局配置参数"""
    try:
        opt = request.app.get("opt")
        if opt:
            return json_ok(data={"config": vars(opt)})
        return json_error("Config not found")
    except Exception as e:
        logger.exception('admin_config exception:')
        return json_error(str(e))


async def admin_sessions(request):
    """Admin: 获取活跃的会话及其配置"""
    try:
        sessions_info = []
        for sid, avatar_session in session_manager.sessions.items():
            if avatar_session:
                s_opt = getattr(avatar_session, 'opt', None)
                s_data = {
                    "sessionid": sid,
                    "speaking": avatar_session.is_speaking() if hasattr(avatar_session, 'is_speaking') else False,
                    "recording": getattr(avatar_session, 'recording', False),
                }
                controller = getattr(avatar_session, 'emotion', None)
                s_data["emotion_active"] = controller.active_name if controller else None
                s_data["emotion_pending"] = controller.pending_name if controller else None
                if s_opt:
                    s_data.update({
                        "model": getattr(s_opt, "model", ""),
                        "avatar_id": getattr(s_opt, "avatar_id", ""),
                        "REF_FILE": getattr(s_opt, "REF_FILE", ""),
                        "transport": getattr(s_opt, "transport", ""),
                        "batch_size": getattr(s_opt, "batch_size", 0),
                        "customopt": getattr(s_opt, "customopt", []),
                    })
                sessions_info.append(s_data)
        return json_ok(data={"sessions": sessions_info})
    except Exception as e:
        logger.exception('admin_sessions exception:')
        return json_error(str(e))


# ─── 路由注册 ──────────────────────────────────────────────────────────────

async def index(request):
    """默认首页重定向"""
    opt = request.app.get("opt")
    pagename = 'index.html'
    if opt and opt.transport == 'rtmp':
        pagename = 'rtmpapi.html'
    elif opt and opt.transport == 'rtcpush':
        pagename = 'rtcpushapi.html'
    raise web.HTTPFound(f'/{pagename}')


def setup_routes(app):
    """注册所有路由到 aiohttp app"""
    app.router.add_get("/", index)
    app.router.add_post("/human", human)
    app.router.add_post("/humanaudio", humanaudio)
    app.router.add_get("/stream_audio", stream_audio)
    app.router.add_post("/set_emotion", set_emotion)
    app.router.add_post("/set_audiotype", set_audiotype)
    app.router.add_post("/record", record)
    app.router.add_post("/interrupt_talk", interrupt_talk)
    app.router.add_post("/is_speaking", is_speaking)
    app.router.add_get("/api/admin/config", admin_config)
    app.router.add_get("/api/admin/sessions", admin_sessions)
    app.router.add_get('/sse', sse_handler)

    # ── Local ASR endpoint (SenseVoice/FunASR) ── Issue #604 ──
    try:
        from server.asr_server import asr_websocket_handler, is_funasr_available
        if is_funasr_available():
            app.router.add_get("/api/asr", asr_websocket_handler)
            logger.info("[ASR] Local SenseVoice ASR endpoint enabled at /api/asr")
        else:
            logger.info("[ASR] funasr not installed — local ASR endpoint disabled "
                        "(pip install funasr modelscope)")
    except Exception as e:
        logger.warning(f"[ASR] Failed to register ASR endpoint: {e}")

    # 注册 avatar 生成相关的路由
    setup_avatar_routes(app)

    # 注册形象管理（/api/admin/avatars）路由
    setup_admin_routes(app)

    app.router.add_static('/', path='web')
