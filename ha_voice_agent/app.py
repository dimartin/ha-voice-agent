"""OpenAI-compatible proxy between Home Assistant and a local LLM server that makes replies speakable.

Content is rewritten for TTS (markdown, emoji, symbols) and cut at the first sentence
boundary past MAX_WORDS. Tool calls pass through untouched. Every chat request is logged
to VOICE_LOG as one JSON line, so repeated and failed questions can be counted later.
"""
import asyncio
import contextvars
import json
import logging
import os
import time

from aiohttp import ClientError, ClientSession, ClientTimeout, web

from . import agent, claims, guard, turns, units, warm
from . import names as heard_names
from .spoken import SentenceCutter, wants_cancel

UPSTREAM = os.environ.get("UPSTREAM", "").rstrip("/")  # e.g. http://llm-host:8000, required
MAX_WORDS = int(os.environ.get("MAX_WORDS", "35"))
VOICE_LOG = os.environ.get("VOICE_LOG", "/data/voice-log.jsonl")
# Agent mode replaces HA's system prompt and compacts its tools (see agent.py).
AGENT_MODE = os.environ.get("AGENT_MODE", "0") == "1"
AGENT_PROMPT_FILE = os.environ.get("AGENT_PROMPT_FILE", "/app/config/agent-prompt.txt")


def _csv(name):
    return frozenset(x.strip() for x in os.environ.get(name, "").split(",") if x.strip())


DROP_PARAMS, DROP_TOOLS = _csv("DROP_PARAMS"), _csv("DROP_TOOLS")
log = logging.getLogger("voice-filter")
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
HOP = {"host", "content-length", "transfer-encoding", "connection", "accept-encoding"}


def last_user(body):
    for m in reversed(body.get("messages", [])):
        if m.get("role") == "user":
            c = m.get("content")
            return c if isinstance(c, str) else " ".join(p.get("text", "") for p in c or [] if isinstance(p, dict))
    return None


# Per-request fields every record() call of the turn carries (aiohttp runs each request
# in its own task, so this is per request).
TURN_FIELDS: contextvars.ContextVar[dict] = contextvars.ContextVar("turn_fields", default={})


def record(**entry):
    entry = {**TURN_FIELDS.get(), **entry}
    entry = {k: v for k, v in entry.items() if v is not None}
    entry["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    try:
        with open(VOICE_LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError as e:
        log.warning("voice log write failed: %s", e)


def chunk_like(template, delta, finish=None):
    return {"id": template.get("id"), "object": "chat.completion.chunk", "created": template.get("created"),
            "model": template.get("model"), "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


CANCEL_REPLY = os.environ.get("CANCEL_REPLY", "Okay.")
# Spoken instead of a reply in which the model narrates its reasoning (spoken.NARRATION).
NARRATION_REPLY = os.environ.get("NARRATION_REPLY", "Sorry, I didn't catch that.")
# Spoken when the upstream fails, times out or returns nothing to say.
FALLBACK_REPLY = os.environ.get("FALLBACK_REPLY", "Sorry, I can't answer right now.")
# Spoken instead of an action tool call when the request contained no command (guard.py).
# It ends in "." on purpose: Home Assistant reopens the mic after a "?", and refusals
# mostly follow false wakes, when the room isn't talking to the assistant.
BLOCK_REPLY = os.environ.get("BLOCK_REPLY", "Sorry, I didn't catch that.")
# How many ListDevices rounds the proxy answers itself in one turn.
LOCAL_ROUNDS = int(os.environ.get("LOCAL_ROUNDS", "2"))
# Spoken instead of a reply that claims an action no tool in the turn carried out, when a
# re-run did not do it either (claims.py).
CLAIM_REPLY = os.environ.get("CLAIM_REPLY", "Sorry, I couldn't do that.")
# Said before a tool runs when nothing else has been said yet; empty turns it off.
WORKING_PHRASE = os.environ.get("WORKING_PHRASE", "One moment.")
# Seconds without data from the upstream before the turn gives up and says FALLBACK_REPLY.
# Home Assistant abandons a voice turn at about 30 s.
UPSTREAM_TIMEOUT = float(os.environ.get("UPSTREAM_TIMEOUT", "25"))


async def canned(request, body, text):
    """Answer with fixed text in whichever shape (streamed or not) the client asked for."""
    tpl = {"id": "voice-filter", "created": int(time.time()), "model": body.get("model")}
    if not body.get("stream"):
        return web.json_response({**tpl, "object": "chat.completion", "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]})
    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"})
    await resp.prepare(request)
    for chunk in (chunk_like(tpl, {"role": "assistant", "content": text}), chunk_like(tpl, {}, "stop")):
        await resp.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
    await resp.write(b"data: [DONE]\n\n")
    return resp


async def chat(request: web.Request):
    raw = await request.read()
    body = json.loads(raw)
    headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP}
    said_messages = list(body.get("messages") or [])  # as Home Assistant sent them, for the guard
    TURN_FIELDS.set(turns.fields(said_messages))
    system = next((m.get("content") for m in said_messages if m.get("role") == "system"), "")
    subs = heard_names.apply(body, agent.parse_static_context(system if isinstance(system, str) else ""))
    if subs:
        raw = json.dumps(body).encode()  # without agent mode the upstream gets these bytes
    user, t0 = last_user(body), time.time()
    # Requests that continue after a tool result have the same last user message; log the turn kind.
    after_tool = bool(body.get("messages")) and body["messages"][-1].get("role") == "tool"
    if subs and not after_tool:
        log.info("misheard name corrected: %s", subs)
        TURN_FIELDS.set({**TURN_FIELDS.get(), "heard": last_user({"messages": said_messages}), "corrected": subs})
    if not after_tool and wants_cancel(user):
        log.info("cancel phrase, turn ends without the LLM: %r", user)
        record(user=user, after_tool=False, raw="", spoken=CANCEL_REPLY, cut=False, tools=[], s=0)
        return await canned(request, body, CANCEL_REPLY)
    local = None if after_tool else units.answer(user)
    if local:
        log.info("weights and measures, answered without the LLM: %r", user)
        record(user=user, after_tool=False, raw="", spoken=local, cut=False, tools=[], s=0, local="units")
        return await canned(request, body, local)
    devices = []
    if AGENT_MODE:
        first = (body.get("messages") or [{}])[0]
        if first.get("role") == "system" and isinstance(first.get("content"), str):
            devices = agent.parse_static_context(first["content"])
        # Read per request so prompt edits apply without a restart; it is a small file.
        with open(AGENT_PROMPT_FILE) as f:
            body = agent.rewrite(body, f.read(), DROP_PARAMS, DROP_TOOLS, agent.load_tool_descriptions())
        raw = json.dumps(body).encode()
        log.debug("agent rewrite: %d bytes, %d tools", len(raw), len(body.get("tools") or []))
    if request.app.get("warmer"):
        request.app["warmer"].note(body)
    guard_cfg = guard.load()
    # A command with no action landed yet: a text reply is held until it is known not to
    # claim something nothing did (claims.py).
    check = claims.needs_check(said_messages, guard_cfg)
    sess: ClientSession = request.app["session"]
    # With a short device list the model may call LIST_TOOL; Home Assistant doesn't know it,
    # so the proxy answers it and asks the model again (agent.LAZY_DEVICES_ABOVE).
    local_tools = AGENT_MODE and agent.lazy(devices)
    listed = []  # areas the proxy looked up for the model this turn, for the voice log

    async def answer_listings(msg, base=None):
        """Answer LIST_TOOL calls until the model replies with something else (at most
        LOCAL_ROUNDS times). Returns the final message, and the areas looked up. `base` is
        the request the calls answer (the claim re-run's body, or by default this one).
        Only the listing calls go into the follow-up: a real call the model made beside a
        listing is dropped on purpose, and the model re-plans with the list in hand."""
        base = base or body
        looked = []
        for _ in range(LOCAL_ROUNDS):
            calls = msg.get("tool_calls") or []
            listing = [c for c in calls if (c.get("function") or {}).get("name") == agent.LIST_TOOL_NAME]
            if not listing:
                break
            results = []
            for c in listing:
                try:
                    area = json.loads((c.get("function") or {}).get("arguments") or "{}").get("area", "")
                except ValueError:
                    area = ""
                looked.append(area)
                results.append({"role": "tool", "tool_call_id": c.get("id"), "content": agent.list_devices(area, devices)})
            follow = {**base, "stream": False, "messages": (base.get("messages") or []) + [
                {"role": "assistant", "content": msg.get("content"), "tool_calls": listing}] + results}
            follow.pop("stream_options", None)
            async with sess.post(UPSTREAM + "/v1/chat/completions", json=follow, headers=headers,
                                 timeout=ClientTimeout(total=None, sock_connect=10, sock_read=UPSTREAM_TIMEOUT)) as r:
                if r.status != 200:
                    raise ClientError(f"status {r.status}")
                msg = (await r.json())["choices"][0]["message"]
        # A listing call left after the last round is never sent to Home Assistant.
        msg["tool_calls"] = [c for c in msg.get("tool_calls") or []
                             if (c.get("function") or {}).get("name") != agent.LIST_TOOL_NAME] or None
        return msg, looked

    async def settle_claim(claimed):
        """A reply claimed an action nothing did. Re-ask once with the action tools offered,
        unless the turn already tried (after a tool result) or the request is for later.
        Returns (text, tool calls, log fields)."""
        why = "after-tool" if after_tool else "later" if claims.is_later(user, guard_cfg) else None
        if why:
            log.warning("reply claims an action with nothing behind it (%s), saying CLAIM_REPLY: %r", why, claimed)
            return CLAIM_REPLY, [], {"claim_unbacked": True, "rerun": f"skipped-{why}"}
        rb = claims.rerun_body(body, guard_cfg, "act")
        log.warning("reply claims an action with nothing behind it, re-asking once: %r", claimed)
        try:
            async with sess.post(UPSTREAM + "/v1/chat/completions", json=rb, headers=headers,
                                 timeout=ClientTimeout(total=None, sock_connect=10, sock_read=UPSTREAM_TIMEOUT)) as r:
                if r.status != 200:
                    raise ClientError(f"status {r.status}")
                msg = (await r.json())["choices"][0]["message"]
            if local_tools and any((c.get("function") or {}).get("name") == agent.LIST_TOOL_NAME
                                   for c in msg.get("tool_calls") or []):
                # The re-run may look a device up first: answer that here too, never in
                # Home Assistant.
                msg, looked = await answer_listings(msg, rb)
                listed.extend(looked)
        except (ClientError, asyncio.TimeoutError, ValueError, KeyError, IndexError) as e:
            log.warning("claim re-run failed (%s), saying CLAIM_REPLY", type(e).__name__)
            return CLAIM_REPLY, [], {"claim_unbacked": True, "rerun": "act", "rerun_failed": type(e).__name__}
        calls = claims.allowed_calls(msg.get("tool_calls") or [], rb.get("tools") or [])
        fields = {"claim_unbacked": True, "rerun": "act",
                  "rerun_tools": [c["function"]["name"] for c in calls], "rerun_raw": msg.get("content") or ""}
        if calls:
            return "", calls, fields
        text = (msg.get("content") or "").strip()
        if not text or claims.claims_action(text, guard_cfg):
            return CLAIM_REPLY, [], fields
        cutter = SentenceCutter(MAX_WORDS, NARRATION_REPLY)
        return (cutter.feed(text) + cutter.flush()).strip() or CLAIM_REPLY, [], fields

    def fail(cause, raw="", **extra):
        # Every turn ends with something speakable; the real cause goes to the log only.
        log.warning("upstream failed (%s), speaking the fallback reply", cause)
        record(user=user, after_tool=after_tool, raw=raw, spoken=FALLBACK_REPLY, cut=False, tools=[], failed=cause,
               s=round(time.time() - t0, 2), listed=listed or None, **extra)

    timeout = ClientTimeout(total=None, sock_connect=10, sock_read=UPSTREAM_TIMEOUT)
    try:
        upstream = await sess.post(UPSTREAM + "/v1/chat/completions", data=raw, headers=headers, timeout=timeout)
    except (ClientError, asyncio.TimeoutError) as e:
        fail(type(e).__name__)
        return await canned(request, body, FALLBACK_REPLY)
    if upstream.status != 200:
        detail = (await upstream.read())[:200].decode(errors="replace")
        upstream.release()
        fail(f"status {upstream.status}", detail=detail)
        return await canned(request, body, FALLBACK_REPLY)

    if not body.get("stream"):
        try:
            data = await upstream.read()
        except (ClientError, asyncio.TimeoutError) as e:
            upstream.release()
            fail(type(e).__name__)
            return await canned(request, body, FALLBACK_REPLY)
        upstream.release()
        out = json.loads(data)
        msg = out["choices"][0]["message"]
        looked = None
        if local_tools and msg.get("tool_calls"):
            try:
                msg, looked = await answer_listings(msg)
            except (ClientError, asyncio.TimeoutError, ValueError, KeyError, IndexError) as e:
                fail(type(e).__name__)
                return await canned(request, body, FALLBACK_REPLY)
            out["choices"][0]["message"] = msg
            out["choices"][0]["finish_reason"] = "tool_calls" if msg.get("tool_calls") else "stop"
            listed += looked
        called = [c["function"]["name"] for c in msg.get("tool_calls") or []]
        refused = guard.blocked(said_messages, called, guard_cfg)
        timings = out.get("timings") or {}
        if refused:
            log.warning("blocked action call with no command in %r: %s", user, msg.get("tool_calls"))
            record(user=user, after_tool=after_tool, raw=msg.get("content") or "", spoken=BLOCK_REPLY, cut=False, tools=[],
                   blocked=refused, s=round(time.time() - t0, 2), listed=listed or None)
            msg.pop("tool_calls", None)
            msg["content"] = BLOCK_REPLY
            out["choices"][0]["finish_reason"] = "stop"
            return web.json_response(out)
        cutter = SentenceCutter(MAX_WORDS, None if after_tool else NARRATION_REPLY)
        orig = msg.get("content") or ""
        if check and not called and claims.claims_action(orig, guard_cfg):
            text, calls, fields = await settle_claim(orig)
            if calls:
                msg["tool_calls"], msg["content"] = calls, None
                out["choices"][0]["finish_reason"] = "tool_calls"
            else:
                msg["content"] = text
            record(user=user, after_tool=after_tool, raw=orig, spoken=text, cut=False, tools=fields.get("rerun_tools") or [], listed=listed or None,
                   calls=turns.calls(calls) or None, s=round(time.time() - t0, 2), **fields)
            return web.json_response(out)
        msg["content"] = (cutter.feed(orig) + cutter.flush() + cutter.narration_tail(called)).strip()
        if not msg["content"] and not called:
            fail("empty reply", raw=orig)
            msg["content"] = FALLBACK_REPLY
        else:
            cut = cutter.done and not cutter.narrated and len(orig.split()) > cutter.words
            record(user=user, after_tool=after_tool, raw=orig, spoken=msg["content"], cut=cut, narrated=cutter.narrated,
                   tools=called, calls=turns.calls(msg.get("tool_calls")) or None, prompt_n=timings.get("prompt_n"), cache_n=timings.get("cache_n"),
                   s=round(time.time() - t0, 2), listed=listed or None)
        return web.json_response(out)

    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"})
    await resp.prepare(request)
    cutter, raw_text, spoken, cut = SentenceCutter(MAX_WORDS, None if after_tool else NARRATION_REPLY), "", "", False
    calls, held = {}, []  # tool calls by index (name, arguments) and their chunks, held until complete
    last, finished, finish, failed, timings, refused = None, False, None, None, {}, []
    hold, claim = "", {}  # text held while `check` is on, and the claim check's log fields

    async def send(obj):
        await resp.write(b"data: " + json.dumps(obj).encode() + b"\n\n")

    async def say(text):
        nonlocal spoken
        if text:
            spoken += text
            await send(chunk_like(last, {"content": text}))

    async def speak(text):
        """Say text now, or hold it while the claim check is pending."""
        nonlocal hold
        if check:
            hold += text
        else:
            await say(text)

    async def end(reason):
        """Finish the turn: decide on held tool calls, then make sure something is said."""
        nonlocal finished, finish, refused, failed, check, hold, claim, listed, raw_text
        finished, finish = True, reason
        names = [c["name"] for c in calls.values() if c["name"]]
        if held and local_tools and agent.LIST_TOOL_NAME in names:
            # Answer the listing here, then send Home Assistant whatever the model does next.
            listing = {"content": raw_text or None, "tool_calls": [
                {"id": f"list{i}", "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
                for i, c in sorted(calls.items())]}
            try:
                msg, looked = await answer_listings(listing)
            except (ClientError, asyncio.TimeoutError, ValueError, KeyError, IndexError) as e:
                failed = type(e).__name__
                held.clear()
                calls.clear()
                await say(FALLBACK_REPLY)
                await send(chunk_like(last, {}, "stop"))
                return
            listed += looked
            held.clear()
            calls.clear()
            for i, c in enumerate(msg.get("tool_calls") or []):
                calls[i] = {"name": c["function"]["name"], "arguments": c["function"].get("arguments") or ""}
                held.append(chunk_like(last, {"tool_calls": [{"index": i, **c}]}))
            if not held:
                reason = finish = "stop"  # the listing call's finish no longer describes the turn
                text = (msg.get("content") or "").strip()
                out = (cutter.feed(" " + text) + cutter.flush()) if text else ""
                if check:
                    # The follow-up is this turn's reply: the claim check below decides it.
                    hold, raw_text = out, text
                else:
                    await say(out)
            else:
                names = [c["name"] for c in calls.values()]
        if check:
            # The turn is over or a tool call came: either way the held text is decided now.
            check, text, hold = False, hold, ""
            if not held and claims.claims_action(raw_text, guard_cfg):
                if failed:
                    # The upstream dropped after a claim: no re-run, but the claim is not spoken.
                    log.warning("reply claims an action with nothing behind it (upstream %s), saying CLAIM_REPLY: %r",
                                failed, raw_text)
                    text, rerun_calls, claim = CLAIM_REPLY, [], {"claim_unbacked": True, "rerun": "skipped-failed"}
                else:
                    # Stop the first generation before re-asking: on a one-slot server the
                    # re-run would otherwise queue behind it (the budget-cut path ends here
                    # while the first stream is still open).
                    upstream.close()
                    text, rerun_calls, claim = await settle_claim(raw_text)
                if rerun_calls:
                    if WORKING_PHRASE:
                        await say(WORKING_PHRASE + " ")
                    for i, c in enumerate(rerun_calls):
                        calls[i] = {"name": c["function"]["name"], "arguments": c["function"].get("arguments") or ""}
                        await send(chunk_like(last, {"tool_calls": [{"index": i, **c}]}))
                    await send(chunk_like(last, {}, "tool_calls"))
                    return
                await say(text)
                await send(chunk_like(last, {}, "stop"))
                return
            await say(text)
        if held:
            refused = guard.blocked(said_messages, names, guard_cfg)
            if refused:
                log.warning("blocked action call with no command in %r: %s", user, list(calls.values()))
                # Text already streamed can't be taken back; the refusal follows it.
                await say(BLOCK_REPLY if not spoken or spoken.endswith(" ") else " " + BLOCK_REPLY)
                await send(chunk_like(last, {}, "stop"))
                return
            # Say what is pending before the tool runs, or a short filler if nothing was said.
            pending = cutter.flush()
            if not pending and not spoken and not after_tool and WORKING_PHRASE:
                pending = WORKING_PHRASE + " "
            await say(pending)
            for c in held:
                await send(c)
            if not any(c["choices"][0].get("finish_reason") for c in held):
                await send(chunk_like(last, {}, reason or "tool_calls"))
            return
        await say(cutter.flush() + cutter.narration_tail(names))
        if not spoken.strip():
            failed = failed or "empty reply"
            await say(FALLBACK_REPLY)
        await send(chunk_like(last, {}, "stop" if reason in (None, "length") else reason))

    try:
        async for line in upstream.content:
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                break
            chunk = json.loads(payload)
            timings = chunk.get("timings") or timings
            if not chunk.get("choices"):
                await send(chunk)
                continue
            choice = chunk["choices"][0]
            last = chunk
            delta = choice.get("delta") or {}
            if delta.get("tool_calls"):
                for tc in delta["tool_calls"]:
                    c = calls.setdefault(tc.get("index", 0), {"name": "", "arguments": ""})
                    f = tc.get("function") or {}
                    c["name"] += f.get("name") or ""
                    c["arguments"] += f.get("arguments") or ""
                held.append(chunk)
                if choice.get("finish_reason"):
                    await end(choice["finish_reason"])
                    break
                continue
            text = delta.get("content") or ""
            raw_text += text
            out = cutter.feed(text) if text else ""
            await speak(out)
            if choice.get("finish_reason"):
                await end(choice["finish_reason"])
                break
            if cutter.done and not calls and not cutter.narrated:
                # Budget spent mid-answer: end the turn here and stop generation upstream.
                cut = True
                await end("stop")
                break
    except (ClientError, asyncio.TimeoutError) as e:
        failed = type(e).__name__
        log.warning("upstream stream failed (%s)", failed)
    try:
        if not finished:
            if last is None:
                last = {"id": "voice-filter", "created": int(time.time()), "model": body.get("model")}
            if failed:
                # A tool call cut off mid-stream is incomplete: never forward it.
                calls.clear()
                held.clear()
            await end(None)
            if failed and spoken.strip() == FALLBACK_REPLY:
                log.warning("spoke the fallback reply after %s", failed)
        await resp.write(b"data: [DONE]\n\n")
    finally:
        upstream.close()
        names = [c["name"] for c in calls.values() if c["name"]]
        record(user=user, after_tool=after_tool, raw=raw_text, spoken=spoken.strip(), cut=cut,
               tools=[] if refused else names, blocked=refused or None,
               calls=None if refused else (turns.calls(calls) or None), narrated=cutter.narrated,
               finish=finish, failed=failed, prompt_n=timings.get("prompt_n"), cache_n=timings.get("cache_n"),
               s=round(time.time() - t0, 2), listed=listed or None, **claim)
        log.info("chat words=%d cut=%s narrated=%s tools=%s blocked=%s failed=%s %.2fs", len(spoken.split()), cut,
                 cutter.narrated, names, refused, failed, time.time() - t0)
    return resp


async def passthrough(request: web.Request):
    sess: ClientSession = request.app["session"]
    headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP}
    async with sess.request(request.method, UPSTREAM + request.rel_url.path_qs, data=await request.read(), headers=headers) as r:
        return web.Response(body=await r.read(), status=r.status, content_type=r.content_type)


async def start(app):
    app["session"] = ClientSession(timeout=ClientTimeout(total=600))
    # Only agent mode makes the prefix the same for every satellite and request.
    if warm.WARM and AGENT_MODE and UPSTREAM:
        app["warmer"] = warm.Warmer(app["session"], UPSTREAM)
        app["warm_watch"] = asyncio.get_running_loop().create_task(app["warmer"].watch())


async def stop(app):
    if app.get("warm_watch"):
        app["warm_watch"].cancel()
    await app["session"].close()


# Cancel a request's handler when its client disconnects. aiohttp 3.9+ leaves the
# handler running, so a turn Home Assistant abandoned at its 30s limit kept
# waiting on a single-slot LLM for up to 600s and every request behind it timed
# out. On cancel the upstream connection closes, and llama-server cancels the task.
SERVER_KWARGS = {"handler_cancellation": True}


def make_app():
    app = web.Application(client_max_size=64 * 1024 * 1024)
    app.router.add_post("/v1/chat/completions", chat)
    app.router.add_route("*", "/{tail:.*}", passthrough)
    app.on_startup.append(start)
    app.on_cleanup.append(stop)
    return app


def main():
    if not UPSTREAM:
        raise SystemExit("UPSTREAM is required, e.g. UPSTREAM=http://llm-host:8000")
    log.info("voice-filter upstream=%s max_words=%d log=%s", UPSTREAM, MAX_WORDS, VOICE_LOG)
    web.run_app(make_app(), host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", "8011")), **SERVER_KWARGS)


if __name__ == "__main__":
    main()
