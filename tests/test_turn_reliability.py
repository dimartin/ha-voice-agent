"""Every turn ends with something speakable; action calls need a command; the
working phrase; the cache warm-up. Driven through the real app against a fake LLM."""
import asyncio
import json

from aiohttp import ClientSession, web

from ha_voice_agent import app, guard, warm

CFG = guard.load()


def sse(*chunks):
    return b"".join(b"data: " + json.dumps(c).encode() + b"\n\n" for c in chunks) + b"data: [DONE]\n\n"


def content(text, finish=None):
    return {"id": "x", "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": finish}]}


def tool(name, args="{}", finish=None):
    return {"id": "x", "choices": [{"index": 0, "finish_reason": finish, "delta": {"tool_calls": [
        {"index": 0, "id": "c1", "type": "function", "function": {"name": name, "arguments": args}}]}}]}


def end(reason):
    return {"id": "x", "choices": [{"index": 0, "delta": {}, "finish_reason": reason}]}


def run(handler, text, stream=True, timeout=None, after_tool=False, history=(), tool_name="x", tool_result="{}", req_tools=None):
    """POST one user turn through the proxy; handler(request) is the fake upstream."""
    async def go():
        up = web.Application()
        up.router.add_post("/v1/chat/completions", handler)
        up_runner = web.AppRunner(up)
        await up_runner.setup()
        await web.TCPSite(up_runner, "127.0.0.1", 0).start()
        app.UPSTREAM = f"http://127.0.0.1:{up_runner.addresses[0][1]}"
        if timeout:
            app.UPSTREAM_TIMEOUT = timeout
        vf = app.make_app()
        vf_runner = web.AppRunner(vf)
        await vf_runner.setup()
        await web.TCPSite(vf_runner, "127.0.0.1", 0).start()
        url = f"http://127.0.0.1:{vf_runner.addresses[0][1]}/v1/chat/completions"
        msgs = list(history) + [{"role": "user", "content": text}]
        if after_tool:
            msgs += [{"role": "assistant", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": tool_name, "arguments": "{}"}}]},
                     {"role": "tool", "tool_call_id": "c1", "content": tool_result}]
        try:
            async with ClientSession() as ha:
                async with ha.post(url, json={"stream": stream, "messages": msgs, **({"tools": req_tools} if req_tools else {})}) as r:
                    raw = await r.read()
        finally:
            await vf_runner.cleanup()
            await up_runner.cleanup()
            app.UPSTREAM_TIMEOUT = 25.0
        if not stream:
            msg = json.loads(raw)["choices"][0]["message"]
            return msg.get("content") or "", msg.get("tool_calls") or [], []
        events = [json.loads(l[5:]) for l in raw.decode().splitlines() if l.startswith("data:") and "[DONE]" not in l]
        said = "".join(e["choices"][0]["delta"].get("content") or "" for e in events if e.get("choices"))
        tools = [e for e in events if e.get("choices") and e["choices"][0]["delta"].get("tool_calls")]
        order = ["content" if e["choices"][0]["delta"].get("content") else "tool" if e["choices"][0]["delta"].get("tool_calls")
                 else "finish:" + str(e["choices"][0]["finish_reason"]) for e in events if e.get("choices")]
        return said, tools, order

    return asyncio.run(go())


def streaming(*chunks):
    async def handler(request):
        r = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await r.prepare(request)
        await r.write(sse(*chunks))
        return r
    return handler


def status(code, text="boom"):
    async def handler(request):
        return web.Response(status=code, text=text)
    return handler


# --- issue 13: every turn ends with something speakable ---

def test_upstream_error_status_says_the_fallback():
    for stream in (True, False):
        said, _, _ = run(status(500), "what is the capital of France", stream=stream)
        assert said.strip() == app.FALLBACK_REPLY, stream


def test_upstream_timeout_says_the_fallback():
    async def slow(request):
        await asyncio.sleep(3)
        return web.json_response({})
    for stream in (True, False):
        said, _, _ = run(slow, "what is the capital of France", stream=stream, timeout=0.5)
        assert said.strip() == app.FALLBACK_REPLY, stream


def test_stream_that_stalls_after_starting_says_the_fallback():
    async def stall(request):
        r = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await r.prepare(request)
        await r.write(b"data: " + json.dumps({"id": "x", "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}).encode() + b"\n\n")
        await asyncio.sleep(3)
        return r
    said, _, order = run(stall, "what is the capital of France", timeout=0.5)
    assert said.strip() == app.FALLBACK_REPLY
    assert order[-1] == "finish:stop"


def test_empty_reply_says_the_fallback():
    said, _, order = run(streaming(end("stop")), "what is the capital of France")
    assert said.strip() == app.FALLBACK_REPLY and order == ["content", "finish:stop"]
    async def empty(request):
        return web.json_response({"choices": [{"message": {"role": "assistant", "content": ""}, "finish_reason": "stop"}]})
    said, _, _ = run(empty, "what is the capital of France", stream=False)
    assert said == app.FALLBACK_REPLY


def test_length_finish_ends_normally():
    said, _, order = run(streaming(content("Paris is the capital. It is"), end("length")), "what is the capital of France")
    assert said.strip() == "Paris is the capital. It is" and order[-1] == "finish:stop"


# --- issue 3: action calls need a command ---

def test_guard_words():
    for text in ["turn off the kitchen light", "set the heat to 70", "make it warmer", "kitchen lights off",
                 "could you play some jazz", "I want to hear Steely Dan", "dim the lamp",
                 # verbless commands (review of PR 18, round 1)
                 "ten minute timer", "timer for 5 minutes", "volume 30", "volume to 30 percent",
                 "kitchen lights to 50 percent", "thermostat to 72", "alarm for 7 am", "kitchen lights red",
                 # round 2
                 "vacuum the dining room", "mow the lawn", "send the vacuum home", "continue the music"]:
        assert guard.has_command(text, CFG), text
    for text in ["warmup cold prompt", "is the kitchen light on", "yeah", "the cabinet", "I'm cold", "I'm warm", "",
                 "how long is left on the timer", "what is the volume", "is it 72 in here"]:
        assert not guard.has_command(text, CFG), text
    assert guard.is_action("light__HassLightSet", CFG) and guard.is_action("intent__HassTurnOn", CFG)
    assert not guard.is_action("homeassistant__GetLiveContext", CFG)


def test_action_call_without_a_command_is_refused():
    said, tools, order = run(streaming(tool("climate__HassClimateSetTemperature", '{"temperature": 75}'), end("tool_calls")),
                             "warmup cold prompt")
    assert not tools and said.strip() == app.BLOCK_REPLY and order[-1] == "finish:stop"
    async def nonstream(request):
        return web.json_response({"choices": [{"finish_reason": "tool_calls", "message": {"role": "assistant", "content": None,
                                  "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "intent__HassTurnOn", "arguments": "{}"}}]}}]})
    said, tools, _ = run(nonstream, "the cabinet", stream=False)
    assert not tools and said == app.BLOCK_REPLY


def test_action_call_with_a_command_and_reads_pass():
    said, tools, _ = run(streaming(tool("intent__HassTurnOn"), end("tool_calls")), "turn on the cabinet light")
    assert len(tools) == 1
    said, tools, _ = run(streaming(tool("homeassistant__GetLiveContext"), end("tool_calls")), "is the cabinet light on")
    assert len(tools) == 1


# --- issue 12: pending text, then the working phrase, before the tool runs ---

def test_pending_text_is_said_before_the_tool_call():
    said, tools, order = run(streaming(content("Let me check the thermostat"), tool("homeassistant__GetLiveContext"), end("tool_calls")),
                             "how warm is it in here")
    assert said.strip() == "Let me check the thermostat" and len(tools) == 1
    assert order == ["content", "tool", "finish:tool_calls"]


def test_working_phrase_only_when_nothing_was_said_and_not_after_a_tool():
    said, _, order = run(streaming(tool("homeassistant__GetLiveContext"), end("tool_calls")), "how warm is it in here")
    assert said.strip() == app.WORKING_PHRASE and order == ["content", "tool", "finish:tool_calls"]
    said, _, _ = run(streaming(tool("homeassistant__GetLiveContext"), end("tool_calls")), "how warm is it in here", after_tool=True)
    assert said == ""


def test_text_before_a_finishing_tool_chunk_comes_first():
    said, _, order = run(streaming(content("Turning on the light"), tool("intent__HassTurnOn", finish="tool_calls")),
                         "turn on the light")
    assert said.strip() == "Turning on the light"
    assert order == ["content", "tool"]  # the tool chunk carries the only finish


# --- issue 5: warm the prompt cache ---

def test_warmer_saves_the_prefix_and_warms_once_per_change(tmp_path):
    async def go():
        seen = []

        async def llm(request):
            seen.append(await request.json())
            return web.json_response({"choices": [{"message": {"content": "h"}}], "timings": {"prompt_n": 3, "cache_n": 0}})

        async def health(request):
            return web.Response(text="ok")

        up = web.Application()
        up.router.add_post("/v1/chat/completions", llm)
        up.router.add_get("/health", health)
        runner = web.AppRunner(up)
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", 0).start()
        base = f"http://127.0.0.1:{runner.addresses[0][1]}"
        body = {"model": "m", "tools": [{"type": "function", "function": {"name": "t"}}],
                "messages": [{"role": "system", "content": "rules"}, {"role": "user", "content": "q1"}]}
        async with ClientSession() as s:
            w = warm.Warmer(s, base, str(tmp_path / "warm.json"))
            w.note(body)
            await w.task
            w.note(dict(body, messages=body["messages"][:1] + [{"role": "user", "content": "q2"}]))  # same prefix
            assert w.task.done()
            # A restart reads the saved prefix and warms when the server answers its health check.
            w2 = warm.Warmer(s, base, str(tmp_path / "warm.json"))
            watch = asyncio.get_running_loop().create_task(w2.watch())
            for _ in range(50):
                if len(seen) == 2:
                    break
                await asyncio.sleep(0.02)
            watch.cancel()
            await w2.task
        await runner.cleanup()
        return seen

    seen = asyncio.run(go())
    assert len(seen) == 2
    for req in seen:
        assert req["max_tokens"] == 1 and not req["stream"]
        assert req["messages"] == [{"role": "system", "content": "rules"}, {"role": "user", "content": "hi"}]
        assert req["tools"] == [{"type": "function", "function": {"name": "t"}}]


def test_answer_to_the_assistants_question_keeps_the_command():
    history = [{"role": "user", "content": "turn on the light"}, {"role": "assistant", "content": "Which room?"}]
    _, tools, _ = run(streaming(tool("intent__HassTurnOn"), end("tool_calls")), "The kitchen.", history=history)
    assert len(tools) == 1
    # A follow-up that extends the last request counts too.
    history = [{"role": "user", "content": "turn on the kitchen light"}, {"role": "assistant", "content": "Done."}]
    for text in ["and the bedroom", "the bedroom too"]:
        _, tools, _ = run(streaming(tool("intent__HassTurnOn"), end("tool_calls")), text, history=history)
        assert len(tools) == 1, text
    # Not when the assistant made a statement and the user said something unrelated.
    _, tools, _ = run(streaming(tool("intent__HassTurnOn"), end("tool_calls")), "the cabinet", history=history)
    assert not tools


def test_refused_call_after_streamed_text_has_one_space():
    # A finished sentence has already been streamed, so the refusal follows it.
    said, _, _ = run(streaming(content("Turning on the cabinet light. "), tool("intent__HassTurnOn"), end("tool_calls")), "the cabinet")
    assert said == "Turning on the cabinet light. " + app.BLOCK_REPLY
    # Text still held in the cutter is dropped: the false claim is never spoken.
    said, _, _ = run(streaming(content("Turning on the cabinet light."), tool("intent__HassTurnOn"), end("tool_calls")), "the cabinet")
    assert said == app.BLOCK_REPLY


def test_prefix_change_during_a_warm_up_is_warmed_after_it(tmp_path):
    async def go():
        seen, gate = [], asyncio.Event()

        async def llm(request):
            seen.append((await request.json())["messages"][0]["content"])
            if len(seen) == 1:
                await gate.wait()
            return web.json_response({"choices": [{"message": {"content": "h"}}]})

        up = web.Application()
        up.router.add_post("/v1/chat/completions", llm)
        runner = web.AppRunner(up)
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", 0).start()
        base = f"http://127.0.0.1:{runner.addresses[0][1]}"
        body = lambda s: {"model": "m", "messages": [{"role": "system", "content": s}, {"role": "user", "content": "q"}]}
        async with ClientSession() as s:
            w = warm.Warmer(s, base, str(tmp_path / "warm.json"))
            w.note(body("OLD"))
            while not seen:
                await asyncio.sleep(0.01)
            w.note(body("NEW"))  # arrives while OLD is still warming
            gate.set()
            await w.task
        await runner.cleanup()
        return seen

    assert asyncio.run(go()) == ["OLD", "NEW"]


def test_refusal_does_not_reopen_the_mic():
    # Home Assistant keeps the mic open after a reply ending in "?" (spoken.py).
    assert not app.BLOCK_REPLY.rstrip().endswith("?")


def test_guard_reads_home_assistants_messages_not_the_rewrite(monkeypatch):
    # In agent mode the area line ("... generic commands like 'turn on the lights' ...")
    # is moved into the user turn. The guard must judge what the user said, not that.
    import json as _json
    from pathlib import Path as _P
    root = _P(__file__).parent
    system = _json.loads((root / "fixtures" / "ha_request.json").read_text())["messages"][0]
    monkeypatch.setattr(app, "AGENT_MODE", True)
    monkeypatch.setattr(app, "AGENT_PROMPT_FILE", str(root.parent / "config" / "agent-prompt.txt"))
    said, tools, _ = run(streaming(tool("intent__HassTurnOn"), end("tool_calls")), "the cabinet", history=[system])
    assert not tools and said.strip() == app.BLOCK_REPLY
