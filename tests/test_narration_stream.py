"""Narration followed by a tool call, streamed through the real app against a fake LLM."""
import asyncio
import json

from aiohttp import ClientSession, web

from ha_voice_agent import app


def sse(*chunks):
    return b"".join(b"data: " + json.dumps(c).encode() + b"\n\n" for c in chunks) + b"data: [DONE]\n\n"


def content(text):
    return {"id": "x", "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]}


TOOL = {"id": "x", "choices": [{"index": 0, "finish_reason": None, "delta": {"tool_calls": [
    {"index": 0, "id": "c1", "type": "function", "function": {"name": "HassTurnOn", "arguments": "{}"}}]}}]}
END = {"id": "x", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}
STOP = {"id": "x", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}

FINISHES = []  # finish_reasons of the last run


def run(stream, messages):
    async def go():
        async def llm(request):
            r = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await r.prepare(request)
            await r.write(stream)
            return r

        up = web.Application()
        up.router.add_post("/v1/chat/completions", llm)
        up_runner = web.AppRunner(up)
        await up_runner.setup()
        await web.TCPSite(up_runner, "127.0.0.1", 0).start()
        app.UPSTREAM = f"http://127.0.0.1:{up_runner.addresses[0][1]}"
        vf_runner = web.AppRunner(app.make_app())
        await vf_runner.setup()
        await web.TCPSite(vf_runner, "127.0.0.1", 0).start()
        url = f"http://127.0.0.1:{vf_runner.addresses[0][1]}/v1/chat/completions"
        try:
            async with ClientSession() as ha:
                async with ha.post(url, json={"stream": True, "messages": messages}) as r:
                    body = (await r.read()).decode()
        finally:
            await vf_runner.cleanup()
            await up_runner.cleanup()
        events = [json.loads(line[5:]) for line in body.splitlines() if line.startswith("data:") and "[DONE]" not in line]
        said = "".join(e["choices"][0]["delta"].get("content") or "" for e in events if e.get("choices"))
        tools = [e for e in events if e.get("choices") and e["choices"][0]["delta"].get("tool_calls")]
        finishes = [e["choices"][0]["finish_reason"] for e in events if e.get("choices") and e["choices"][0].get("finish_reason")]
        FINISHES[:] = finishes
        return said, tools

    return asyncio.run(go())


USER = [{"role": "user", "content": "turn on the kitchen light"}]


def test_narration_then_a_tool_call_forwards_the_tool_and_says_only_the_filler():
    said, tools = run(sse(content("The user wants the kitchen light on. "), content("I will turn it on."), TOOL, END), USER)
    assert len(tools) == 1
    assert said.strip() == "One moment."  # none of the narration; the working phrase while the tool runs


def test_narration_alone_says_the_fixed_reply():
    said, tools = run(sse(content("The user said \"Yeah.\" "), content("This is filler."), STOP), USER)
    assert not tools
    assert said.strip() == "Sorry, I didn't catch that."


def test_after_a_tool_result_narration_is_spoken_as_is():
    msgs = USER + [{"role": "assistant", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "HassTurnOn", "arguments": "{}"}}]},
                   {"role": "tool", "tool_call_id": "c1", "content": "{}"}]
    said, _ = run(sse(content("The user's request has been completed."), STOP), msgs)
    assert said.strip() == "The user's request has been completed."


def test_a_tool_call_chunk_that_finishes_the_stream_gets_one_finish():
    tool_end = {"id": "x", "choices": [{"index": 0, "finish_reason": "tool_calls", "delta": TOOL["choices"][0]["delta"]}]}
    said, tools = run(sse(tool_end), USER)
    assert len(tools) == 1 and said.strip() == "One moment."
    assert FINISHES == ["tool_calls"]


def test_a_stream_without_a_finish_still_says_its_last_sentence():
    said, _ = run(sse(content("The light is on")), USER)
    assert said.strip() == "The light is on"
    said, _ = run(sse(content("The user wants the light on")), USER)
    assert said.strip() == "Sorry, I didn't catch that."


def test_text_before_a_finishing_tool_chunk_is_spoken():
    tool_end = {"id": "x", "choices": [{"index": 0, "finish_reason": "tool_calls", "delta": TOOL["choices"][0]["delta"]}]}
    said, tools = run(sse(content("Turning on the light"), tool_end), USER)
    assert said.strip() == "Turning on the light" and len(tools) == 1
    assert FINISHES == ["tool_calls"]
