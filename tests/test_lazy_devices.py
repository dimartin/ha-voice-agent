"""Issue 6: above a device count the prompt lists areas, the satellite's own area rides in
the user's turn, and the proxy answers ListDevices itself."""
import json
from pathlib import Path

from aiohttp import web

from ha_voice_agent import agent, app
from test_turn_reliability import content, end, run

HERE = Path(__file__).parent
REQ = json.loads((HERE / "fixtures" / "ha_request.json").read_text())
RULES = (HERE.parent / "config" / "agent-prompt.txt").read_text()
SYSTEM = REQ["messages"][0]["content"]
DEVICES = agent.parse_static_context(SYSTEM)


def test_below_the_threshold_nothing_changes():
    assert agent.rewrite(REQ, RULES, above=0) == agent.rewrite(REQ, RULES)
    assert agent.rewrite(REQ, RULES, above=len(DEVICES)) == agent.rewrite(REQ, RULES)


def test_lazy_rewrite():
    new = agent.rewrite(REQ, RULES, above=5)
    system = new["messages"][0]["content"]
    assert "Areas in this home (number of devices): " in system and "Kitchen (4)" in system
    assert "Office Desk Lamp" not in system and "Devices in this home, by area" not in system
    user = [m for m in new["messages"] if m["role"] == "user"][-1]["content"]
    assert user.splitlines()[1] == "(Devices here: Kitchen: Kitchen Light [light]; Kitchen Counter [light]; " \
                                   "Coffee Maker [switch]; Fridge Temperature [sensor])"
    assert new["tools"][-1]["function"]["name"] == agent.LIST_TOOL_NAME


def test_lazy_prefix_is_the_same_for_every_satellite():
    other = SYSTEM.replace("You are in area Kitchen", "You are in area Office")
    a = agent.rewrite(REQ, RULES, above=5)
    b = agent.rewrite(dict(REQ, messages=[{"role": "system", "content": other}] + REQ["messages"][1:]), RULES, above=5)
    assert a["messages"][0] == b["messages"][0] and a["tools"] == b["tools"]
    ua = [m for m in a["messages"] if m["role"] == "user"][-1]["content"]
    ub = [m for m in b["messages"] if m["role"] == "user"][-1]["content"]
    assert "Kitchen Light" in ua and "Office Desk Lamp" in ub


def test_list_devices_answer():
    assert agent.list_devices("dining room", DEVICES).startswith("Dining Room: ")
    assert agent.list_devices("Attic", DEVICES).startswith("No area called 'Attic'. Areas: ")
    assert agent.satellite_area(SYSTEM) == "Kitchen"


def listing_then(final):
    """A fake upstream: the streamed first reply calls ListDevices(Office); the proxy's
    follow-up (not streamed) gets `final`."""
    seen = []

    async def handler(request):
        body = await request.json()
        seen.append(body)
        if len(seen) == 1:
            r = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await r.prepare(request)
            chunk = {"id": "x", "choices": [{"index": 0, "finish_reason": "tool_calls", "delta": {"tool_calls": [
                {"index": 0, "id": "c1", "type": "function",
                 "function": {"name": "ListDevices", "arguments": '{"area": "Office"}'}}]}}]}
            await r.write(b"data: " + json.dumps(chunk).encode() + b"\n\ndata: [DONE]\n\n")
            return r
        return web.json_response({"choices": [{"index": 0, "message": {"role": "assistant", **final}, "finish_reason": "stop"}]})
    return handler, seen


def lazy_run(handler, text, **kw):
    saved = app.AGENT_MODE, app.AGENT_PROMPT_FILE, agent.LAZY_DEVICES_ABOVE
    app.AGENT_MODE, app.AGENT_PROMPT_FILE = True, str(HERE.parent / "config" / "agent-prompt.txt")
    agent.LAZY_DEVICES_ABOVE = 5
    try:
        return run(handler, text, history=[REQ["messages"][0]], req_tools=REQ["tools"], **kw)
    finally:
        app.AGENT_MODE, app.AGENT_PROMPT_FILE, agent.LAZY_DEVICES_ABOVE = saved


def test_proxy_answers_list_devices_and_sends_the_next_call_on():
    h, seen = listing_then({"content": None, "tool_calls": [
        {"id": "t1", "type": "function", "function": {"name": "HassTurnOff", "arguments": '{"name": "Office Desk Lamp"}'}}]})
    said, tools, order = lazy_run(h, "turn off the office desk lamp")
    names = [c["function"]["name"] for t in tools for c in t["choices"][0]["delta"]["tool_calls"]]
    assert names == ["HassTurnOff"] and len(seen) == 2  # Home Assistant never sees ListDevices
    follow = seen[1]["messages"]
    assert follow[-2]["tool_calls"][0]["function"]["name"] == "ListDevices"
    assert follow[-1]["role"] == "tool" and follow[-1]["content"].startswith("Office: Office Desk Lamp")
    assert seen[1]["messages"][0] == seen[0]["messages"][0] and seen[1]["tools"] == seen[0]["tools"]  # same prefix


def test_proxy_answers_list_devices_and_speaks_the_reply():
    h, seen = listing_then({"content": "The office has a desk lamp and a speaker."})
    said, tools, order = lazy_run(h, "what is in the office")
    assert said.strip() == "The office has a desk lamp and a speaker." and not tools and order[-1] == "finish:stop"


def test_list_devices_call_left_after_the_last_round_is_dropped():
    h, seen = listing_then({"content": None, "tool_calls": [
        {"id": "c2", "type": "function", "function": {"name": "ListDevices", "arguments": '{"area": "Porch"}'}}]})
    app.LOCAL_ROUNDS, old = 1, app.LOCAL_ROUNDS
    try:
        said, tools, _ = lazy_run(h, "what is on the porch")
    finally:
        app.LOCAL_ROUNDS = old
    assert not tools and said.strip() == app.FALLBACK_REPLY


# --- with the claim check (PR #19): round-5 review probes on PR #22 ---

def sequence(*replies):
    """Each request gets the next reply: "list:<Area>" streams a ListDevices call (for the
    first, streamed request), a dict is a non-streamed message."""
    seen = []

    async def handler(request):
        body = await request.json()
        seen.append(body)
        reply = replies[min(len(seen), len(replies)) - 1]
        if isinstance(reply, str) and reply.startswith("list:"):
            area = reply.split(":", 1)[1]
            msg = {"content": None, "tool_calls": [{"id": f"l{len(seen)}", "type": "function", "function": {
                "name": "ListDevices", "arguments": json.dumps({"area": area})}}]}
            if body.get("stream"):
                r = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
                await r.prepare(request)
                chunk = {"id": "x", "choices": [{"index": 0, "finish_reason": "tool_calls", "delta": {"tool_calls": [
                    {"index": 0, **msg["tool_calls"][0]}]}}]}
                await r.write(b"data: " + json.dumps(chunk).encode() + b"\n\ndata: [DONE]\n\n")
                return r
            reply = msg
        return web.json_response({"choices": [{"index": 0, "message": {"role": "assistant", **reply},
                                               "finish_reason": "tool_calls" if reply.get("tool_calls") else "stop"}]})
    return handler, seen


TURN_OFF = {"content": None, "tool_calls": [
    {"id": "t1", "type": "function", "function": {"name": "intent__HassTurnOff", "arguments": '{"name": "Office Desk Lamp"}'}}]}


def tool_names(tools):
    return [c["function"]["name"] for t in tools for c in t["choices"][0]["delta"]["tool_calls"]]


def test_claim_after_a_listing_round_is_checked():
    # ListDevices, then a text claim: the claim check re-asks, and a second claim is refused.
    h, seen = sequence("list:Office", {"content": "I've turned off the office desk lamp."},
                       {"content": "I've turned it off."})
    said, tools, _ = lazy_run(h, "turn off the office desk lamp")
    assert said.strip() == app.CLAIM_REPLY and not tools and len(seen) == 3


def test_claim_after_a_listing_round_rerun_acts():
    h, seen = sequence("list:Office", {"content": "I've turned off the office desk lamp."}, TURN_OFF)
    said, tools, _ = lazy_run(h, "turn off the office desk lamp")
    assert tool_names(tools) == ["intent__HassTurnOff"]


def test_a_rerun_listing_is_answered_here_not_by_home_assistant():
    # The first reply claims; the re-run looks the office up, then acts.
    for stream in (True, False):
        first = [content("I've turned off the office desk lamp."), end("stop")] if stream else \
            {"content": "I've turned off the office desk lamp."}
        h, seen = sequence(first, "list:Office", TURN_OFF) if not stream else (None, None)
        if stream:
            calls = []

            async def handler(request, calls=calls):
                calls.append(await request.json())
                if len(calls) == 1:
                    r = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
                    await r.prepare(request)
                    for c in first:
                        await r.write(b"data: " + json.dumps(c).encode() + b"\n\n")
                    await r.write(b"data: [DONE]\n\n")
                    return r
                return await rest(request)
            rest, _ = sequence("list:Office", TURN_OFF)
            h, seen = handler, calls
        said, tools, _ = lazy_run(h, "turn off the office desk lamp", stream=stream)
        names = tool_names(tools) if stream else [c["function"]["name"] for c in tools]
        assert names == ["intent__HassTurnOff"], (stream, names)  # Home Assistant never sees ListDevices


def test_a_rerun_listing_is_in_the_voice_log(tmp_path):
    # Non-streamed: the first reply claims, the re-run looks the office up, then acts.
    h, _ = sequence({"content": "I've turned off the office desk lamp."}, "list:Office", TURN_OFF)
    app.VOICE_LOG, old = str(tmp_path / "log.jsonl"), app.VOICE_LOG
    try:
        lazy_run(h, "turn off the office desk lamp", stream=False)
    finally:
        app.VOICE_LOG = old
    rows = [json.loads(l) for l in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert [r.get("listed") for r in rows] == [["Office"]]


# --- review follow-ups: the non-streamed paths (PR #22 and #21 nits) ---

def _logged(tmp_path, h, text):
    app.VOICE_LOG, old = str(tmp_path / "log.jsonl"), app.VOICE_LOG
    try:
        out = lazy_run(h, text, stream=False)
    finally:
        app.VOICE_LOG = old
    return out, [json.loads(l) for l in (tmp_path / "log.jsonl").read_text().splitlines()]


def test_non_streamed_listing_is_answered_here():
    h, seen = sequence("list:Office", TURN_OFF)
    _, tools, _ = lazy_run(h, "turn off the office desk lamp", stream=False)
    assert [c["function"]["name"] for c in tools] == ["intent__HassTurnOff"] and len(seen) == 2


def test_non_streamed_refusal_logs_the_listing(tmp_path):
    # "what is in the office" is no command, so the guard blocks the call after the listing.
    h, _ = sequence("list:Office", TURN_OFF)
    _, rows = _logged(tmp_path, h, "what is in the office")
    assert [(r.get("blocked"), r.get("listed")) for r in rows] == [(["intent__HassTurnOff"], ["Office"])]


def test_non_streamed_empty_reply_logs_the_listing(tmp_path):
    h, _ = sequence("list:Office", {"content": ""})
    (said, _, _), rows = _logged(tmp_path, h, "what is in the office")
    assert [(r.get("failed"), r.get("listed")) for r in rows] == [("empty reply", ["Office"])]


def test_non_streamed_claim_rerun_logs_its_calls(tmp_path):
    h, _ = sequence({"content": "I've turned off the office desk lamp."}, TURN_OFF)
    _, rows = _logged(tmp_path, h, "turn off the office desk lamp")
    assert [[c["name"] for c in r.get("calls") or []] for r in rows] == [["intent__HassTurnOff"]]
