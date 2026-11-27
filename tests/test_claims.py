"""Issue 14: a reply that claims an action must be backed by a successful call, to a tool
that is not a read, in the same turn. Driven through the real app against a fake LLM, plus
the pure helpers."""
import asyncio
import json

from aiohttp import web

from ha_voice_agent import app, claims, guard
from test_turn_reliability import content, end, run, tool

CFG = guard.load()
TOOLS = [{"type": "function", "function": {"name": n, "parameters": {}}}
         for n in ("light__HassTurnOff", "light__HassLightSet", "homeassistant__GetLiveContext", "intent__HassStartTimer")]


def scripted(*replies):
    """A fake upstream that answers each request with the next reply: a list of SSE chunks
    for a streamed request, or a message dict for a non-streamed one (the re-run)."""
    seen = []

    async def handler(request):
        body = await request.json()
        seen.append(body)
        reply = replies[min(len(seen), len(replies)) - 1]
        if isinstance(reply, dict):
            return web.json_response({"choices": [{"index": 0, "message": {"role": "assistant", **reply},
                                                   "finish_reason": "tool_calls" if reply.get("tool_calls") else "stop"}]})
        r = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await r.prepare(request)
        await r.write(b"".join(b"data: " + json.dumps(c).encode() + b"\n\n" for c in reply) + b"data: [DONE]\n\n")
        return r
    return handler, seen


def call(name, args="{}"):
    return {"id": "r1", "type": "function", "function": {"name": name, "arguments": args}}


# --- helpers ---

def test_claim_phrases():
    for t in ["I've turned off the den light.", "The kitchen light is now on.", "Done.", "Okay, done!",
              "I set the thermostat to 68 degrees.", "The lights have been turned off.", "Turning off the porch light.",
              "I have paused the music.", "All set."]:
        assert claims.claims_action(t, CFG), t
    for t in ["Which light do you mean?", "The den light is off.", "Sorry, I can't do that.",
              "It is 68 degrees in the den.", "I don't see a porch light.", "",
              # A reply that asks back confirms nothing, even after a status line (PR 19 review).
              "The thermostat is currently set to 70 degrees. What would you like it set to?",
              "The dining room is now at 70 degrees. Which thermostat do you mean?",
              "Playing jazz requires a media player. Which one?  ",
              "Done? I couldn't find that light."]:
        assert not claims.claims_action(t, CFG), t
    # Still claims: it reports a change as made, and a command turn with nothing behind it
    # must not speak that. The cost of the "?" rule: a claim with a question after it passes.
    assert claims.claims_action("Sorry, the vacuum is now docked and charging.", CFG)
    assert not claims.claims_action("I've turned off the light. Anything else?", CFG)


def test_turn_writes_reads_results_since_the_last_user_message():
    old = [{"role": "user", "content": "turn off the light"},
           {"role": "assistant", "tool_calls": [call("light__HassTurnOff") | {"id": "a"}]},
           {"role": "tool", "tool_call_id": "a", "content": json.dumps({"response_type": "action_done"})}]
    msgs = old + [{"role": "assistant", "content": "Done."}, {"role": "user", "content": "turn on the fan"}]
    assert claims.turn_writes(msgs, CFG) == (0, 0)  # the earlier turn's write doesn't count
    ok = msgs + [{"role": "assistant", "tool_calls": [call("light__HassTurnOff") | {"id": "b"}]},
                 {"role": "tool", "tool_call_id": "b", "content": json.dumps({"response_type": "action_done",
                                                                             "data": {"success": [{"name": "x"}], "failed": []}})}]
    assert claims.turn_writes(ok, CFG) == (1, 0)
    for bad in [{"error": "HomeAssistantError", "error_text": "no such entity"},
                {"response_type": "error"}, {"data": {"success": [], "failed": [{"name": "x"}]}}]:
        failed = msgs + [{"role": "assistant", "tool_calls": [call("light__HassTurnOff") | {"id": "c"}]},
                         {"role": "tool", "tool_call_id": "c", "content": json.dumps(bad)}]
        assert claims.turn_writes(failed, CFG) == (0, 1), bad
    for name in ["homeassistant__GetLiveContext", "llm__GetDateTime", "intent__HassTimerStatus", "todo__get_items",
                 "intent_script__WeatherForecast", "intent_script__AlarmStatus", "media_player__search_media"]:
        read = msgs + [{"role": "assistant", "tool_calls": [call(name) | {"id": "d"}]},
                       {"role": "tool", "tool_call_id": "d", "content": "{}"}]
        assert claims.turn_writes(read, CFG) == (0, 0), name  # a read is neither
    # A tool that acts but is not in the guard's "actions" list still backs a claim.
    for name in ["assist_satellite__HassBroadcast", "intent__HassIncreaseTimer", "intent__HassPauseTimer",
                 "intent_script__FeedTheCat"]:
        assert not guard.is_action(name, CFG), name
        acted = msgs + [{"role": "assistant", "tool_calls": [call(name) | {"id": "e"}]},
                        {"role": "tool", "tool_call_id": "e", "content": json.dumps({"response_type": "action_done"})}]
        assert claims.turn_writes(acted, CFG) == (1, 0), name
    orphan = msgs + [{"role": "tool", "tool_call_id": "nope", "content": "{}"}]
    assert claims.turn_writes(orphan, CFG) == (0, 0)  # a result with no call behind it


def test_reads_do_not_change_the_guard():
    # search_media is in both lists: it needs a command, but only searches.
    assert guard.is_action("media_player__search_media", CFG) and guard.is_read("media_player__search_media", CFG)
    for name in ["homeassistant__GetLiveContext", "intent__HassTimerStatus", "todo__get_items"]:
        assert guard.is_read(name, CFG) and not guard.is_action(name, CFG), name
    msgs = [{"role": "user", "content": "the cabinet"}]
    assert guard.blocked(msgs, ["assist_satellite__HassBroadcast", "homeassistant__GetLiveContext"], CFG) == []
    assert guard.blocked(msgs, ["media_player__search_media"], CFG) == ["media_player__search_media"]


def test_questions_and_non_commands_are_never_checked():
    for text in ["did you turn off the light?", "is the den light on", "what is the capital of Chile", "I'm cold"]:
        assert not claims.needs_check([{"role": "user", "content": text}], CFG), text
    assert claims.needs_check([{"role": "user", "content": "turn off the den light"}], CFG)


def test_scope_tools():
    names = lambda ts: [t["function"]["name"] for t in ts]
    assert names(claims.scope_tools(TOOLS, "act", CFG)) == names(TOOLS)
    for purpose in ("ground", "later"):
        kept = names(claims.scope_tools(TOOLS, purpose, CFG))
        assert not [n for n in kept if guard.is_action(n, CFG)], purpose
        assert "homeassistant__GetLiveContext" in kept
    # Only reads: a tool that acts but is missing from "actions" is not offered either.
    extra = TOOLS + [{"type": "function", "function": {"name": "assist_satellite__HassBroadcast", "parameters": {}}}]
    assert names(claims.scope_tools(extra, "later", CFG)) == ["homeassistant__GetLiveContext"]


def test_later_phrases():
    for t in ["turn off the light in 30 minutes", "turn on the porch light at 7", "start the vacuum tomorrow",
              "turn off the fan in an hour"]:
        assert claims.is_later(t, CFG), t
    for t in ["turn off the light", "set the thermostat to 68", "turn on the kitchen counter light"]:
        assert not claims.is_later(t, CFG), t


def test_rerun_body_keeps_the_prefix():
    body = {"messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "turn off the light"}],
            "tools": TOOLS, "stream": True, "stream_options": {"include_usage": True}}
    rb = claims.rerun_body(body, CFG)
    assert rb["messages"][0] == body["messages"][0] and rb["tools"] == TOOLS  # cache prefix unchanged
    assert rb["messages"][1]["content"].startswith("turn off the light\n(") and not rb["stream"]
    assert "stream_options" not in rb and body["messages"][1]["content"] == "turn off the light"


def test_allowed_calls_drops_tools_not_offered():
    offered = claims.scope_tools(TOOLS, "ground", CFG)
    got = claims.allowed_calls([call("light__HassTurnOff"), call("homeassistant__GetLiveContext")], offered)
    assert [c["function"]["name"] for c in got] == ["homeassistant__GetLiveContext"]


# --- through the proxy ---

def test_unbacked_claim_is_rerun_and_the_tool_call_goes_to_home_assistant():
    h, seen = scripted([content("I've turned off the den light."), end("stop")],
                       {"content": None, "tool_calls": [call("light__HassTurnOff", '{"area": "den"}')]})
    said, tools, order = run(h, "turn off the den light", req_tools=TOOLS)
    assert "turned off" not in said and len(seen) == 2
    assert seen[1]["stream"] is False and seen[1]["messages"][-1]["content"].endswith(f"({claims.RERUN_NOTE})")
    assert tools[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "light__HassTurnOff"
    assert order[-1] == "finish:tool_calls"
    assert said.strip() == app.WORKING_PHRASE


def test_rerun_call_to_a_tool_not_offered_is_dropped():
    h, seen = scripted([content("I've turned off the light."), end("stop")], {"content": None, "tool_calls": [call("lock__HassUnlock")]})
    said, tools, _ = run(h, "turn off the den light", req_tools=TOOLS)
    assert said.strip() == app.CLAIM_REPLY and not tools


def test_rerun_that_asks_back_is_spoken():
    h, seen = scripted([content("I've turned off the light."), end("stop")], {"content": "Which light do you mean?"})
    said, tools, _ = run(h, "turn off the light", req_tools=TOOLS)
    assert said.strip() == "Which light do you mean?" and not tools and len(seen) == 2


def test_failed_rerun_says_claim_reply():
    calls = []

    async def handler(request):
        calls.append(1)
        if len(calls) == 1:
            return await scripted([content("I've turned off the light."), end("stop")])[0](request)
        return web.Response(status=500, text="boom")
    said, _, _ = run(handler, "turn off the light", req_tools=TOOLS)
    assert said.strip() == app.CLAIM_REPLY and len(calls) == 2


def test_unbacked_claim_twice_says_claim_reply():
    h, seen = scripted([content("Done, the light is now off."), end("stop")], {"content": "I've turned it off."})
    said, tools, _ = run(h, "turn off the den light")
    assert said.strip() == app.CLAIM_REPLY and not tools and len(seen) == 2


def test_unbacked_claim_non_streamed():
    h, seen = scripted({"content": "I've turned off the light."}, {"content": "Okay, done."})
    said, tools, _ = run(h, "turn off the den light", stream=False)
    assert said == app.CLAIM_REPLY and not tools and len(seen) == 2
    h, seen = scripted({"content": "I've turned off the light."}, {"content": None, "tool_calls": [call("light__HassTurnOff")]})
    said, tools, _ = run(h, "turn off the den light", stream=False, req_tools=TOOLS)
    assert tools[0]["function"]["name"] == "light__HassTurnOff" and not said


def test_backed_claim_passes():
    # The action landed earlier in this turn: the confirmation after its result is spoken as is.
    h, seen = scripted([content("The den light is now off."), end("stop")])
    said, _, _ = run(h, "turn off the den light", after_tool=True, tool_name="light__HassTurnOff",
                     tool_result=json.dumps({"response_type": "action_done"}))
    assert said.strip() == "The den light is now off." and len(seen) == 1


def test_claim_after_a_failed_action_is_replaced_without_a_rerun():
    for result in [{"error": "HomeAssistantError", "error_text": "not found"}, {"response_type": "error"}]:
        h, seen = scripted([content("Done, the light is off."), end("stop")])
        said, _, _ = run(h, "turn off the den light", after_tool=True, tool_name="light__HassTurnOff",
                         tool_result=json.dumps(result))
        assert said.strip() == app.CLAIM_REPLY and len(seen) == 1, result


def test_claim_after_only_a_read_is_replaced():
    h, seen = scripted([content("I've turned off the den light."), end("stop")])
    said, _, _ = run(h, "turn off the den light", after_tool=True, tool_name="homeassistant__GetLiveContext")
    assert said.strip() == app.CLAIM_REPLY and len(seen) == 1


def test_question_is_never_checked_through_the_proxy():
    h, seen = scripted([content("Yes, I turned it off."), end("stop")])
    said, _, _ = run(h, "did you turn off the den light?")
    assert said.strip() == "Yes, I turned it off." and len(seen) == 1


def test_read_only_reply_to_a_command_is_not_replaced():
    h, seen = scripted([content("Which light do you mean?"), end("stop")])
    said, _, _ = run(h, "turn off the light")
    assert said.strip() == "Which light do you mean?" and len(seen) == 1


def test_later_request_gets_no_rerun():
    h, seen = scripted([content("Okay, I've set the light to turn off in 30 minutes."), end("stop")])
    said, _, _ = run(h, "turn off the light in 30 minutes")
    assert said.strip() == app.CLAIM_REPLY and len(seen) == 1


def test_confirmation_after_a_tool_missing_from_actions_is_spoken():
    # PR 19 review: HassBroadcast acts but is not in "actions"; its success backs "Done.".
    h, seen = scripted([content("Done."), end("stop")])
    said, _, _ = run(h, "send a message to everyone that dinner is ready", after_tool=True,
                     tool_name="assist_satellite__HassBroadcast", tool_result=json.dumps({"response_type": "action_done"}))
    assert said.strip() == "Done." and len(seen) == 1


def test_status_then_question_is_spoken_without_a_rerun():
    # PR 19 review: a reply that reports the state and asks back is not a claim.
    h, seen = scripted([content("The thermostat is currently set to 70 degrees. What would you like it set to?"), end("stop")])
    said, _, _ = run(h, "adjust the thermostat", after_tool=True, tool_name="homeassistant__GetLiveContext")
    assert said.strip() == "The thermostat is currently set to 70 degrees. What would you like it set to?" and len(seen) == 1
    h, seen = scripted([content("The dining room is now at 70 degrees. Which thermostat do you mean?"), end("stop")])
    said, _, _ = run(h, "set the dining room to 72", req_tools=TOOLS)
    assert said.strip() == "The dining room is now at 70 degrees. Which thermostat do you mean?" and len(seen) == 1
    h, seen = scripted({"content": "Playing jazz requires a media player. Which one?"})
    said, tools, _ = run(h, "play some jazz", stream=False, req_tools=TOOLS)
    assert said == "Playing jazz requires a media player. Which one?" and not tools and len(seen) == 1


def test_claim_before_the_upstream_drops_says_claim_reply():
    # PR 19 review: the claim streams in full, then the connection stalls past the timeout.
    async def stall(request):
        r = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await r.prepare(request)
        await r.write(b"data: " + json.dumps(content("I've turned off the light.")).encode() + b"\n\n")
        await asyncio.sleep(3)
        return r
    said, _, order = run(stall, "turn off the light", timeout=0.5)
    assert said.strip() == app.CLAIM_REPLY and order[-1] == "finish:stop"
    # Text that claims nothing, cut off the same way, is still spoken as before.
    async def stall_plain(request):
        r = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await r.prepare(request)
        await r.write(b"data: " + json.dumps(content("Which light do you mean? ")).encode() + b"\n\n")
        await asyncio.sleep(3)
        return r
    said, _, _ = run(stall_plain, "turn off the light", timeout=0.5)
    assert said.strip() == "Which light do you mean?"


def test_tool_call_still_streams_with_held_text_first():
    h, seen = scripted([content("Turning it off. "), tool("light__HassTurnOff", finish="tool_calls")])
    said, tools, order = run(h, "turn off the den light")
    assert said.strip() == "Turning it off." and tools and order[-1] == "tool" and len(seen) == 1


def test_first_stream_is_closed_before_the_rerun_is_sent():
    # Review round 5: a claim longer than MAX_WORDS ends on the budget-cut path, and the
    # re-run must not be posted while the first generation is still streaming. The claim
    # has sentence ends, so the cutter really reaches its budget mid-stream.
    state = {"n": 0, "first": None, "first_closing": None}

    async def handler(request):
        state["n"] += 1
        if state["n"] == 1:
            state["first"] = request
            r = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await r.prepare(request)
            text = "I've turned off the light. " + "It is off now. " * 15
            try:
                for w in text.split(" "):
                    await r.write(b"data: " + json.dumps(content(w + " ")).encode() + b"\n\n")
                    await asyncio.sleep(0.02)
            except (ConnectionResetError, ConnectionError):
                pass
            return r
        await asyncio.sleep(0)
        t = state["first"].transport
        state["first_closing"] = t is None or t.is_closing()
        return web.json_response({"choices": [{"index": 0, "message": {"role": "assistant", "content": "Which light?"},
                                               "finish_reason": "stop"}]})

    said, _, _ = run(handler, "turn off the light", req_tools=TOOLS)
    assert state["n"] == 2 and state["first_closing"] is True, state
    assert said.strip() == "Which light?"
