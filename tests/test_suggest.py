"""Issue 7: the voice log ties a turn's requests together, and repeated, verified
commands become suggested custom sentences."""
import asyncio
import json
from pathlib import Path

import pytest
from aiohttp import web

from ha_voice_agent import app, guard, suggest, turns
from test_turn_reliability import content, end, run, streaming, tool

CFG = guard.load()
REQ = json.loads((Path(__file__).parent / "fixtures" / "ha_request.json").read_text())
OK = json.dumps({"response_type": "action_done", "data": {"success": [{"id": "light.den_sconce_two", "name": "x"}], "failed": []}})


def turn(text, call="light__HassTurnOff", args=None, result=OK, first=True, spoken="Done.", tid=None):
    tid = tid or f"t{abs(hash((text, call, result, first, spoken, id(args))))}"
    args = args if args is not None else {"name": "Den Sconce Two"}
    return [{"turn": tid, "first_turn": first, "after_tool": False, "user": text, "calls": [{"name": call, "args": args}]},
            {"turn": tid, "first_turn": first, "after_tool": True, "user": text, "spoken": spoken,
             "results": [{"name": call, **turns.result(result)}]}]


def rows(*ts, times=1):
    # A repeat keeps its turn id: the proxy hashes the same first turn to the same id in
    # every conversation.
    out = []
    for n in range(times):
        for t in ts:
            out += [dict(r) for r in t]
    return out


def test_turn_fields_are_stable_across_a_turns_requests():
    first = [{"role": "system", "content": "S"}, {"role": "user", "content": "turn off the light"}]
    after = first + [{"role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": "light__HassTurnOff"}}]},
                     {"role": "tool", "tool_call_id": "c1", "content": OK}]
    a, b = turns.fields(first), turns.fields(after)
    assert a["turn"] == b["turn"] and a["first_turn"] and "results" not in a
    assert b["results"] == [{"name": "light__HassTurnOff", "ok": True, "entities": ["light.den_sconce_two"]}]
    later = after + [{"role": "assistant", "content": "Done."}, {"role": "user", "content": "and the fan"}]
    assert turns.fields(later)["turn"] != a["turn"] and not turns.fields(later)["first_turn"]


def test_results_report_errors():
    assert not turns.result(json.dumps({"error": "HomeAssistantError"}))["ok"]
    assert not turns.result(json.dumps({"data": {"success": [], "failed": [{"id": "light.x"}]}}))["ok"]
    assert turns.result("not json") == {"ok": True, "entities": []}


def test_repeated_verified_command_is_suggested():
    got, dropped = suggest.suggest(rows(turn("Turn off den sconce two."), times=3), CFG)
    assert got == [{"text": "turn off den sconce two", "intent": "HassTurnOff",
                    "slots": {"name": "Den Sconce Two"}, "count": 3}] and not dropped
    y = suggest.to_yaml(got)
    assert '          - "turn off den sconce two"' in y and '          name: "Den Sconce Two"' in y
    assert "  HassTurnOff:\n    data:\n" in y


def test_below_the_repeat_threshold_nothing():
    assert suggest.suggest(rows(turn("turn off den sconce two"), times=2), CFG) == ([], [])


def test_eligibility_rules():
    bad = {
        "not first": turn("turn off den sconce two", first=False),
        "failed": turn("turn off den sconce two", result=json.dumps({"error": "x"})),
        "no entity": turn("turn off den sconce two", result=json.dumps({"response_type": "action_done"})),
        "question": turn("turn off den sconce two", spoken="Which one?"),
        "read only": turn("is den sconce two on", call="homeassistant__GetLiveContext", args={}),
        "safety": turn("lock the front door", call="lock__HassTurnOn", args={"name": "Front Door"},
                       result=json.dumps({"data": {"success": [{"id": "lock.front_door"}]}})),
    }
    for why, t in bad.items():
        assert suggest.suggest(rows(t, times=4), CFG)[0] == [], why


def test_same_words_that_once_did_something_else_are_dropped():
    good = rows(turn("turn off the lights"), times=3)
    other = rows(turn("turn off the lights", args={"area": "Kitchen"}, tid="o"))
    got, dropped = suggest.suggest(good + other, CFG)
    assert got == [] and "did something else" in dropped[0]["why"]


def test_proxy_logs_calls_and_results_against_one_turn(tmp_path, monkeypatch):
    log = tmp_path / "voice.jsonl"
    monkeypatch.setattr(app, "VOICE_LOG", str(log))
    run(streaming(tool("light__HassTurnOff", '{"name": "Den Sconce Two"}', finish="tool_calls")),
        "turn off den sconce two")
    run(streaming(content("Done."), end("stop")), "turn off den sconce two", after_tool=True,
        tool_name="light__HassTurnOff", tool_result=OK)
    a, b = [json.loads(l) for l in log.read_text().splitlines()]
    assert a["turn"] == b["turn"] and a["first_turn"] and b["first_turn"]
    assert a["calls"] == [{"name": "light__HassTurnOff", "args": {"name": "Den Sconce Two"}}]
    assert b["results"] == [{"name": "light__HassTurnOff", "ok": True, "entities": ["light.den_sconce_two"]}]


def test_same_command_in_three_conversations_through_the_proxy_is_suggested(tmp_path, monkeypatch):
    log = tmp_path / "voice.jsonl"
    monkeypatch.setattr(app, "VOICE_LOG", str(log))
    for _ in range(3):
        run(streaming(tool("light__HassTurnOff", '{"name": "Den Sconce Two"}', finish="tool_calls")),
            "turn off den sconce two")
        run(streaming(content("Done."), end("stop")), "turn off den sconce two", after_tool=True,
            tool_name="light__HassTurnOff", tool_result=OK)
    got, dropped = suggest.suggest(suggest.load(str(log)), CFG)
    assert got == [{"text": "turn off den sconce two", "intent": "HassTurnOff",
                    "slots": {"name": "Den Sconce Two"}, "count": 3}] and not dropped


def test_a_fixed_area_is_marked_for_the_reviewer():
    got, _ = suggest.suggest(rows(turn("turn off the lights", args={"area": "Kitchen", "domain": ["light"]}), times=3), CFG)
    y = suggest.to_yaml(got)
    assert "# area is fixed" in y and '          area: "Kitchen"' in y
    assert "# area is fixed" not in suggest.to_yaml(suggest.suggest(rows(turn("turn off den sconce two"), times=3), CFG)[0])


def debugger(answer, texts):
    """already_local against a fake Home Assistant websocket that sends answer for id 1."""
    async def ws(request):
        w = web.WebSocketResponse()
        await w.prepare(request)
        await w.send_json({"type": "auth_required"})
        await w.receive_json()
        await w.send_json({"type": "auth_ok"})
        await w.receive_json()
        await w.send_json({"id": 1, "type": "result", **answer})
        await w.close()
        return w

    async def go():
        a = web.Application()
        a.router.add_get("/api/websocket", ws)
        runner = web.AppRunner(a)
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", 0).start()
        try:
            return await suggest.already_local(f"ws://127.0.0.1:{runner.addresses[0][1]}/api/websocket", "t", texts)
        finally:
            await runner.cleanup()

    return asyncio.run(go())


def test_a_failed_debugger_call_is_reported(capsys):
    assert debugger({"success": False, "error": {"code": "unknown_command"}}, ["x"]) == {}
    assert "sentence debugger failed" in capsys.readouterr().err


def test_only_a_debugger_match_counts_as_local():
    # Home Assistant names the closest intent even when it did not match; a sentence
    # trigger matches with no intent at all.
    results = [{"intent": {"name": "HassTurnOff"}, "match": False, "unmatched_slots": {"name": "den sconce two"}},
               {"match": True, "source": "trigger", "sentence_template": "movie time"},
               {"intent": {"name": "HassTurnOn"}, "match": True, "source": "builtin"},
               None]
    got = debugger({"success": True, "result": {"results": results}},
                   ["turn off den sconce two", "movie time", "turn on the kitchen light", "what now"])
    assert got == {"movie time": "a sentence trigger", "turn on the kitchen light": "HassTurnOn"}


def test_main_drops_only_what_the_debugger_matched(tmp_path, monkeypatch, capsys):
    log = tmp_path / "voice.jsonl"
    log.write_text("".join(json.dumps(r) + "\n" for r in rows(turn("turn off den sconce two"),
                                                              turn("turn off the porch light"), times=3)))
    seen = {}

    async def fake(url, token, texts, insecure=False):
        seen.update(url=url, token=token, insecure=insecure)
        return {"turn off the porch light": "a sentence trigger"}

    monkeypatch.setattr(suggest, "already_local", fake)
    monkeypatch.setenv("HA_TOKEN", "tok")
    suggest.main([str(log), "--ha", "wss://ha:8123/api/websocket"])
    out, err = capsys.readouterr()
    assert '"turn off den sconce two"' in out and "porch" not in out
    assert "'turn off the porch light': already matches a sentence trigger locally" in err
    assert seen == {"url": "wss://ha:8123/api/websocket", "token": "tok", "insecure": False}
    suggest.main([str(log), "--ha", "wss://ha:8123/api/websocket", "--insecure"])
    assert seen["insecure"] is True


def test_ha_without_a_token_says_so(tmp_path, monkeypatch, capsys):
    log = tmp_path / "voice.jsonl"
    log.write_text("")
    monkeypatch.delenv("HA_TOKEN", raising=False)
    with pytest.raises(SystemExit) as e:
        suggest.main([str(log), "--ha", "wss://ha:8123/api/websocket"])
    assert e.value.code == 2 and "HA_TOKEN" in capsys.readouterr().err


def test_grouping_uses_what_was_heard_not_the_correction():
    def corrected(heard):
        t = turn("turn off the Office Speaker", args={"name": "Office Speaker"}, tid=f"h{heard}")
        return [{**r, "heard": heard} if not r["after_tool"] else r for r in t]

    # Home Assistant's matcher saw "speeker", so that is the sentence that would save the LLM.
    got, _ = suggest.suggest(rows(corrected("turn off the office speeker"), times=3), CFG)
    assert [c["text"] for c in got] == ["turn off the office speeker"]
    # Three corrected turns heard three ways are three groups of one, not one of three.
    mixed = rows(corrected("turn off the office speeker"), corrected("turn off the offis speaker"),
                 turn("turn off the office speaker", args={"name": "Office Speaker"}, tid="plain"))
    assert suggest.suggest(mixed, CFG) == ([], [])


def test_safety_words_are_searched_in_the_arguments_not_the_tool_name():
    ok = json.dumps({"data": {"success": [{"id": "input_datetime.wake_up"}]}})
    alarm = turn("wake me up at seven", call="set_alarm", args={"time": "07:00"}, result=ok)
    assert suggest.judge(alarm, CFG)[2] == "ok"
    got, dropped = suggest.suggest(rows(alarm, times=3), CFG)
    assert got == [] and dropped[0]["why"] == "set_alarm is not a built-in intent"
    locked = json.dumps({"data": {"success": [{"id": "lock.front"}]}})
    assert suggest.judge(turn("turn on the front", call="lock__HassTurnOn", args={"name": "Front"}, result=locked),
                         CFG)[2] == "safety-critical"
    assert suggest.judge(turn("turn on the porch", args={"name": "Garage Light"}), CFG)[2] == "safety-critical"
    # The corrected request is checked as well as what was heard.
    t = turn("turn on the Gate Light", args={"area": "Outside"})
    assert suggest.judge([{**t[0], "heard": "turn on the great light"}, t[1]], CFG)[2] == "safety-critical"


def test_a_corrected_first_request_keeps_its_turn_fields(tmp_path, monkeypatch):
    async def upstream(request):
        return web.json_response({"choices": [{"message": {"role": "assistant", "content": "OK."}, "finish_reason": "stop"}]})

    system = {"role": "system", "content": REQ["messages"][0]["content"]}
    monkeypatch.setattr(app, "AGENT_MODE", False)
    monkeypatch.setattr(app, "VOICE_LOG", str(tmp_path / "voice.jsonl"))
    run(upstream, "turn off the office speeker", stream=False, history=[system])
    line = json.loads((tmp_path / "voice.jsonl").read_text().splitlines()[-1])
    assert line["heard"] == "turn off the office speeker" and line["user"] == "turn off the Office Speaker"
    assert line["turn"] == turns.fields([system, {"role": "user", "content": "turn off the office speeker"}])["turn"]
    assert line["first_turn"] is True


def test_the_certificate_is_verified_unless_insecure(monkeypatch):
    import aiohttp
    seen = []

    def ws_connect(self, url, **kw):
        seen.append(kw.get("ssl"))
        raise RuntimeError("stop")

    monkeypatch.setattr(aiohttp.ClientSession, "ws_connect", ws_connect)
    for insecure in (False, True):
        with pytest.raises(RuntimeError):
            asyncio.run(suggest.already_local("wss://ha:8123/api/websocket", "t", ["x"], insecure))
    assert seen == [True, False]
