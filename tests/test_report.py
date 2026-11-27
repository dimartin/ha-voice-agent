"""Issue 9: a report of repeated and failed requests, from the voice log and Home
Assistant's pipeline runs."""
import asyncio
from types import SimpleNamespace

import aiohttp
import pytest

from ha_voice_agent import guard, report

CFG = guard.load()
T0 = "2026-11-25T10:00:{:02d}+0000"


def v(sec, user, spoken="", tools=(), after_tool=False, **kw):
    return {"ts": T0.format(sec), "user": user, "spoken": spoken, "tools": list(tools), "after_tool": after_tool, "s": 2, **kw}


def run_(sec, heard, device="pod1", local=False, speech=""):
    return {"id": f"r{sec}", "ts": f"2026-11-25T10:00:{sec:02d}+00:00", "heard": heard, "device": device, "local": local,
            "speech": speech}


def test_epoch_reads_both_timestamp_forms():
    assert report.epoch("2026-11-25T10:00:00+0000") == report.epoch("2026-11-25T10:00:00.000+00:00")
    assert report.epoch(None) is None and report.epoch("nonsense") is None


def test_repeats_on_one_device_within_the_window():
    runs = [run_(0, "turn on the kitchen light", speech="Which one?"), run_(8, "Turn on the kitchen light."),
            run_(9, "turn on the kitchen light", device="dot2"), run_(50, "what time is it")]
    rep = report.build([], runs, [], CFG)
    assert [(r["first"], r["after_s"], r["first_reply"]) for r in rep["repeated"]] == \
           [("turn on the kitchen light", 8.0, "Which one?")]


def test_scripted_replays_and_warmups_are_not_repeats():
    runs = [run_(0, "turn on the kitchen light"), run_(1, "turn on the kitchen light"),
            run_(10, "warmup"), run_(20, "warmup")]
    assert report.build([], runs, [], CFG)["repeated"] == []


def test_llm_problems():
    voice = [
        v(0, "turn on the kitchen counter light", spoken="On it is."),                       # command, no tool
        v(10, "what is the capital of France", spoken="Paris."),                             # fine
        v(20, "how do I make scrambled eggs", spoken="Whisk them."),                         # a question, not a command
        v(30, "set the heat to seventy wand", spoken="Did you mean 70 degrees?"),            # asked back
        v(40, "is it going to rain", spoken="Sorry, I can't answer right now.", failed="SocketTimeoutError"),
        v(45, "cancel", spoken="Okay.", s=0),                                               # answered by the proxy
        v(50, "turn off the porch light", spoken="One moment.", tools=["light__HassTurnOff"]),
        v(52, "turn off the porch light", spoken="Done.", after_tool=True),                  # fine: a tool ran
    ]
    got = [(p["text"], p["why"]) for p in report.build(voice, [], [], CFG)["llm_problems"]]
    assert got == [("turn on the kitchen counter light", "command, no tool call"),
                   ("set the heat to seventy wand", "asked back"),
                   ("is it going to rain", "error: SocketTimeoutError")]


def test_frequent_and_voice_log_only():
    voice = [v(0, "what time is it"), v(30, "What time is it?"), v(59, "warmup")]
    rep = report.build(voice, [], [], CFG)
    assert rep["frequent"] == [{"text": "what time is it", "count": 2}] and rep["requests"] == 3


def test_satellite_rows_no_run_accounts_for():
    runs = [run_(0, "turn on the light")]
    sat = [{"t": report.epoch(T0.format(1)), "text": "turn on the light", "device": "pod1", "src": "satellite"},
           {"t": report.epoch(T0.format(40)), "text": "hey", "device": "pod1", "src": "satellite"}]
    assert [s["text"] for s in report.build([], runs, sat, CFG)["satellite_only"]] == ["hey"]


def test_summarize_a_pipeline_run():
    events = [
        {"type": "run-start", "timestamp": "2026-11-24T23:34:07+00:00", "data": {"conversation_id": "c1"}},
        {"type": "intent-start", "data": {"intent_input": "is the den light on", "device_id": "d1", "engine": "conversation.x"}},
        {"type": "intent-progress", "data": {"chat_log_delta": {"tool_calls": [{"tool_name": "homeassistant__GetLiveContext"}]}}},
        {"type": "intent-progress", "data": {"chat_log_delta": {"role": "tool_result", "result": {"error": "boom"}}}},
        {"type": "intent-end", "data": {"processed_locally": False, "intent_output": {"response": {
            "response_type": "action_done", "speech": {"plain": {"speech": "It is off. "}}}}}},
        {"type": "run-end", "data": None}]
    s = report.summarize("r1", events)
    assert s == {"id": "r1", "ts": "2026-11-24T23:34:07+00:00", "conversation_id": "c1", "heard": "is the den light on",
                 "device": "d1", "agent": "conversation.x", "tools": ["homeassistant__GetLiveContext"], "tool_errors": 1,
                 "local": False, "response_type": "action_done", "speech": "It is off."}


def sat_(sec, text, device="kitchen-dot"):
    return {"t": report.epoch(T0.format(sec)), "text": text, "device": device, "src": "satellite"}


def test_satellite_rows_a_run_accounts_for_count_once():
    # Review finding 1: the satellite's own rows for two runs were added on top of them,
    # doubling the count, the frequencies and the repeats.
    runs = [run_(0, "what time is it", device="d1"), run_(8, "what time is it", device="d1")]
    sat = [sat_(1, "what time is it"), sat_(9, "what time is it"), sat_(40, "hey")]
    rep = report.build([], runs, sat, CFG)
    assert rep["requests"] == 3
    assert rep["frequent"] == [{"text": "what time is it", "count": 2}, {"text": "hey", "count": 1}]
    assert [(r["device"], r["after_s"]) for r in rep["repeated"]] == [("d1", 8.0)]
    assert [s["text"] for s in rep["satellite_only"]] == ["hey"]


def test_satellite_rows_without_runs_all_count():
    rep = report.build([v(0, "what time is it")], [], [sat_(30, "turn on the light")], CFG)
    assert rep["requests"] == 2


def test_a_repeated_opening_is_two_turns_although_its_turn_id_is_the_same():
    # The turn id hashes the conversation up to the user's message, so the same words
    # opening two conversations share it. Each must stay its own turn.
    voice = [v(0, "turn on the kitchen counter light", spoken="On it is.", turn="t1"),
             v(5, "turn on the porch light", spoken="One moment.", tools=["light__HassTurnOn"], turn="t2"),
             v(20, "turn on the kitchen counter light", spoken="Sure thing.", turn="t1"),
             v(21, "turn on the porch light", spoken="Done.", after_tool=True, turn="t2")]
    turns = report.llm_turns(voice)
    assert [[r["ts"][-7:-5] for r in g] for g in turns] == [["00"], ["05", "21"], ["20"]]
    got = [(p["ts"][-7:-5], p["why"]) for p in report.build(voice, [], [], CFG)["llm_problems"]]
    assert got == [("00", "command, no tool call"), ("20", "command, no tool call")]


def _collect_with(monkeypatch, tmp_path, outcomes):
    """Run collect --every with snapshot results (or exceptions) taken in turn; stop after
    the last."""
    todo = list(outcomes)

    async def fake_snapshot(url, token, seen, new):
        o = todo.pop(0)
        if isinstance(o, tuple):  # (rows fetched, then the failure)
            new.extend(o[0])
            raise o[1]
        if isinstance(o, BaseException):
            raise o
        new.extend(o)
        return new

    class Done(BaseException):
        pass

    def fake_sleep(_):
        if not todo:
            raise Done

    monkeypatch.setattr(report, "snapshot", fake_snapshot)
    monkeypatch.setattr(report.time, "sleep", fake_sleep)
    out = tmp_path / "runs.jsonl"
    with pytest.raises(Done):
        report.collect("wss://ha/api/websocket", "t", str(out), every=60)
    return [r["id"] for r in report.load(str(out))]


def test_collect_every_survives_websocket_failures(monkeypatch, tmp_path):
    # Review finding 2: a proxy's 502 while HA restarts, and HA closing the socket
    # mid-snapshot, are neither OSError nor TimeoutError and ended the loop.
    handshake = aiohttp.WSServerHandshakeError(request_info=SimpleNamespace(real_url="wss://ha/api/websocket"), history=(),
                                               status=502, message="Bad Gateway")
    closed = getattr(aiohttp, "WSMessageTypeError", TypeError)("Received message 8 is not WSMsgType.TEXT")
    assert isinstance(closed, TypeError) and not isinstance(handshake, OSError)
    assert _collect_with(monkeypatch, tmp_path, [handshake, closed, aiohttp.ServerDisconnectedError(),
                                                 asyncio.TimeoutError(), [{"id": "r1"}]]) == ["r1"]


def test_collect_still_stops_on_failed_auth(monkeypatch, tmp_path):
    def sleep(_):
        raise AssertionError("collect went on after a failed auth")

    monkeypatch.setattr(report.time, "sleep", sleep)

    async def refused(url, token, seen, new):
        raise SystemExit("Home Assistant auth failed")

    monkeypatch.setattr(report, "snapshot", refused)
    with pytest.raises(SystemExit):
        report.collect("wss://ha/api/websocket", "t", str(tmp_path / "runs.jsonl"), every=60)


def test_collect_keeps_runs_fetched_before_a_failure(monkeypatch, tmp_path):
    # Review round 3: HA closing the socket after r1 was fetched lost r1 for good, since
    # its id was already in `seen`.
    r = lambda i: {"id": i}
    assert _collect_with(monkeypatch, tmp_path, [([r("r1")], TypeError("closed")), [r("r2")]]) == ["r1", "r2"]


def test_collect_once_fails_loudly(monkeypatch, tmp_path):
    async def fake_snapshot(url, token, seen, new):
        new.append({"id": "r1"})
        raise OSError("unreachable")
    monkeypatch.setattr(report, "snapshot", fake_snapshot)
    out = tmp_path / "runs.jsonl"
    with pytest.raises(OSError):
        report.collect("wss://ha/api/websocket", "t", str(out))
    assert [x["id"] for x in report.load(str(out))] == ["r1"]


class _FakeWS:
    """Home Assistant's websocket for snapshot(): one pipeline, runs r1..r3; the get for
    any run in `fail_on` closes the socket (aiohttp raises TypeError from receive_json)."""

    def __init__(self, fail_on=()):
        self.fail_on, self.out = set(fail_on), [{"type": "auth_required"}]

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def send_json(self, m):
        t = m.get("type")
        if t == "auth":
            self.out.append({"type": "auth_ok"})
        elif t == "assist_pipeline/pipeline/list":
            self.out.append({"id": m["id"], "result": {"pipelines": [{"id": "p", "name": "Home"}]}})
        elif t == "assist_pipeline/pipeline_debug/list":
            self.out.append({"id": m["id"], "result": {"pipeline_runs": [{"pipeline_run_id": r} for r in ("r1", "r2", "r3")]}})
        elif t == "assist_pipeline/pipeline_debug/get":
            rid = m["pipeline_run_id"]
            self.out.append(TypeError("closed") if rid in self.fail_on else {"id": m["id"], "result": {"events": [
                {"type": "run-start", "timestamp": "2026-11-25T10:00:00+00:00", "data": {}},
                {"type": "run-end", "timestamp": "2026-11-25T10:00:01+00:00", "data": {}}]}})

    async def receive_json(self):
        r = self.out.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def test_snapshot_keeps_runs_fetched_before_the_socket_closes(monkeypatch, tmp_path):
    # Review round 4 nit: drive the real snapshot(), not a fake of it.
    sockets = [_FakeWS(fail_on={"r2"}), _FakeWS()]

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        def ws_connect(self, url, **kw):
            return sockets.pop(0)

    monkeypatch.setattr(aiohttp, "ClientSession", Session)
    out = tmp_path / "runs.jsonl"
    with pytest.raises(TypeError):
        report.collect("wss://ha/api/websocket", "t", str(out))
    assert [r["id"] for r in report.load(str(out))] == ["r1"]
    report.collect("wss://ha/api/websocket", "t", str(out))
    assert [r["id"] for r in report.load(str(out))] == ["r1", "r2", "r3"]
