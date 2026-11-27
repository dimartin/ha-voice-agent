"""Report repeated and failed voice requests across the whole pipeline.

The proxy only sees requests that reached the LLM. Home Assistant sees every request,
but keeps only the last 10 runs per pipeline, in memory, until it restarts
(assist_pipeline/run.py, STORED_PIPELINE_RUNS). So there are two commands:

    ha-voice-report collect --ha wss://<host>:8123/api/websocket --out runs.jsonl [--every 60]

snapshots Home Assistant's pipeline runs into a file (token in HA_TOKEN; it only reads),
skipping runs already in it. Run it with --every so no run is missed between snapshots.

    ha-voice-report report --voice-log voice-log.jsonl [--runs runs.jsonl]
                           [--satellite-log turns.jsonl [--ts-key ts] [--text-key text]
                            [--device-key satellite]] [--window 60] [--min-gap 3]
                           [--action-tools <file>] [--json]

prints three lists:
- requests repeated on the same device within --window seconds and at least --min-gap
  seconds apart (the first answer probably didn't help). The device is Home Assistant's
  device id from the runs, or a satellite row's --device-key. The voice log names no
  device, so with the voice log alone repeats are not split by satellite;
- requests that reached the LLM and ended in an error, a question back, or a command with
  no tool call (command words from --action-tools, default ACTION_TOOLS_FILE);
- the most frequent requests overall, as input to ha-voice-suggest.

--satellite-log adds what a satellite itself recorded (one JSON object per line; field
names set with --ts-key, --text-key and --device-key). With --runs, a row a pipeline run
accounts for within 5 seconds is the same request and counts once; rows no run accounts
for are counted and listed too: the satellite heard something that never became a run.
"""
import argparse
import asyncio
import difflib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime

from . import guard
from .suggest import load, normalize

SIMILAR = 0.8
# A person can't ask again sooner than this: the reply alone takes seconds. Closer pairs
# are scripted replays (benchmarks, tests) and the proxy's own warm-ups.
MIN_GAP = 3.0
SKIP = {"warmup", "hi"}


def epoch(ts) -> float | None:
    if isinstance(ts, (int, float)):
        return float(ts)
    if not ts:
        return None
    s = str(ts).replace("Z", "+00:00")
    if len(s) > 5 and s[-5] in "+-" and s[-3] != ":":
        s = s[:-2] + ":" + s[-2:]  # +0000 -> +00:00, for older Pythons
    try:
        return datetime.fromisoformat(s).timestamp()
    except ValueError:
        return None


# --- collect ---

def summarize(run_id: str, events: list[dict]) -> dict:
    """One line per pipeline run, from its debug events."""
    out = {"id": run_id, "tools": [], "tool_errors": 0}
    for e in events:
        t, d = e.get("type"), e.get("data") or {}
        if t == "run-start":
            out["ts"] = e.get("timestamp")
            out["conversation_id"] = d.get("conversation_id")
        elif t == "stt-end":
            out["heard"] = (d.get("stt_output") or {}).get("text")
        elif t == "intent-start":
            out.setdefault("heard", d.get("intent_input"))
            out["device"] = d.get("device_id") or d.get("satellite_id")
            out["agent"] = d.get("engine")
        elif t == "intent-progress":
            delta = d.get("chat_log_delta") or {}
            out["tools"] += [c.get("tool_name") for c in delta.get("tool_calls") or []]
            res = delta.get("result")
            if delta.get("role") == "tool_result" and isinstance(res, dict) and (res.get("error") or res.get("success") is False):
                out["tool_errors"] += 1
        elif t == "intent-end":
            out["local"] = d.get("processed_locally")
            resp = (d.get("intent_output") or {}).get("response") or {}
            out["response_type"] = resp.get("response_type")
            out["speech"] = ((resp.get("speech") or {}).get("plain") or {}).get("speech", "").strip()
        elif t == "error":
            out["error"] = d.get("code") or d.get("message")
    return out


async def snapshot(url: str, token: str, seen: set, new: list | None = None) -> list[dict]:
    """New finished runs, appended to `new` as each is fetched, so a caller still has the
    ones gathered before a failure (their ids are already in `seen`)."""
    from aiohttp import ClientSession
    n, new = 0, [] if new is None else new
    async with ClientSession() as http, http.ws_connect(url, ssl=False) as ws:
        await ws.receive_json()
        await ws.send_json({"type": "auth", "access_token": token})
        if (await ws.receive_json()).get("type") != "auth_ok":
            raise SystemExit("Home Assistant auth failed")

        async def call(msg):
            nonlocal n
            n += 1
            await ws.send_json({"id": n, **msg})
            while True:
                r = await ws.receive_json()
                if r.get("id") == n:
                    return r.get("result") or {}

        for p in (await call({"type": "assist_pipeline/pipeline/list"})).get("pipelines") or []:
            runs = (await call({"type": "assist_pipeline/pipeline_debug/list", "pipeline_id": p["id"]})).get("pipeline_runs") or []
            for run in runs:
                rid = run["pipeline_run_id"]
                if rid in seen:
                    continue
                got = await call({"type": "assist_pipeline/pipeline_debug/get", "pipeline_id": p["id"], "pipeline_run_id": rid})
                events = got.get("events") or []
                if not any(e.get("type") == "run-end" for e in events):
                    continue  # still running; the next snapshot gets it whole
                new.append({**summarize(rid, events), "pipeline": p.get("name")})
                seen.add(rid)
    return new


def _write(out: str, rows: list[dict]) -> None:
    with open(out, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def collect(url: str, token: str, out: str, every: float = 0) -> None:
    seen = {r.get("id") for r in load(out)} if os.path.exists(out) else set()
    while True:
        new = []
        try:
            asyncio.run(snapshot(url, token, seen, new))
        except Exception as e:  # noqa: BLE001
            # A reverse proxy's 502 while HA restarts (WSServerHandshakeError), HA closing
            # the socket mid-snapshot (WSMessageTypeError, a TypeError), a dropped
            # connection: all pass, and the next snapshot tries again. A failed auth is a
            # SystemExit and still ends the loop.
            # The runs fetched before the failure are written below: their ids are in
            # `seen`, so no later snapshot would fetch them again.
            print(f"collect: {type(e).__name__}: {e}", file=sys.stderr)
            if not every:
                _write(out, new)
                raise  # one snapshot: a failure is the command's exit status
        _write(out, new)
        if new:
            print(f"collect: {len(new)} new runs", file=sys.stderr)
        if not every:
            return
        time.sleep(every)


# --- report ---

def llm_turns(voice: list[dict]) -> list[list[dict]]:
    """The voice log's records grouped into turns, in log order. Each first request starts
    a turn. An after-tool request joins the latest turn with its `turn` id, or, with no id
    logged, the turn just before it. The same words opening a new conversation hash to the
    same `turn` id, so an id alone never merges two first requests."""
    out, latest, cur = [], {}, None
    for r in voice:
        tid = r.get("turn")
        if r.get("after_tool"):
            g = latest.get(tid) if tid else cur
            if g is not None:
                g.append(r)
                continue
        cur = [r]
        out.append(cur)
        if tid:
            latest[tid] = cur
    return out


def llm_problems(voice: list[dict], cfg: dict) -> list[dict]:
    out = []
    for recs in llm_turns(voice):
        first = recs[0]
        if normalize(first.get("user")) in SKIP or first.get("local") or first.get("s") == 0:
            continue  # warm-ups, and turns the proxy answered itself (cancel, units)
        last = [r for r in recs if r.get("spoken")] or recs
        spoken = (last[-1].get("spoken") or "").strip()
        made = [t for r in recs for t in r.get("tools") or []]
        if any(r.get("failed") for r in recs):
            why = "error: " + next(r["failed"] for r in recs if r.get("failed"))
        elif spoken.endswith("?"):
            why = "asked back"
        elif not made and not question(first.get("user"), cfg) and not first.get("local") and not first.get("blocked") \
                and guard.request_has_command([{"role": "user", "content": first.get("user") or ""}], cfg):
            why = "command, no tool call"
        else:
            continue
        out.append({"ts": first.get("ts"), "text": first.get("user"), "spoken": spoken, "why": why})
    return out


def question(text, cfg) -> bool:
    t = (text or "").strip()
    words = normalize(t).split()
    return t.endswith("?") or bool(words and words[0] in cfg.get("question_words", ()))


def requests_of(runs: list[dict], voice: list[dict], sat: list[dict]) -> list[dict]:
    """Every request with a time, a device and what was heard, each counted once. Pipeline
    runs when there are any, else the voice log's first requests. Satellite rows are added
    as well, except, with runs, those a run already accounts for."""
    if runs:
        reqs = [{"t": epoch(r.get("ts")), "device": r.get("device"), "text": r.get("heard") or "", "src": "run",
                 "local": r.get("local"), "speech": r.get("speech")} for r in runs]
    else:
        reqs = [{"t": epoch(r.get("ts")), "device": None, "text": r.get("user") or "", "src": "llm"}
                for r in voice if not r.get("after_tool")]
    if runs:
        sat = unmatched_satellite(sat, runs)
    return [r for r in reqs + sat if r["t"] is not None]


def repeats(reqs: list[dict], window: float, min_gap: float = MIN_GAP) -> list[dict]:
    out, by = [], defaultdict(list)
    for r in reqs:
        if normalize(r["text"]) not in SKIP:
            by[r.get("device")].append(r)
    for dev, rs in by.items():
        rs.sort(key=lambda r: r["t"])
        for a, b in zip(rs, rs[1:]):
            na, nb = normalize(a["text"]), normalize(b["text"])
            if na and nb and min_gap <= b["t"] - a["t"] <= window and difflib.SequenceMatcher(None, na, nb).ratio() >= SIMILAR:
                out.append({"device": dev, "first": a["text"], "again": b["text"], "after_s": round(b["t"] - a["t"], 1),
                            "first_reply": a.get("speech")})
    return out


def unmatched_satellite(sat: list[dict], runs: list[dict], slack: float = 5) -> list[dict]:
    times = sorted(epoch(r.get("ts")) for r in runs if epoch(r.get("ts")) is not None)
    out = []
    for s in sat:
        if not any(abs(s["t"] - t) <= slack for t in times):
            out.append(s)
    return out


def read_satellite(path: str, ts_key: str, text_key: str, device_key: str) -> list[dict]:
    rows = []
    for r in load(path):
        rows.append({"t": epoch(r.get(ts_key)), "text": r.get(text_key) or "", "device": r.get(device_key), "src": "satellite"})
    return [r for r in rows if r["t"] is not None]


def build(voice: list[dict], runs: list[dict], sat: list[dict], cfg: dict, window: float = 60, top: int = 20,
          min_gap: float = MIN_GAP) -> dict:
    reqs = requests_of(runs, voice, sat)
    freq = Counter(normalize(r["text"]) for r in reqs if normalize(r["text"]) and normalize(r["text"]) not in SKIP)
    return {
        "requests": len(reqs),
        "repeated": repeats(reqs, window, min_gap),
        "llm_problems": llm_problems(voice, cfg),
        "frequent": [{"text": t, "count": n} for t, n in freq.most_common(top)],
        "satellite_only": [{"ts": r["t"], "text": r["text"], "device": r.get("device")} for r in unmatched_satellite(sat, runs)]
        if sat and runs else [],
    }


def to_text(rep: dict) -> str:
    lines = [f"{rep['requests']} requests", "", f"Repeated within the window ({len(rep['repeated'])}):"]
    lines += [f"  {r['after_s']:>5}s  {r['first']!r} -> {r['again']!r}" + (f"  (first reply: {r['first_reply']!r})" if r.get("first_reply") else "")
              for r in rep["repeated"]] or ["  none"]
    lines += ["", f"Reached the LLM and ended badly ({len(rep['llm_problems'])}):"]
    lines += [f"  {p['why']:<22} {p['text']!r} -> {p['spoken']!r}" for p in rep["llm_problems"]] or ["  none"]
    lines += ["", "Most frequent:"] + [f"  {f['count']:>4}  {f['text']}" for f in rep["frequent"]]
    if rep["satellite_only"]:
        lines += ["", f"Heard by a satellite but never a pipeline run ({len(rep['satellite_only'])}):"]
        lines += [f"  {s['text']!r} ({s['device']})" for s in rep["satellite_only"]]
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(prog="ha-voice-report", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect", help="snapshot Home Assistant's pipeline runs into a file")
    c.add_argument("--ha", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--every", type=float, default=0, help="seconds between snapshots; 0 runs once")
    r = sub.add_parser("report", help="print the report")
    r.add_argument("--voice-log", required=True)
    r.add_argument("--runs")
    r.add_argument("--satellite-log")
    r.add_argument("--ts-key", default="ts")
    r.add_argument("--text-key", default="text")
    r.add_argument("--device-key", default="satellite")
    r.add_argument("--window", type=float, default=60)
    r.add_argument("--min-gap", type=float, default=MIN_GAP, help="closer repeats are scripted replays")
    r.add_argument("--action-tools", default=guard.ACTION_TOOLS_FILE,
                   help="action tools and command words, as for the proxy (ACTION_TOOLS_FILE)")
    r.add_argument("--json", action="store_true")
    a = p.parse_args(argv)
    if a.cmd == "collect":
        collect(a.ha, os.environ["HA_TOKEN"], a.out, a.every)
        return
    sat = read_satellite(a.satellite_log, a.ts_key, a.text_key, a.device_key) if a.satellite_log else []
    rep = build(load(a.voice_log), load(a.runs) if a.runs else [], sat, guard.load(a.action_tools), a.window, min_gap=a.min_gap)
    sys.stdout.write(json.dumps(rep, indent=2) + "\n" if a.json else to_text(rep))


if __name__ == "__main__":
    main()
