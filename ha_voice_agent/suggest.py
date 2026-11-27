"""Suggest custom sentences from requests that repeat and always worked.

Every request that reaches the LLM costs seconds; a custom sentence answers the same
words in Home Assistant's own matcher in milliseconds. This reads the proxy's voice log
and prints candidate sentences for a person to review. Nothing is written into Home
Assistant.

A turn counts only when all of these hold (the rules come from Assist Learner):
- it was the first turn of a conversation;
- it made exactly one action call, every tool call in it succeeded, and the action
  resolved at least one real entity;
- the reply didn't end in a question;
- nothing it touched is safety-critical: no entity in the lock, alarm_control_panel, valve
  or cover domains, and no "lock", "alarm", "valve", "garage", "door" or "gate" in the
  request or the call's arguments.

Turns are grouped by their normalized words plus the call they produced. The words are
what speech-to-text heard (the log's "heard", when the proxy corrected a misheard name),
since that is what Home Assistant's matcher would see, not the proxy's correction. A group becomes a
candidate once it has --min-count repeats or more, unless the same words also appear in a
turn that failed, did something else, or was refused (the sentence would change what that
request does). Only built-in Hass* intents are suggested; a custom sentence can't target a
script tool such as search_media.

That last rule is the replay check issue #7 asks for, done statically. A suggested
sentence is literal (no slots in the words), so the only logged requests it can change are
those with the same normalized words, and the log already says what each of those did.
Running every logged utterance through Home Assistant's debugger with and without the
candidates would need the candidates installed first, which this tool never does.

    ha-voice-suggest /data/voice-log.jsonl [--min-count 3] [--ha wss://ha:8123/api/websocket [--insecure]]

With --ha (token in HA_TOKEN), each candidate also goes through Home Assistant's sentence
debugger, which parses without running anything; one it reports as a match (an intent or a
sentence trigger) is dropped. A closest-guess intent without a match is kept: that request
still goes to the LLM. The certificate is verified unless --insecure is given.
A candidate whose slots fix an area is marked in the output: said in one room, it would act
on that room from every satellite.
"""
import argparse
import asyncio
import json
import os
import re
import sys
from collections import defaultdict

from . import guard

SAFETY_DOMAINS = ("lock", "alarm_control_panel", "valve", "cover")
SAFETY_WORDS = re.compile(r"lock|alarm|valve|garage|door|gate", re.I)
FILLER = re.compile(r"^(please|hey|ok|okay|can you|could you|would you)\s+|\s+please$")


def normalize(text: str) -> str:
    t = re.sub(r"^\(You are in area[^)]*\)\s*", "", text or "")
    t = re.sub(r"[^\w\s']", " ", t.lower())
    t = " ".join(t.split())
    prev = None
    while prev != t:
        prev, t = t, FILLER.sub("", t).strip()
    return t


def load(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows


def turns_of(rows: list[dict]) -> list[list[dict]]:
    """One list of records per turn, in log order.

    The turn id only ties a turn's requests together: the same command said as the first
    turn of two conversations hashes to the same id. The log is written in order, so a
    request that is not after a tool result starts a new turn, and the follow-up requests
    with its id join the latest turn that has it."""
    out, open_ = [], {}
    for r in rows:
        tid = r.get("turn")
        if not tid:
            continue
        if not r.get("after_tool") or tid not in open_:
            open_[tid] = []
            out.append(open_[tid])
        open_[tid].append(r)
    return out


def judge(records: list[dict], cfg: dict) -> tuple[str | None, tuple | None, str]:
    """(normalized text, call signature, verdict). verdict is "ok" or why not."""
    first = next((r for r in records if not r.get("after_tool")), None)
    if first is None:
        return None, None, "no first request"
    said = first.get("user", "")
    text = normalize(first.get("heard") or said)
    if first.get("blocked"):
        return text, None, "refused"
    if not first.get("first_turn"):
        return text, None, "not the first turn"
    if any(r.get("failed") for r in records):
        return text, None, "failed"
    made = [c for r in records for c in r.get("calls") or []]
    actions = [c for c in made if guard.is_action(c.get("name", ""), cfg)]
    if len(actions) != 1:
        return text, None, "no action" if not actions else "several actions"
    results = [x for r in records for x in r.get("results") or []]
    if not results or not all(x.get("ok") for x in results):
        return text, None, "a tool failed"
    acted = [e for x in results if guard.is_action(x.get("name", ""), cfg) for e in x.get("entities") or []]
    if not acted:
        return text, None, "no entity resolved"
    a = actions[0]
    if any(e.split(".")[0] in SAFETY_DOMAINS for e in acted) or SAFETY_WORDS.search(json.dumps(a.get("args"))) \
            or SAFETY_WORDS.search(text) or SAFETY_WORDS.search(said):
        return text, None, "safety-critical"
    final = [r for r in records if r.get("spoken")]
    if final and final[-1]["spoken"].rstrip().endswith("?"):
        return text, None, "ended in a question"
    args = a.get("args") if isinstance(a.get("args"), dict) else {}
    return text, (a["name"].rsplit("__", 1)[-1], json.dumps(args, sort_keys=True)), "ok"


def suggest(rows: list[dict], cfg: dict, min_count: int = 3) -> tuple[list[dict], list[dict]]:
    """(candidates, dropped). Each candidate: text, intent, slots, count."""
    groups, others = defaultdict(int), defaultdict(list)
    for recs in turns_of(rows):
        text, sig, verdict = judge(recs, cfg)
        if not text:
            continue
        if verdict == "ok":
            groups[(text, sig)] += 1
        others[text].append((sig, verdict))
    out, dropped = [], []
    for (text, sig), n in sorted(groups.items(), key=lambda kv: -kv[1]):
        if n < min_count:
            continue
        intent, args = sig[0], json.loads(sig[1])
        cand = {"text": text, "intent": intent, "slots": args, "count": n}
        clash = [v for s, v in others[text] if s != sig]
        if not intent.startswith("Hass"):
            dropped.append({**cand, "why": f"{intent} is not a built-in intent"})
        elif clash:
            dropped.append({**cand, "why": "the same words also: " + ", ".join(sorted(set(
                v if v != "ok" else "did something else" for v in clash)))})
        else:
            out.append(cand)
    return out, dropped


def to_yaml(cands: list[dict]) -> str:
    """Home Assistant custom sentences (config/custom_sentences/en/<file>.yaml)."""
    lines = ["# Suggested by ha-voice-suggest. Review each one before copying it into", "# config/custom_sentences/en/.",
             "language: en", "intents:"]
    by = defaultdict(list)
    for c in cands:
        by[c["intent"]].append(c)
    for intent in sorted(by):
        lines += [f"  {intent}:", "    data:"]
        for c in by[intent]:
            lines += [f"      # seen {c['count']} times", "      - sentences:", f"          - {json.dumps(c['text'])}"]
            if "area" in c["slots"]:
                lines.append("        # area is fixed: check the request named it, not the satellite's area")
            if c["slots"]:
                lines.append("        slots:")
                lines += [f"          {k}: {json.dumps(v)}" for k, v in sorted(c["slots"].items())]
    return "\n".join(lines) + "\n"


async def already_local(url: str, token: str, texts: list[str], insecure: bool = False) -> dict[str, str]:
    """Run texts through Home Assistant's sentence debugger. Returns, for each text it reports
    as a match, what matched: the intent name, or "a sentence trigger".

    The debugger names an intent for the closest guess too, and says separately whether it
    matched ("match"), so only "match" counts."""
    from aiohttp import ClientSession
    out = {}
    async with ClientSession() as http, http.ws_connect(url, ssl=False if insecure else True) as ws:
        await ws.receive_json()
        await ws.send_json({"type": "auth", "access_token": token})
        if (await ws.receive_json()).get("type") != "auth_ok":
            raise SystemExit("Home Assistant auth failed")
        await ws.send_json({"id": 1, "type": "conversation/agent/homeassistant/debug", "sentences": texts})
        while True:
            r = await ws.receive_json()
            if r.get("id") == 1:
                break
        if not r.get("success"):
            print(f"# sentence debugger failed, nothing checked: {r.get('error')}", file=sys.stderr)
        for text, res in zip(texts, (r.get("result") or {}).get("results") or []):
            if isinstance(res, dict) and res.get("match"):
                out[text] = (res.get("intent") or {}).get("name") or "a sentence trigger"
    return out


def main(argv=None):
    p = argparse.ArgumentParser(prog="ha-voice-suggest", description=__doc__.split("\n")[0])
    p.add_argument("log")
    p.add_argument("--min-count", type=int, default=3)
    p.add_argument("--action-tools", default=guard.ACTION_TOOLS_FILE)
    p.add_argument("--ha", help="Home Assistant websocket URL; token in HA_TOKEN")
    p.add_argument("--insecure", action="store_true",
                   help="with --ha, skip certificate verification (a self-signed install)")
    a = p.parse_args(argv)
    token = os.environ.get("HA_TOKEN")
    if a.ha and not token:
        p.error("--ha needs a Home Assistant long-lived access token in HA_TOKEN")
    cands, dropped = suggest(load(a.log), guard.load(a.action_tools), a.min_count)
    if a.ha and cands:
        local = asyncio.run(already_local(a.ha, token, [c["text"] for c in cands], a.insecure))
        keep = []
        for c in cands:
            if local.get(c["text"]):
                dropped.append({**c, "why": f"already matches {local[c['text']]} locally"})
            else:
                keep.append(c)
        cands = keep
    sys.stdout.write(to_yaml(cands))
    for d in dropped:
        print(f"# not suggested ({d['count']}x) {d['text']!r}: {d['why']}", file=sys.stderr)


if __name__ == "__main__":
    main()
