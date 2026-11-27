"""Replay transcripts through Home Assistant's sentence debugger before and after the STT
filter, and report what changed. Parses only; nothing runs. Exits 1 if any transcript
that matched before stops matching, or matches a different intent or target.

    HA_TOKEN=... python eval/stt_replay.py wss://<host>:8123/api/websocket names.json transcripts.txt
"""
import asyncio
import json
import sys

from aiohttp import ClientSession

sys.path.insert(0, ".")
from ha_voice_agent.stt_filter import Names, normalize  # noqa: E402


async def debug(url, token, sentences):
    async with ClientSession() as http, http.ws_connect(url, ssl=False) as ws:
        await ws.receive_json()
        await ws.send_json({"type": "auth", "access_token": token})
        await ws.receive_json()
        await ws.send_json({"id": 1, "type": "conversation/agent/homeassistant/debug", "sentences": sentences})
        while True:
            r = await ws.receive_json()
            if r.get("id") == 1:
                return r["result"]["results"]


def key(res):
    # The debugger names a closest-guess intent even without a match; only "match" counts.
    if not res or res.get("match") is False:
        return None
    targets = res.get("targets")
    targets = sorted(targets) if isinstance(targets, dict) else targets
    # A sentence trigger match has no "intent" key; it counts as matched, by its template.
    name = (res.get("intent") or {}).get("name") or f"trigger: {res.get('sentence_template')}"
    return (name, json.dumps(targets), bool(res.get("unmatched_slots")))


def main():
    url, names_path, path = sys.argv[1:4]
    names = Names(names_path)
    names.refresh()
    texts = [l.strip() for l in open(path) if l.strip()]
    after = [normalize(t, names) for t in texts]
    import os
    token = os.environ["HA_TOKEN"]
    a, b = asyncio.run(debug(url, token, texts)), asyncio.run(debug(url, token, after))
    broke = fixed = changed = 0
    for t, n, x, y in zip(texts, after, a, b):
        kx, ky = key(x), key(y)
        # Matched means an intent with no unmatched slots; intents with no entity targets
        # (timers, time, weather) count too, so a rewrite that breaks one shows as BROKE.
        ok_x, ok_y = bool(kx and not kx[2]), bool(ky and not ky[2])
        if t == n:
            continue
        changed += 1
        status = "fixed" if ok_y and not ok_x else "BROKE" if ok_x and (not ok_y or kx != ky) else "same"
        broke += status == "BROKE"
        fixed += status == "fixed"
        print(json.dumps({"status": status, "heard": t, "sent": n, "before": kx, "after": ky}))
    print(json.dumps({"transcripts": len(texts), "rewritten": changed, "fixed": fixed, "broke": broke}))
    sys.exit(1 if broke else 0)


if __name__ == "__main__":
    main()
