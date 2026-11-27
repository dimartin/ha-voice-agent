"""How often the model answers a command with a confirmation and no tool call.

Replays each command N times against an OpenAI-compatible server, using a saved agent-mode
prefix (the proxy's WARM_FILE: model, system prompt, tools), first model turn only. Tool
calls are recorded, never executed. One JSON line per run; kind is tool, claim or other,
and model is what the server reports serving.

    python eval/claim_rate.py http://llm:8090 /data/warm-prefix.json [runs] > results.jsonl
"""
import json
import re
import sys
import urllib.request

CMDS = ["Turn off the den light", "Turn on the kitchen counter light", "Set den sconce two to fifty percent",
        "Turn off all the lights in the kitchen", "Set the thermostat to 68", "Pause the music", "Skip to the next song",
        "Set the volume to 30 percent", "Turn the den fan on high", "Turn on the porch light"]
AREA = "(You are in area Den and all generic commands like 'turn on the lights' should target this area.)\n"
CLAIM = re.compile(r"\b(i've|i have|turned|is now|are now|set (it|the)|done|okay,? (i|turning)|switched|paused|skipped)\b", re.I)


def run(url, prefix, text):
    body = {**prefix, "messages": prefix["messages"] + [{"role": "user", "content": AREA + text}],
            "temperature": 0.6, "top_p": 0.95, "max_tokens": 150, "stream": False}
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    out = json.load(urllib.request.urlopen(req, timeout=600))
    msg = out["choices"][0]["message"]
    tools = [c["function"]["name"] for c in msg.get("tool_calls") or []]
    text = (msg.get("content") or "").strip()
    return {"kind": "tool" if tools else "claim" if CLAIM.search(text) else "other", "text": text[:120], "tools": tools,
            "model": out.get("model")}


def main():
    url, prefix = sys.argv[1].rstrip("/"), json.load(open(sys.argv[2]))
    runs = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    for c in CMDS:
        for i in range(runs):
            print(json.dumps({"cmd": c, "i": i, **run(url, prefix, c)}), flush=True)


if __name__ == "__main__":
    main()
