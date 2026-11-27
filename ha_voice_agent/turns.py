"""What the voice log needs to know about a turn beyond its text.

A voice turn can take several requests: the model calls a tool, Home Assistant runs it
and sends the result back in the next request, and the model answers. Each request is
one voice log line. These fields tie them together and record what the tools did, so
the log can later say which requests repeat and always succeed (suggest.py).
"""
import hashlib
import json


def _text(m: dict) -> str:
    c = m.get("content")
    return c if isinstance(c, str) else " ".join(p.get("text", "") for p in c or [] if isinstance(p, dict))


def _last_user(messages: list[dict]) -> int:
    return max((i for i, m in enumerate(messages or []) if m.get("role") == "user"), default=-1)


def fields(messages: list[dict]) -> dict:
    """turn: the same id for every request of one turn (a hash of the conversation up to
    and including the user's message); first_turn: no earlier user message;
    results: what each tool returned since the user's message, for requests after a tool."""
    i = _last_user(messages)
    upto = [{"role": m.get("role"), "content": _text(m)} for m in (messages or [])[: i + 1] if m.get("role") != "system"]
    out = {"turn": hashlib.sha1(json.dumps(upto, sort_keys=True).encode()).hexdigest()[:12],
           "first_turn": sum(1 for m in upto if m["role"] == "user") == 1}
    names, results = {}, []
    for m in (messages or [])[i + 1:]:
        for c in m.get("tool_calls") or []:
            names[c.get("id")] = (c.get("function") or {}).get("name", "")
        if m.get("role") == "tool":
            results.append({"name": names.get(m.get("tool_call_id"), ""), **result(_text(m))})
    if results:
        out["results"] = results
    return out


def result(text: str) -> dict:
    """ok: no error and, for an intent response, at least one target succeeded;
    entities: the entity ids it acted on, when the response names them."""
    try:
        r = json.loads(text)
    except (TypeError, ValueError):
        return {"ok": True, "entities": []}
    if not isinstance(r, dict):
        return {"ok": True, "entities": []}
    data = r.get("data") if isinstance(r.get("data"), dict) else {}
    ents = [t.get("id") for t in data.get("success") or [] if isinstance(t, dict) and t.get("id")]
    failed = bool(r.get("error")) or r.get("response_type") == "error" or r.get("success") is False \
        or (bool(data.get("failed")) and not data.get("success"))
    return {"ok": not failed, "entities": ents}


def calls(tool_calls) -> list[dict]:
    """[{name, args}] from {index: {name, arguments}} (streamed) or OpenAI tool_calls."""
    items = tool_calls.values() if isinstance(tool_calls, dict) else [
        {"name": (c.get("function") or {}).get("name", ""), "arguments": (c.get("function") or {}).get("arguments", "")}
        for c in tool_calls or []]
    out = []
    for c in items:
        try:
            args = json.loads(c.get("arguments") or "{}")
        except ValueError:
            args = c.get("arguments")
        out.append({"name": c.get("name", ""), "args": args})
    return out
