"""Check a reply that claims an action against the turn's successful tool writes.

A small model sometimes answers a command with "I've turned off the lights" and no tool
call, so nothing changes. A prompt rule only lowers the rate. Home Assistant sends each
tool result back in the next request, so the proxy can see whether an action tool ran in
this turn and succeeded, and refuse a confirmation that nothing backs.

Which tools only read comes from the "reads" list in config/action-tools.json. A
successful result from any other tool backs a confirmation: the guard's "actions" list is
open (an unlisted tool passes), so it can't say what acted. The claim phrases and the
corrective line live in the same file.
"""
import json
import re

from . import guard

RERUN_NOTE = "Use a tool to do this. Never say it is done unless the tool ran."


def _text(m: dict) -> str:
    c = m.get("content")
    return c if isinstance(c, str) else " ".join(p.get("text", "") for p in c or [] if isinstance(p, dict))


def claims_action(text: str | None, cfg: dict) -> bool:
    """The reply says something was done: "I've turned", "is now on", "done", "set to".

    A reply that ends in "?" asks back, so it confirms nothing, even when it states the
    current state first ("The thermostat is set to 70. What would you like it set to?")."""
    pats = cfg.get("claim_phrases") or ()
    t = (text or "").strip().lower()
    if t.endswith("?"):
        return False
    return any(re.search(p, t) for p in pats)


def is_question(text: str | None, cfg: dict) -> bool:
    t = (text or "").strip().lower()
    words = re.findall(r"[a-z']+", t)
    return t.endswith("?") or bool(words and words[0] in cfg.get("question_words", ()))


def is_later(text: str | None, cfg: dict) -> bool:
    """A request for later ("in 30 minutes", "at 7", "tomorrow"): a tool that acts now is wrong for it."""
    t = (text or "").lower()
    return any(re.search(p, t) for p in cfg.get("later_phrases") or ())


def _failed(result: str) -> bool:
    """A tool result that reports an error, or an intent response where nothing succeeded."""
    try:
        r = json.loads(result)
    except (TypeError, ValueError):
        return False
    if not isinstance(r, dict):
        return False
    if r.get("error") or r.get("response_type") == "error" or r.get("success") is False:
        return True
    data = r.get("data")
    return isinstance(data, dict) and bool(data.get("failed")) and not data.get("success")


def turn_writes(messages: list[dict], cfg: dict) -> tuple[int, int]:
    """(landed, failed) calls to tools that are not reads, since the last user message,
    read from the history Home Assistant sends back: assistant tool_calls paired with
    role=tool results. A result with no matching call is not counted."""
    last_user = max((i for i, m in enumerate(messages or []) if m.get("role") == "user"), default=-1)
    names, landed, failed = {}, 0, 0
    for m in (messages or [])[last_user + 1:]:
        for c in m.get("tool_calls") or []:
            names[c.get("id")] = (c.get("function") or {}).get("name", "")
        name = names.get(m.get("tool_call_id"), "") if m.get("role") == "tool" else ""
        if name and not guard.is_read(name, cfg):
            if _failed(_text(m)):
                failed += 1
            else:
                landed += 1
    return landed, failed


def needs_check(messages: list[dict], cfg: dict) -> bool:
    """Whether this turn's text reply must be checked before it is spoken: the user gave a
    command (not a question), and no tool other than a read has succeeded in the turn yet."""
    if not cfg or not cfg.get("claim_phrases"):
        return False
    user = next((_text(m) for m in reversed(messages or []) if m.get("role") == "user"), "")
    if is_question(user, cfg) or not guard.request_has_command(messages, cfg):
        return False
    return turn_writes(messages, cfg)[0] == 0


def scope_tools(tools: list[dict], purpose: str, cfg: dict) -> list[dict]:
    """The tools a re-run is offered, by why it runs.

    - "act": force the commanded action; every tool stays.
    - "ground" (answer a state question with a lookup) and "later" (a request for later):
      only the tools in "reads", so the re-run can't repeat or bring forward a change,
      including through a tool that acts but is missing from "actions".
    """
    if purpose == "act":
        return list(tools or [])
    return [t for t in tools or [] if guard.is_read((t.get("function") or {}).get("name", ""), cfg)]


def rerun_body(body: dict, cfg: dict, purpose: str = "act") -> dict:
    """The request for one corrective re-run. The note goes on the end of the last user
    message, not into the system prompt, so the cached prefix is reused."""
    note = cfg.get("rerun_note") or RERUN_NOTE
    msgs = [dict(m) for m in body.get("messages") or []]
    for m in reversed(msgs):
        if m.get("role") == "user":
            m["content"] = f"{_text(m)}\n({note})"
            break
    tools = scope_tools(body.get("tools") or [], purpose, cfg)
    out = {**body, "messages": msgs, "stream": False}
    out.pop("stream_options", None)
    if tools:
        out["tools"] = tools
    else:
        out.pop("tools", None)
    return out


def allowed_calls(calls: list[dict], tools: list[dict]) -> list[dict]:
    """Drop any call to a tool the re-run was not offered."""
    offered = {(t.get("function") or {}).get("name") for t in tools or []}
    return [c for c in calls or [] if (c.get("function") or {}).get("name") in offered]
