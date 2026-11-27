"""Stop an action tool call when the request contains no command.

A small model can turn something that isn't a command into an action: a misheard
fragment became HassTurnOn on a cabinet light, and a warm-up prompt set the
thermostat to 75. Home Assistant runs whatever tool call it receives, so the proxy
is the last place to stop it. Which tools are actions, and which words make a
command, live in config/action-tools.json.
"""
import json
import logging
import os
import re

ACTION_TOOLS_FILE = os.environ.get("ACTION_TOOLS_FILE", "/app/config/action-tools.json")
log = logging.getLogger("voice-filter")


def load(path: str | None = None) -> dict:
    path = path or ACTION_TOOLS_FILE
    try:
        with open(path) as f:
            cfg = json.load(f)
    except (OSError, ValueError) as e:
        # No config means no guard: never block on a broken file.
        log.warning("action tools file %s unreadable (%s): action calls are not checked", path, e)
        return {}
    # Word lists are matched lowercased. Keys ending in _phrases hold regular expressions,
    # kept as written (lowercasing would turn \W into \w); a string such as rerun_note stays a string.
    out = {}
    for k, v in cfg.items():
        if k.startswith("_"):
            continue
        if isinstance(v, list):
            out[k] = tuple(v) if k.endswith("_phrases") else frozenset(w.lower() for w in v)
        elif isinstance(v, str):
            out[k] = v
    return out


def is_action(tool_name: str, cfg: dict) -> bool:
    return (tool_name or "").rsplit("__", 1)[-1].lower() in cfg.get("actions", ()) if cfg else False


def is_read(tool_name: str, cfg: dict) -> bool:
    """A tool that only looks something up (the "reads" list). Used by the claim check,
    never by the guard."""
    return (tool_name or "").rsplit("__", 1)[-1].lower() in cfg.get("reads", ()) if cfg else False


def has_command(text: str | None, cfg: dict) -> bool:
    words = re.findall(r"[a-z']+|\d+", (text or "").lower())
    if any(w in cfg.get("command_words", ()) for w in words):
        return True
    # Verbless requests are commands unless they are questions: "kitchen lights off",
    # "volume 30", "ten minute timer", "thermostat to 72", but not "is the light on" or
    # "how long is left on the timer".
    if not words or words[0] in cfg.get("question_words", ()):
        return False
    return any(w.isdigit() or w in cfg.get("particles", ()) or w in cfg.get("command_nouns", ()) for w in words)


def _text(m: dict) -> str:
    c = m.get("content")
    return c if isinstance(c, str) else " ".join(p.get("text", "") for p in c or [] if isinstance(p, dict))


def request_has_command(messages: list[dict], cfg: dict) -> bool:
    """A command in the last user turn, or in the turn before it when the last one only
    answers the assistant's question ("Which room?" "The kitchen.") or extends the
    previous request ("and the bedroom too")."""
    turns = [(m.get("role"), _text(m)) for m in messages or [] if m.get("role") in ("user", "assistant")]
    users = [i for i, (r, _) in enumerate(turns) if r == "user"]
    if not users:
        return False
    last = turns[users[-1]][1]
    if has_command(last, cfg):
        return True
    if len(users) < 2:
        return False
    before = turns[users[-1] - 1] if users[-1] > 0 else (None, "")
    answers_question = before[0] == "assistant" and before[1].rstrip().endswith("?")
    extends = bool(re.match(r"\s*(and|also|plus)\b", last, re.I) or re.search(r"\b(too|as well)\W*$", last, re.I))
    return (answers_question or extends) and has_command(turns[users[-2]][1], cfg)


def blocked(messages: list[dict], tool_names: list[str], cfg: dict) -> list[str]:
    """The action tools to refuse: all of them when the request holds no command."""
    actions = [t for t in tool_names if is_action(t, cfg)]
    return actions if actions and not request_has_command(messages, cfg) else []
