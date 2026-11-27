"""Rewrite Home Assistant's LLM request into a small, cache-stable one.

HA sends a ~7.7k-token request: its own system prompt, a verbose YAML list of every
exposed device, and 30+ tool schemas with long descriptions. This keeps what the model
needs and drops the rest:

- the system prompt becomes our voice rules (AGENT_PROMPT_FILE), a compact device list
  grouped by area, and HA's own control line;
- tool schemas keep their names, parameter names, types, enums and required lists (HA
  validates and runs the calls), but lose long descriptions and parameter prose;
- HA's satellite-area line moves to the start of the last user turn, so the system
  prompt and tools are the same for every satellite and stay in the server's cache;
- every other non-system message (tool calls, tool results) passes through as is.

Everything derived here is a pure function of HA's request, so the prefix stays byte-stable
between requests and the server's prompt cache keeps working.
"""
import copy
import json
import logging
import os
import re

log = logging.getLogger("voice-filter.agent")

STATIC_HEADER = "Static Context:"
AREA_PREFIX = "You are in area"
KEEP_LINE_PREFIXES = (AREA_PREFIX, "When controlling Home Assistant")

# Replacement descriptions by tool name. HA's own are either long usage essays
# (GetLiveContext) or empty boilerplate for intent scripts ("Execute ... intent").
TOOL_DESCRIPTIONS_FILE = os.environ.get("TOOL_DESCRIPTIONS_FILE", "/app/config/tool-descriptions.json")
MAX_DESC = 110
NO_AREA = "Other devices (no area)"

# Above this many devices the system prompt lists areas with device counts instead of
# every device, the satellite's own area goes in full with the user's turn, and the model
# looks up other areas with LIST_TOOL, which the proxy answers itself. 0 keeps the full
# list always. On a CPU the full list is cheaper per request, because it sits in the
# cached prefix (see the README's measurements).
LAZY_DEVICES_ABOVE = int(os.environ.get("LAZY_DEVICES_ABOVE", "0"))
LIST_TOOL_NAME = "ListDevices"
LIST_TOOL = {"type": "function", "function": {
    "name": LIST_TOOL_NAME,
    "description": "List the devices in one area of this home, by name and type.",
    "parameters": {"type": "object", "properties": {"area": {"type": "string"}}, "required": ["area"]}}}


def load_tool_descriptions(path: str = TOOL_DESCRIPTIONS_FILE) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        log.debug("agent: no tool description file at %s", path)
        return {}


def parse_static_context(system: str) -> list[dict]:
    """Parse HA's 'Static Context' YAML-ish device list into [{names, domain, areas}]."""
    i = system.find(STATIC_HEADER)
    if i < 0:
        return []
    devices, cur = [], None
    for line in system[i:].splitlines()[1:]:
        if line.startswith("- names:"):
            cur = {"names": line[len("- names:"):].strip(), "domain": "", "areas": ""}
            devices.append(cur)
        elif cur is not None and line.startswith("  domain:"):
            cur["domain"] = line.split(":", 1)[1].strip()
        elif cur is not None and line.startswith("  areas:"):
            cur["areas"] = line.split(":", 1)[1].strip()
        elif cur is not None and line.startswith("  "):
            continue
        elif devices:
            break
    return devices


def compact_devices(devices: list[dict]) -> str:
    by_area: dict[str, list[str]] = {}
    for d in devices:
        names = [n.strip() for n in d["names"].split(",") if n.strip()]
        label = names[0] + (f" (also {', '.join(names[1:])})" if len(names) > 1 else "")
        by_area.setdefault(d["areas"] or NO_AREA, []).append(f"{label} [{d['domain']}]")
    order = sorted(a for a in by_area if a != NO_AREA) + ([NO_AREA] if NO_AREA in by_area else [])
    return "\n".join(f"{a}: {'; '.join(by_area[a])}" for a in order)


def kept_lines(system: str) -> list[str]:
    return [l.strip() for l in system.splitlines() if l.strip().startswith(KEEP_LINE_PREFIXES)]


def lazy(devices: list[dict], above: int | None = None) -> bool:
    above = LAZY_DEVICES_ABOVE if above is None else above
    return bool(above) and len(devices) > above


def area_counts(devices: list[dict]) -> str:
    counts: dict[str, int] = {}
    for d in devices:
        counts[d["areas"] or NO_AREA] = counts.get(d["areas"] or NO_AREA, 0) + 1
    order = sorted(a for a in counts if a != NO_AREA) + ([NO_AREA] if NO_AREA in counts else [])
    return "; ".join(f"{a} ({counts[a]})" for a in order)


def satellite_area(system: str) -> str | None:
    m = re.match(re.escape(AREA_PREFIX) + r" (.+?)(?: \(floor [^)]*\))? and all generic commands", area_line(system) or "")
    return m.group(1).strip() if m else None


def list_devices(area: str, devices: list[dict]) -> str:
    """The answer to LIST_TOOL: one area's devices, matched without regard to case or spaces."""
    key = re.sub(r"\W", "", (area or "").lower())
    hits = [d for d in devices if re.sub(r"\W", "", (d["areas"] or "").lower()) == key]
    if not hits:
        return f"No area called {area!r}. Areas: {area_counts(devices)}"
    return compact_devices(hits)


def build_system(system: str, voice_rules: str, above: int | None = None) -> str:
    parts = [voice_rules.strip()]
    devices = parse_static_context(system)
    if devices and lazy(devices, above):
        parts.append("Areas in this home (number of devices): " + area_counts(devices) + ". The devices in the "
                     f"user's own area come with their message; call {LIST_TOOL_NAME} for any other area.")
    elif devices:
        parts.append("Devices in this home, by area (name [type]):\n" + compact_devices(devices))
    parts.append("Use tools only for requests about this home: its devices, weather, timers and lists. "
                 "Answer cooking and general knowledge questions directly, without tools.")
    parts += [l for l in kept_lines(system) if not l.startswith(AREA_PREFIX)]
    return "\n\n".join(parts)


def area_line(system: str) -> str | None:
    return next((l for l in kept_lines(system) if l.startswith(AREA_PREFIX)), None)


def _short(text: str) -> str:
    text = " ".join((text or "").split())
    first = re.split(r"(?<=\.)\s", text, maxsplit=1)[0]
    return first if len(first) <= MAX_DESC else first[: MAX_DESC - 1].rsplit(" ", 1)[0] + "…"


def _slim_schema(schema):
    if isinstance(schema, dict):
        return {k: _slim_schema(v) for k, v in schema.items() if k not in ("description", "additionalProperties", "title")}
    if isinstance(schema, list):
        return [_slim_schema(v) for v in schema]
    return schema


def _drop_optional(params: dict, drop: frozenset) -> dict:
    """Remove optional parameters named in `drop`; required ones always stay."""
    required = set(params.get("required", []))
    props = params.get("properties")
    if isinstance(props, dict):
        params["properties"] = {k: v for k, v in props.items() if k in required or k not in drop}
    return params


def compact_tools(tools: list[dict], drop_params: frozenset = frozenset(), drop_tools: frozenset = frozenset(),
                  descriptions: dict | None = None) -> list[dict]:
    descriptions = descriptions or {}
    out = []
    for t in tools:
        t = copy.deepcopy(t)
        f = t.get("function", {})
        name = f.get("name", "")
        if name in drop_tools:
            continue
        f["description"] = descriptions.get(name) or _short(f.get("description", ""))
        if "parameters" in f:
            f["parameters"] = _drop_optional(_slim_schema(f["parameters"]), drop_params)
        out.append(t)
    return out


def rewrite(body: dict, voice_rules: str, drop_params: frozenset = frozenset(), drop_tools: frozenset = frozenset(),
            descriptions: dict | None = None, above: int | None = None) -> dict:
    """Return a new request body; the input is not modified."""
    msgs = body.get("messages") or []
    if not msgs or msgs[0].get("role") != "system" or not isinstance(msgs[0].get("content"), str):
        log.debug("agent: no leading text system message, passing request through")
        return body
    new = dict(body)
    devices = parse_static_context(msgs[0]["content"])
    short = lazy(devices, above)
    new["messages"] = [{"role": "system", "content": build_system(msgs[0]["content"], voice_rules, above)}] + msgs[1:]
    # The satellite's area differs per satellite, so it goes in the last user turn, not
    # the system prompt: chat templates render the tools after the system text, and a
    # per-satellite line there would make every satellite re-read every tool.
    area = area_line(msgs[0]["content"])
    users = [i for i, m in enumerate(new["messages"]) if m.get("role") == "user" and isinstance(m.get("content"), str)]
    if area and users:
        i = users[-1]
        here = ""
        own = satellite_area(msgs[0]["content"])
        if short and own:
            here = f"(Devices here: {list_devices(own, devices)})\n"
        new["messages"][i] = {**new["messages"][i], "content": f"({area})\n{here}{new['messages'][i]['content']}"}
    if body.get("tools"):
        new["tools"] = compact_tools(body["tools"], drop_params, drop_tools, descriptions)
        if short:
            new["tools"].append(copy.deepcopy(LIST_TOOL))  # always last and always the same: the prefix stays stable
    return new
