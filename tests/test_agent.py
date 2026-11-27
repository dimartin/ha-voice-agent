import json
from pathlib import Path

from ha_voice_agent import agent

HERE = Path(__file__).parent
REQ = json.loads((HERE / "fixtures" / "ha_request.json").read_text())
RULES = (HERE.parent / "config" / "agent-prompt.txt").read_text()


def test_parses_every_device_in_static_context():
    devices = agent.parse_static_context(REQ["messages"][0]["content"])
    assert len(devices) == REQ["messages"][0]["content"].count("- names:")
    spk = [d for d in devices if d["names"].startswith("Office Speaker")][0]
    assert spk["domain"] == "media_player" and spk["areas"] == "Office"


def test_aliases_become_one_label():
    out = agent.compact_devices([{"names": "Office Speaker, office dot, dot", "domain": "media_player", "areas": "Office"}])
    assert out == "Office: Office Speaker (also office dot, dot) [media_player]"


def test_devices_without_area_are_listed_last():
    out = agent.compact_devices([{"names": "Shopping List", "domain": "todo", "areas": ""},
                                 {"names": "Porch Light", "domain": "light", "areas": "Porch"}])
    assert out.splitlines() == ["Porch: Porch Light [light]", "Other devices (no area): Shopping List [todo]"]


def test_rewrite_keeps_satellite_area_and_drops_ha_prompt():
    new = agent.rewrite(REQ, RULES)
    system = new["messages"][0]["content"]
    # The area is per satellite, so it rides in the last user turn, not the cached prefix.
    assert "You are in area" not in system
    user = [m for m in new["messages"] if m["role"] == "user"][-1]["content"]
    assert user.startswith("(You are in area Kitchen")
    assert user.endswith(REQ["messages"][-1]["content"]) or REQ["messages"][-1]["role"] != "user"
    assert "Static Context" not in system and "You ARE equipped" not in system
    assert system.startswith(RULES.strip().splitlines()[0])


DROP_P = frozenset({"floor", "device_class", "media_class"})
# The deployed DROP_TOOLS list (compose.yaml): edits HA's local intents already cover by exact phrase.
DROP_T = frozenset({"intent__HassIncreaseTimer", "intent__HassDecreaseTimer", "intent__HassPauseTimer",
                    "intent__HassUnpauseTimer", "intent__HassCancelAllTimers", "intent__HassStopMoving",
                    "intent__HassSetPosition", "media_player__HassMediaPlayerMute", "media_player__HassMediaPlayerUnmute",
                    "media_player__HassMediaPrevious", "media_player__HassSetVolumeRelative",
                    "todo__HassListCompleteItem", "todo__HassListRemoveItem"})


def test_dropped_params_go_but_required_ones_stay():
    tools = agent.compact_tools(REQ["tools"], DROP_P)
    for t in tools:
        p = t["function"]["parameters"]
        assert not (DROP_P - set(p.get("required", []))) & set(p.get("properties", {}))
        assert set(p.get("required", [])) <= set(p.get("properties", {}))


def test_dropped_tools_are_removed():
    tools = agent.compact_tools(REQ["tools"], drop_tools=frozenset({"intent__HassPauseTimer"}))
    assert "intent__HassPauseTimer" not in [t["function"]["name"] for t in tools]
    assert len(tools) == len(REQ["tools"]) - 1


def test_tools_keep_names_params_and_required():
    new = agent.rewrite(REQ, RULES)
    for old, slim in zip(REQ["tools"], new["tools"]):
        fo, fs = old["function"], slim["function"]
        assert fo["name"] == fs["name"]
        assert set(fo["parameters"].get("properties", {})) == set(fs["parameters"].get("properties", {}))
        assert fo["parameters"].get("required") == fs["parameters"].get("required")
        assert "description" not in json.dumps(fs["parameters"])


def test_request_shrinks_by_more_than_half():
    before = len(json.dumps(REQ["messages"][0])) + len(json.dumps(REQ["tools"]))
    new = agent.rewrite(REQ, RULES, DROP_P, DROP_T)
    after = len(json.dumps(new["messages"][0])) + len(json.dumps(new["tools"]))
    assert after < before / 2, (before, after)


def test_rewrite_is_deterministic_and_does_not_mutate_input():
    snapshot = json.dumps(REQ)
    a, b = agent.rewrite(REQ, RULES), agent.rewrite(REQ, RULES)
    assert json.dumps(a) == json.dumps(b)
    assert json.dumps(REQ) == snapshot


def test_non_system_messages_pass_through():
    # Everything after the system prompt is unchanged, except that the last user turn
    # gains the satellite's area line at its start.
    req = dict(REQ, messages=REQ["messages"] + [{"role": "assistant", "content": None,
                                                   "tool_calls": [{"id": "x", "type": "function", "function": {"name": "a", "arguments": "{}"}}]},
                                                  {"role": "tool", "tool_call_id": "x", "content": "{}"}])
    out = agent.rewrite(req, RULES)["messages"][1:]
    assert out[1:] == req["messages"][2:]
    assert out[0]["content"] == "(" + agent.area_line(REQ["messages"][0]["content"]) + ")\n" + req["messages"][1]["content"]


def test_system_prompt_and_tools_are_the_same_for_every_satellite():
    kitchen = agent.rewrite(REQ, RULES)
    sys2 = REQ["messages"][0]["content"].replace("You are in area Kitchen (floor Ground Floor)", "You are in area Office")
    office = agent.rewrite(dict(REQ, messages=[{"role": "system", "content": sys2}] + REQ["messages"][1:]), RULES)
    assert kitchen["messages"][0] == office["messages"][0] and kitchen["tools"] == office["tools"]
    assert kitchen["messages"][-1] != office["messages"][-1]


def test_request_without_system_passes_through_unchanged():
    req = {"messages": [{"role": "user", "content": "hi"}]}
    assert agent.rewrite(req, RULES) is req


def test_description_overrides_apply_by_name():
    tools = agent.compact_tools(REQ["tools"], descriptions={"intent_script__WeatherForecast": "Check for rain."})
    rc = [t for t in tools if t["function"]["name"] == "intent_script__WeatherForecast"][0]
    assert rc["function"]["description"] == "Check for rain."


def test_shipped_description_file_names_real_tools():
    names = {t["function"]["name"] for t in REQ["tools"]}
    shipped = agent.load_tool_descriptions(str(HERE.parent / "config" / "tool-descriptions.json"))
    assert shipped and set(shipped) <= names
