"""Issue 2: a misheard device name is corrected against Home Assistant's own list."""
import json
from pathlib import Path

from aiohttp import web

from ha_voice_agent import agent, app, names
from test_turn_reliability import run

HERE = Path(__file__).parent
REQ = json.loads((HERE / "fixtures" / "ha_request.json").read_text())
NAMES = names.names_of(agent.parse_static_context(REQ["messages"][0]["content"]))


def fix(text, ns=NAMES):
    return names.correct(text, ns)


def test_misheard_name_is_corrected_and_logged_with_its_score():
    out, subs = fix("turn off the office speeker")
    assert out == "turn off the Office Speaker"
    assert subs == [{"from": "office speeker", "to": "Office Speaker", "score": subs[0]["score"]}] and subs[0]["score"] >= 0.85
    out, subs = names.correct("turn on the cabnets", ["Cabinets", "Kitchen Light"])
    assert out == "turn on the Cabinets" and subs == [{"from": "cabnets", "to": "Cabinets", "score": 0.933}]
    assert fix("turn on the kitchen cownter")[0] == "turn on the Kitchen Counter"


def test_near_tie_is_left_alone():
    # "bedroom fam" is close to two names: no single winner, so nothing changes.
    out, subs = names.correct("turn on the bedroom fam", ["Bedroom Fan", "Bedroom Fax"])
    assert out == "turn on the bedroom fam" and subs == []


def test_ordinary_words_are_never_replaced():
    for text in ["set the timer to cook pasta", "turn on the lamp", "what is the weather", "how do I cook rice",
                 "set the thermostat to 70", "turn on the kitchen light", "turn on the kitchen counter",
                 "play music on the office dot", "turn off the dining room"]:
        assert fix(text) == (text, []), text
    # A name that is itself an ordinary word does not pull ordinary words toward it.
    assert names.correct("cook it for ten minutes", ["Cooktop"]) == ("cook it for ten minutes", [])
    # Found replaying the live voice log: one letter from a player's name, but a word.
    assert names.correct("I'm in the mood for some jazz", ["moOde"]) == ("I'm in the mood for some jazz", [])


def test_short_form_of_a_real_name_is_left_alone():
    assert fix("turn on the dining room") == ("turn on the dining room", [])


def test_an_inflected_name_is_left_alone():
    # Nothing here was misheard: the extra letters are the user's, not the speech-to-text's.
    for text, name in [("is the front door locked", "Front Door Lock"), ("is the back door unlocked", "Back Door Lock"),
                       ("is the garage door locked", "Garage Door Lock"), ("start the robot vacuuming", "Robot Vacuum"),
                       ("are the bedroom blinds opened", "Bedroom Blind"), ("turn off the kitchen speakers", "Kitchen Speaker"),
                       ("turn on the dining room lamps", "Dining Room Lamp"),
                       ("switch off the bathroom heaters", "Bathroom Heater")]:
        assert names.correct(text, [name]) == (text, []), text
    # A word cut short is a short form too: the name word starts with it.
    assert names.correct("turn up the office speak", ["Office Speaker"]) == ("turn up the office speak", [])
    # A mishearing is still one: neither word starts with or contains the other.
    assert names.correct("turn off the kitchen speeker", ["Kitchen Speaker"])[0] == "turn off the Kitchen Speaker"


def test_a_short_name_word_inside_a_misheard_word_is_not_an_inflection():
    # "net" is inside "cabnets", but a three-letter name word turns up inside too many words to count.
    assert names.correct("turn on the cabnets", ["Cabinets", "Net"])[0] == "turn on the Cabinets"


def test_a_span_that_starts_or_ends_with_an_ordinary_word_is_left_alone():
    # Each would eat the request's own word: the verb, or "on".
    assert names.correct("close the shades", ["Closet Shades"]) == ("close the shades", [])
    assert names.correct("turn the porch on", ["Porch Neon"]) == ("turn the porch on", [])


def test_a_span_overlapping_a_name_said_correctly_is_left_alone():
    # "speaker garden" is close to Speeker Garden, but "speaker" already belongs to Office Speaker.
    text = "turn off the office speaker garden lights"
    assert names.correct(text, ["Office Speaker", "Speeker Garden"]) == (text, [])


def test_disabled_and_empty():
    assert names.correct("turn off the office speeker", NAMES, threshold=0) == ("turn off the office speeker", [])
    assert names.correct("turn off the office speeker", []) == ("turn off the office speeker", [])


def test_apply_touches_only_the_last_user_message_and_keeps_the_original_dict():
    first = {"role": "user", "content": "turn off the office speeker"}
    tool = {"role": "tool", "tool_call_id": "c1", "content": "office speeker not found"}
    last = {"role": "user", "content": "and the bedroom fam"}
    body = {"messages": [first, tool, last]}
    subs = names.apply(body, [{"names": "Office Speaker"}, {"names": "Bedroom Fan"}])
    assert body["messages"][0] is first and body["messages"][1] is tool  # earlier turns and tool results untouched
    assert body["messages"][2]["content"] == "and the Bedroom Fan" and last["content"] == "and the bedroom fam"
    assert [s["to"] for s in subs] == ["Bedroom Fan"]


def test_through_the_proxy_with_agent_mode_off(tmp_path, monkeypatch):
    seen = []

    async def upstream(request):
        seen.append(await request.json())
        return web.json_response({"choices": [{"message": {"role": "assistant", "content": "OK."}, "finish_reason": "stop"}]})

    system = {"role": "system", "content": REQ["messages"][0]["content"]}
    monkeypatch.setattr(app, "AGENT_MODE", False)
    monkeypatch.setattr(app, "VOICE_LOG", str(tmp_path / "voice-log.jsonl"))
    said, _, _ = run(upstream, "turn off the office speeker", stream=False, history=[system])
    assert seen[0]["messages"][-1]["content"] == "turn off the Office Speaker"
    turn = [json.loads(line) for line in (tmp_path / "voice-log.jsonl").read_text().splitlines()][-1]
    assert turn["heard"] == "turn off the office speeker"
    assert [(c["from"], c["to"]) for c in turn["corrected"]] == [("office speeker", "Office Speaker")]
