"""The Wyoming filter in front of Piper: the rules it applies, and a real round trip."""
import asyncio

from wyoming.event import Event, async_read_event, async_write_event

from ha_voice_agent import tts_filter
from ha_voice_agent.spoken import to_spoken
from ha_voice_agent.tts_filter import fix

# The live filter's cases (2026-10-09), with "1 tablespoon" now singular.
CASES = {
    "Chicken thighs should reach 165°F (74°C) internally.":
        "Chicken thighs should reach 165 degrees Fahrenheit (74 degrees Celsius) internally.",
    "Set to 20% brightness": "Set to 20 percent brightness",
    "boil for **6–7 minutes**": "boil for 6 to 7 minutes",
    "1 tbsp lemon juice": "1 tablespoon lemon juice",
    "2 tsp. of salt": "2 teaspoons of salt",
    "Add 3 tbsp.": "Add 3 tablespoons.",
    "Add 1 tbsp. Then stir.": "Add 1 tablespoon. Then stir.",
    "Add 2 tsp. Salt to taste.": "Add 2 teaspoons. Salt to taste.",
    "Next collection is 2026-11-25.": "Next collection is 2026-11-25.",
    "Boil 6-7 minutes": "Boil 6 to 7 minutes",
    "It is 10-15 degrees": "It is 10 to 15 degrees",
    "It is 72° outside": "It is 72 degrees outside",
    "Line one.\n\nLine two.": "Line one. Line two.",
}


def test_rules():
    for text, want in CASES.items():
        assert fix(text) == want, text


def test_rules_are_idempotent():
    # LLM replies are cleaned by the proxy and then again here.
    for text in list(CASES) + list(CASES.values()):
        once = to_spoken(text)
        assert to_spoken(once) == once, text


def test_chunk_whitespace_is_kept():
    assert fix("Hello ") == "Hello "
    assert fix(" 57 °F with fog") == " 57 degrees Fahrenheit with fog"
    assert fix("  ") == "  "
    assert fix("") == ""


def test_round_trip_through_a_fake_piper():
    async def run():
        got = []

        async def piper(reader, writer):
            while (event := await async_read_event(reader)) is not None:
                got.append((event.type, dict(event.data or {})))
                await async_write_event(Event("audio-stop"), writer)
            writer.close()

        up = await asyncio.start_server(piper, "127.0.0.1", 0)
        tts_filter.UPSTREAM_HOST, tts_filter.UPSTREAM_PORT = "127.0.0.1", up.sockets[0].getsockname()[1]
        front = await tts_filter.serve("127.0.0.1", 0)
        reader, writer = await asyncio.open_connection("127.0.0.1", front.sockets[0].getsockname()[1])
        for e in [Event("describe"),
                  Event("synthesize-chunk", {"text": "57 °F "}),
                  Event("synthesize", {"text": "57 °F with fog", "voice": {"name": "en_US"}})]:
            await async_write_event(e, writer)
        replies = [await async_read_event(reader) for _ in range(3)]
        writer.close()
        front.close()
        up.close()
        return got, replies

    got, replies = asyncio.run(run())
    assert got == [("describe", {}),
                   ("synthesize-chunk", {"text": "57 degrees Fahrenheit "}),
                   ("synthesize", {"text": "57 degrees Fahrenheit with fog", "voice": {"name": "en_US"}})]
    assert [r.type for r in replies] == ["audio-stop"] * 3  # Piper's side passes back untouched
