"""Issue 1: the Wyoming STT filter makes number forms agree with registered names."""
import asyncio
import json

from wyoming.event import Event, async_read_event, async_write_event

from ha_voice_agent import stt_filter
from ha_voice_agent.stt_filter import Names, normalize, variants

# Every name with a number in a sample home (synthetic): 20, counting
# "[Acme] Smart TV ACME65X1ZZ", whose digits sit inside a word and so are never a number
# token. Plus two made-up ones ("Kitchen Counter" has no number; "Two" is a name made
# only of a number). "Den Sconce 2" is
# not a name, because that entity's only alias is "Den Sconce Two"; the dining room
# sconces have both spellings as aliases.
OURS = ["Den Sconce One", "Den Sconce Two", "Heating Zone 1", "Heating Zone 2", "Heating Zone 3",
        "Heating Zone One", "Heating Zone Three", "Heating Zone Two", "pod one", "pod two", "pod two Voice Assistant",
        "pod1", "Dining Room Sconce 1", "Dining Room Sconce 2", "Dining Room Sconce One", "Dining Room Sconce Two",
        "Diningroom Sconce One", "Diningroom Sconce Two", "Hall Light Two", "[Acme] Smart TV ACME65X1ZZ",
        "Kitchen Counter", "Two"]


def names(ns=OURS):
    n = Names("/nonexistent")
    n.load(ns)
    return n


def test_variants():
    assert variants("Den Sconce Two") == {"den sconce 2": "Den Sconce Two"}
    assert variants("Zone 1 Fan 2") == {"zone one fan 2": "Zone 1 Fan 2", "zone 1 fan two": "Zone 1 Fan 2",
                                         "zone one fan two": "Zone 1 Fan 2"}
    assert variants("Two") == {} and variants("Kitchen Counter") == {}


def test_number_forms_follow_the_registered_name():
    n = names()
    assert normalize("Set den sconce 2 to 5%", n) == "Set Den Sconce Two to 5%"
    assert normalize("turn on den sconce 2", n) == "turn on den sconce two"
    assert normalize("Turn on diningroom sconce 1", n) == "Turn on Diningroom Sconce One"


def test_a_spelling_that_is_itself_a_name_is_left_alone():
    n = names()
    # Both forms are registered for these, so either already matches.
    assert normalize("turn on dining room sconce 2", n) == "turn on dining room sconce 2"
    assert normalize("turn on dining room sconce one", n) == "turn on dining room sconce one"
    assert normalize("turn on dining room sconce two", n) == "turn on dining room sconce two"
    assert normalize("what is heating zone one", n) == "what is heating zone one"


def test_the_longest_name_wins_and_is_not_rewritten_again():
    # A shorter name inside a longer one, written with the other number convention.
    assert normalize("turn on dining room sconce 2", names(["Dining Room Sconce Two", "Sconce 2"])) \
        == "turn on dining room sconce two"
    assert normalize("turn on kitchen light one", names(["Kitchen Light 1", "Light One"])) \
        == "turn on kitchen light 1"
    assert normalize("turn on zone 1 fan 2", names(["Zone One Fan Two", "Fan 2"])) == "turn on zone one fan two"
    assert normalize("turn on sconce two", names(["Dining Room Sconce Two", "Sconce 2"])) == "turn on sconce 2"
    # A longer registered name said as registered is never rewritten by a shorter one inside it.
    assert normalize("turn on dining room sconce two", names(["Dining Room Sconce Two", "Sconce 2"])) \
        == "turn on dining room sconce two"
    assert normalize("turn on sconce two lamp", names(["Sconce Two Lamp", "Sconce 2"])) == "turn on sconce two lamp"
    assert normalize("turn on kitchen light 1", names(["Kitchen Light 1", "Light One"])) == "turn on kitchen light 1"


def test_merged_range_is_split():
    n = names()
    assert normalize("Set den sconce 2-10%", n) == "Set Den Sconce Two to 10%"
    assert normalize("Set den sconce 2-5%.", n) == "Set Den Sconce Two to 5%."
    assert normalize("set it to 20-30 percent", n) == "set it to 20 to 30 percent"
    assert normalize("set a timer for 2-3 minutes", n) == "set a timer for 2-3 minutes"  # only before a percent


def test_number_only_names_and_partial_words_never_rewrite():
    n = names()
    assert variants("[Acme] Smart TV ACME65X1ZZ") == {}  # digits inside a word are not a number
    assert normalize("turn on the acme smart tv acme65x1zz", n) == "turn on the acme smart tv acme65x1zz"
    assert normalize("set a timer for 2 minutes", n) == "set a timer for 2 minutes"  # "Two" is a name
    assert normalize("the den sconce 20 percent", n) == "the den sconce 20 percent"
    assert normalize("hall light 2s", n) == "hall light 2s"


def test_names_file_reloads_on_change(tmp_path):
    p = tmp_path / "names.json"
    p.write_text(json.dumps(["Den Sconce Two"]))
    n = Names(str(p))
    n.refresh()
    assert normalize("den sconce 2", n) == "den sconce two"
    p.write_text("not json")
    import os
    os.utime(p, (1, 1))
    n.refresh()  # a broken file keeps the previous rules
    assert normalize("den sconce 2", n) == "den sconce two"


def test_round_trip_through_a_fake_stt_server():
    async def run():
        got = []

        async def whisper(reader, writer):
            while (event := await async_read_event(reader)) is not None:
                got.append(event.type)
                if event.type == "audio-stop":
                    await async_write_event(Event("transcript", {"text": "Set den sconce 2-10%", "language": "en"}), writer)
            writer.close()

        up = await asyncio.start_server(whisper, "127.0.0.1", 0)
        stt_filter.UPSTREAM_HOST, stt_filter.UPSTREAM_PORT = "127.0.0.1", up.sockets[0].getsockname()[1]
        stt_filter.NAMES = names()
        front = await stt_filter.serve("127.0.0.1", 0)
        reader, writer = await asyncio.open_connection("127.0.0.1", front.sockets[0].getsockname()[1])
        for e in [Event("transcribe", {"language": "en"}), Event("audio-start", {"rate": 16000, "width": 2, "channels": 1}),
                  Event("audio-chunk", {"rate": 16000, "width": 2, "channels": 1}, payload=b"\0" * 320),
                  Event("audio-stop")]:
            await async_write_event(e, writer)
        reply = await async_read_event(reader)
        writer.close()
        front.close()
        up.close()
        return got, reply

    got, reply = asyncio.run(run())
    assert got == ["transcribe", "audio-start", "audio-chunk", "audio-stop"]  # audio goes up untouched
    assert reply.type == "transcript" and reply.data == {"text": "Set Den Sconce Two to 10%", "language": "en"}
