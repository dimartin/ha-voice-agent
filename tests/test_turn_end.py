import asyncio
import json

from aiohttp import ClientSession, web

from ha_voice_agent import app
from ha_voice_agent.spoken import SentenceCutter, wants_cancel


def spoken(text):
    c = SentenceCutter(35)
    return (c.feed(text) + c.flush()).strip()


def test_repeat_requests_close_the_turn():
    # Replies to false wakes on 2026-10-29; each "?" kept pod1 listening to the room.
    assert spoken("I'm sorry, I didn't quite catch that. Could you please repeat it?") == \
        "I'm sorry, I didn't quite catch that. Could you please repeat it."
    assert spoken("I'm still not sure what you mean. Could you please clarify?") == \
        "I'm still not sure what you mean. Could you please clarify."
    assert spoken("I'm listening. What's up?") == "I'm listening. What's up."


def test_real_clarifying_questions_stay_open():
    assert spoken("Which room, kitchen or office?") == "Which room, kitchen or office?"
    assert spoken("Should I set it for 7 AM?") == "Should I set it for 7 AM?"


def test_trailing_cancel_is_detected():
    assert wants_cancel("Thanks, Lad Lab'em. Cancel.")
    assert wants_cancel("never mind")
    assert wants_cancel("Oh, forget it!")
    assert not wants_cancel("Cancel the kitchen timer.")
    assert not wants_cancel("Turn on nothing but the lamp")
    assert not wants_cancel(None)


def test_cancel_answers_without_calling_the_llm():
    async def run():
        calls = []

        async def llm(request):
            calls.append(1)
            return web.Response(text="")

        up = web.Application()
        up.router.add_post("/v1/chat/completions", llm)
        up_runner = web.AppRunner(up)
        await up_runner.setup()
        await web.TCPSite(up_runner, "127.0.0.1", 0).start()
        app.UPSTREAM = f"http://127.0.0.1:{up_runner.addresses[0][1]}"
        vf_runner = web.AppRunner(app.make_app())
        await vf_runner.setup()
        await web.TCPSite(vf_runner, "127.0.0.1", 0).start()
        url = f"http://127.0.0.1:{vf_runner.addresses[0][1]}/v1/chat/completions"
        try:
            async with ClientSession() as ha:
                req = {"stream": True, "messages": [{"role": "user", "content": "Thanks. Cancel."}]}
                async with ha.post(url, json=req) as r:
                    lines = [x for x in (await r.text()).split("\n\n") if x]
                async with ha.post(url, json={**req, "stream": False}) as r:
                    plain = await r.json()
        finally:
            await vf_runner.cleanup()
            await up_runner.cleanup()
        return calls, lines, plain

    calls, lines, plain = asyncio.run(run())
    assert calls == []
    assert lines[-1] == "data: [DONE]"
    first = json.loads(lines[0][5:])
    assert first["choices"][0]["delta"] == {"role": "assistant", "content": "Okay."}
    assert json.loads(lines[1][5:])["choices"][0]["finish_reason"] == "stop"
    assert plain["choices"][0]["message"]["content"] == "Okay."
