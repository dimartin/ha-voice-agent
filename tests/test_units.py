import asyncio

from aiohttp import ClientSession, web

from ha_voice_agent import app
from ha_voice_agent.units import answer

# Every wording the voice log recorded going to the LLM, 2026-10-23 to 10-30.
LOGGED = [
    "How many teaspoons in a tablespoon?",
    "How many teaspoons is one tablespoon?",
    "How many teaspoons in one tablespoon?",
    "How many teaspoons are in a tablespoon?",
]


def test_logged_wordings_answer_locally():
    for q in LOGGED:
        assert answer(q) == "There are 3 teaspoons in a tablespoon.", q
    assert answer("How much is a quarter cup in tablespoons?") == "A quarter cup is 4 tablespoons."


def test_volume_and_weight():
    assert answer("How many tablespoons in a cup?") == "There are 16 tablespoons in a cup."
    assert answer("How many ounces in a cup?") == "There are 8 fluid ounces in a cup."
    assert answer("How many ounces in a pound?") == "There are 16 ounces in a pound."
    assert answer("how many cups in a quart") == "There are 4 cups in a quart."
    assert answer("How many tablespoons in 3 teaspoons?") == "There is 1 tablespoon in 3 teaspoons."
    assert answer("How much is one and a half cups in tablespoons?") == "1 and a half cups is 24 tablespoons."
    assert answer("What is three quarters of a cup in tablespoons?") == "Three quarters of a cup is 12 tablespoons."
    assert answer("How many tablespoons in a quarter cup?") == "There are 4 tablespoons in a quarter cup."
    assert answer("How many cups in 4 tablespoons?") == "There is a quarter cup in 4 tablespoons."



def test_digits_with_a_decimal_point():
    # Speech-to-text writes "one and a half" as "1.5".
    assert answer("How much is 1.5 cups in tablespoons?") == "1 and a half cups is 24 tablespoons."
    assert answer("How much is 0.5 cups in tablespoons?") == "A half cup is 8 tablespoons."
    assert answer("How many tablespoons in 2.5 cups?") == "There are 40 tablespoons in 2 and a half cups."


def test_one_quarter_is_a_quarter_not_one_and_a_quarter():
    assert answer("How much is one quarter cup in tablespoons?") == "A quarter cup is 4 tablespoons."
    assert answer("How many tablespoons in one half cup?") == "There are 8 tablespoons in a half cup."
    assert answer("How much is one and a quarter cups in tablespoons?") == "1 and a quarter cups is 20 tablespoons."


def test_a_rounded_one_is_singular():
    assert answer("How many teaspoons in 5 millilitres?") == "There is about 1 teaspoon in 5 millilitres."

def test_metric_is_rounded_and_says_about():
    assert answer("How many grams in a pound?") == "There are about 454 grams in a pound."
    assert answer("How many millilitres in a cup?") == "There are about 237 millilitres in a cup."
    assert answer("Convert 2 kilograms to pounds") == "2 kilograms is about 4.4 pounds."


def test_everything_else_goes_to_the_llm():
    for q in ["How many cups of flour in a pound?",    # depends on the ingredient
              "How many grams in a cup?",              # volume to weight
              "How many teaspoons of salt in this recipe?",
              "How long should I boil an egg?",
              "Turn on the kitchen light.",
              "How many teaspoons in a teaspoon?",
              "", None]:
        assert answer(q) is None, q


def test_proxy_answers_without_calling_the_llm():
    async def run():
        calls = []

        async def llm(request):
            calls.append(1)
            return web.json_response({"choices": [{"message": {"role": "assistant", "content": "From the LLM."}}]})

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
        out = []
        try:
            async with ClientSession() as ha:
                for q in ["How many teaspoons are in a tablespoon?", "How long should I boil an egg?"]:
                    req = {"stream": False, "messages": [{"role": "user", "content": q}]}
                    async with ha.post(url, json=req) as r:
                        out.append((await r.json())["choices"][0]["message"]["content"])
        finally:
            await vf_runner.cleanup()
            await up_runner.cleanup()
        return calls, out

    calls, out = asyncio.run(run())
    assert out[0] == "There are 3 teaspoons in a tablespoon."
    assert len(calls) == 1  # only the egg question reached the LLM
