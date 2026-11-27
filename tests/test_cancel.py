import asyncio

from aiohttp import ClientSession, ClientTimeout, web

from ha_voice_agent import app


async def _serve(application, **kwargs):
    runner = web.AppRunner(application, **kwargs)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    return runner, f"http://127.0.0.1:{runner.addresses[0][1]}"


def test_abandoned_request_cancels_the_upstream_call():
    async def run():
        cancelled = asyncio.Event()

        async def slow_llm(request):
            try:
                await asyncio.sleep(30)  # a cold prompt, still processing
            except asyncio.CancelledError:
                cancelled.set()
                raise
            return web.Response(text="")

        llm = web.Application()
        llm.router.add_post("/v1/chat/completions", slow_llm)
        llm_runner, app.UPSTREAM = await _serve(llm, handler_cancellation=True)
        vf_runner, vf_url = await _serve(app.make_app(), **app.SERVER_KWARGS)
        try:
            async with ClientSession(timeout=ClientTimeout(total=0.5)) as ha:
                try:
                    await ha.post(vf_url + "/v1/chat/completions",
                                  json={"stream": True, "messages": [{"role": "user", "content": "hi"}]})
                except asyncio.TimeoutError:
                    pass  # HA gives up and closes the connection
            await asyncio.wait_for(cancelled.wait(), 3)
        finally:
            await vf_runner.cleanup()
            await llm_runner.cleanup()

    asyncio.run(run())
