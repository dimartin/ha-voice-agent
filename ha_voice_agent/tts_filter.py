"""Wyoming text-to-speech filter: rewrites text Piper reads badly, then forwards to Piper.

It sits on Piper's port, so Home Assistant needs no change. Every event passes
through untouched except the text of synthesize and synthesize-chunk events, which
goes through the same spoken.to_spoken rules as the proxy's LLM replies ("165°F"
becomes "165 degrees Fahrenheit"). That covers the replies that never pass through
the proxy: Home Assistant's built-in intents and intent scripts.
"""
import asyncio
import logging
import os
import re

from wyoming.event import async_read_event, async_write_event

from .spoken import to_spoken

UPSTREAM_HOST = os.environ.get("UPSTREAM_HOST", "piper")
UPSTREAM_PORT = int(os.environ.get("UPSTREAM_PORT", "10200"))
LISTEN_PORT = int(os.environ.get("LISTEN_PORT", "10200"))
REWRITE = {"synthesize", "synthesize-chunk"}

log = logging.getLogger("tts-filter")


def fix(text: str) -> str:
    """to_spoken, keeping the text's own leading and trailing whitespace: streamed
    chunks are joined back together by Piper, so "Hello " + "world" must stay apart."""
    m = re.fullmatch(r"(\s*)(.*?)(\s*)", text, re.S)
    core = to_spoken(m.group(2)) if m.group(2) else ""
    return m.group(1) + core + m.group(3)


async def pump(reader, writer, rewrite: bool) -> None:
    try:
        while (event := await async_read_event(reader)) is not None:
            if rewrite and event.type in REWRITE and event.data and isinstance(event.data.get("text"), str):
                new = fix(event.data["text"])
                if new != event.data["text"]:
                    log.debug("%s: %r -> %r", event.type, event.data["text"], new)
                    event.data["text"] = new
            await async_write_event(event, writer)
    finally:
        writer.close()


async def handle(client_reader, client_writer) -> None:
    try:
        up_reader, up_writer = await asyncio.open_connection(UPSTREAM_HOST, UPSTREAM_PORT)
    except OSError as err:
        log.error("cannot reach %s:%s: %s", UPSTREAM_HOST, UPSTREAM_PORT, err)
        client_writer.close()
        return
    results = await asyncio.gather(
        pump(client_reader, up_writer, rewrite=True),
        pump(up_reader, client_writer, rewrite=False),
        return_exceptions=True,
    )
    for err in results:
        # A client hanging up is normal; anything else (a rule raising) would otherwise
        # leave a silent utterance with nothing in the log.
        if isinstance(err, Exception) and not isinstance(err, ConnectionError):
            log.warning("connection ended with %r", err)


async def serve(host: str = "0.0.0.0", port: int = LISTEN_PORT) -> asyncio.base_events.Server:
    return await asyncio.start_server(handle, host, port)


def main() -> None:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))

    async def run():
        server = await serve()
        log.info("listening on %s, forwarding to %s:%s", LISTEN_PORT, UPSTREAM_HOST, UPSTREAM_PORT)
        async with server:
            await server.serve_forever()

    asyncio.run(run())


if __name__ == "__main__":
    main()
