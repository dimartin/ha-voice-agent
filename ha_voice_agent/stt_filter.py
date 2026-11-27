"""Wyoming speech-to-text filter: forwards audio unchanged, rewrites the transcript.

Parakeet writes the same request as "sconce two to ten percent" on one run and
"sconce 2-10%" on the next. Home Assistant's built-in intents match device names
literally, as registered, so a transcript whose number form differs from every registered
name and alias falls through to the LLM: seconds instead of milliseconds. This sits on
the STT server's port and fixes two things in the final `transcript` event:

- a merged range: "2-10%" becomes "2 to 10%", so the first number stays with the name;
- a name's number form: "den sconce 2" becomes "den sconce two" when that is how
  the name is registered and "den sconce 2" is not (and the other way round).

The registered names come from NAMES_FILE (a JSON list of strings), reloaded when it
changes. `ha-voice-stt-filter export-names` writes it from Home Assistant: for each entity
exposed to Assist, its aliases, with its name standing in for the computed-name alias (new
entities get that alias by default). That is the rule Home Assistant 2026.11 matches by
(entity_registry.async_get_entity_aliases), except for an entity whose alias list was
emptied: HA then matches it by no name, and the export still adds its name. Rewriting
towards a name that never matches breaks nothing.
"""
import argparse
import asyncio
import json
import logging
import os
import re
import sys

UPSTREAM_HOST = os.environ.get("UPSTREAM_HOST", "wyoming-whisper")
UPSTREAM_PORT = int(os.environ.get("UPSTREAM_PORT", "10300"))
LISTEN_PORT = int(os.environ.get("LISTEN_PORT", "10300"))
NAMES_FILE = os.environ.get("NAMES_FILE", "/data/stt-names.json")

log = logging.getLogger("stt-filter")

WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
         "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"]
TO_WORD = {str(i): w for i, w in enumerate(WORDS)}
TO_DIGIT = {w: str(i) for i, w in enumerate(WORDS)}
RANGE = re.compile(r"\b(\d+)\s*[-–]\s*(\d+)(\s*)(%|percent\b)", re.I)


def _flip(token: str) -> str | None:
    t = token.lower()
    return TO_WORD.get(t) or TO_DIGIT.get(t)


def variants(name: str) -> dict[str, str]:
    """{other spelling (lowercase): the registered spelling} for a name with numbers in it:
    every combination of its number tokens written the other way."""
    tokens = name.split()
    spots = [i for i, t in enumerate(tokens) if _flip(t)]
    out: dict[str, str] = {}
    if len(spots) == len(tokens):
        return out  # a name that is only numbers ("Two") would rewrite every "2" said
    for mask in range(1, 2 ** len(spots)):
        alt = list(tokens)
        for bit, i in enumerate(spots):
            if mask >> bit & 1:
                alt[i] = _flip(tokens[i]) or tokens[i]
        out[" ".join(alt).lower()] = name
    return out


class Names:
    """The registered names, and the rewrites they imply, reloaded when the file changes."""

    def __init__(self, path: str = NAMES_FILE):
        self.path, self.mtime, self.table, self.pattern = path, None, {}, None

    def load(self, names: list[str]) -> None:
        registered = {n.lower() for n in names}
        table = {}
        for n in names:
            for alt, real in variants(n).items():
                if alt not in registered:  # a spelling that is itself a name is never rewritten
                    table.setdefault(alt, real)
        # Registered names with numbers in them go into the pattern too, matched and left as
        # heard, so a correctly said "dining room sconce two" claims its span before a shorter
        # name's variant ("sconce two" for "Sconce 2") can match inside it.
        keep = {" ".join(n.split()).lower() for n in names if variants(n)}
        # One alternation, longest first, applied in a single pass: at each position the
        # longest spelling wins ("dining room sconce 2" over a shorter "sconce 2" inside it),
        # and a span already matched is never scanned again by a shorter name.
        self.table = table
        spans = sorted(set(table) | keep, key=len, reverse=True) if table else []
        self.pattern = re.compile(r"(?<![\w'])(?:" + "|".join(re.escape(a) for a in spans)
                                  + r")(?![\w'])", re.I) if spans else None

    def refresh(self) -> None:
        try:
            m = os.stat(self.path).st_mtime
        except OSError:
            return
        if m != self.mtime:
            try:
                with open(self.path) as f:
                    self.load([n for n in json.load(f) if isinstance(n, str)])
                self.mtime = m
                log.info("loaded %d name rewrites from %s", len(self.table), self.path)
            except (OSError, ValueError) as e:
                log.warning("names file %s unreadable (%s); keeping the previous rules", self.path, e)


def normalize(text: str, names: Names) -> str:
    out = RANGE.sub(lambda m: f"{m.group(1)} to {m.group(2)}{m.group(3)}{m.group(4)}", text)
    if names.pattern is None:
        return out
    lower = out.islower()

    def swap(m):
        real = names.table.get(m.group(0).lower())
        if real is None:  # a registered spelling: leave it as heard
            return m.group(0)
        return real.lower() if lower else real
    return names.pattern.sub(swap, out)


NAMES = Names()


async def pump(reader, writer, rewrite: bool) -> None:
    from wyoming.event import async_read_event, async_write_event
    try:
        while (event := await async_read_event(reader)) is not None:
            if rewrite and event.type == "transcript" and event.data and isinstance(event.data.get("text"), str):
                NAMES.refresh()
                new = normalize(event.data["text"], NAMES)
                if new != event.data["text"]:
                    log.info("transcript: %r -> %r", event.data["text"], new)
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
        pump(client_reader, up_writer, rewrite=False),  # audio and requests go up untouched
        pump(up_reader, client_writer, rewrite=True),   # the transcript comes back rewritten
        return_exceptions=True,
    )
    for err in results:
        if isinstance(err, Exception) and not isinstance(err, ConnectionError):
            log.warning("connection ended with %r", err)


async def serve(host: str = "0.0.0.0", port: int = LISTEN_PORT):
    return await asyncio.start_server(handle, host, port)


async def fetch_names(url: str, token: str) -> list[str]:
    """Names Home Assistant's matcher uses for entities exposed to Assist: the aliases, or
    the friendly name when there are none (or when an alias is the computed name)."""
    from aiohttp import ClientSession
    n = 0
    async with ClientSession() as http, http.ws_connect(url, ssl=False) as ws:
        await ws.receive_json()
        await ws.send_json({"type": "auth", "access_token": token})
        if (await ws.receive_json()).get("type") != "auth_ok":
            raise SystemExit("Home Assistant auth failed")

        async def call(msg):
            nonlocal n
            n += 1
            await ws.send_json({"id": n, **msg})
            while True:
                r = await ws.receive_json()
                if r.get("id") == n:
                    return r.get("result") or {}

        exposed = [e for e, v in (await call({"type": "homeassistant/expose_entity/list"})).get("exposed_entities", {}).items()
                   if v.get("conversation")]
        entries = await call({"type": "config/entity_registry/get_entries", "entity_ids": exposed})
        states = {s["entity_id"]: s for s in await call({"type": "get_states"})}
    names = set()
    for e in exposed:
        friendly = ((states.get(e) or {}).get("attributes") or {}).get("friendly_name")
        aliases = (entries.get(e) or {}).get("aliases") or []
        for a in aliases or [None]:
            a = a if isinstance(a, str) else friendly  # None stands for the computed name
            if a and a.strip():
                names.add(a.strip())
    return sorted(names, key=str.lower)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="ha-voice-stt-filter", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd")
    ex = sub.add_parser("export-names", help="write the names file from Home Assistant (token in HA_TOKEN)")
    ex.add_argument("--ha", required=True, help="wss://<host>:8123/api/websocket")
    a = p.parse_args(argv)
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    if a.cmd == "export-names":
        json.dump(asyncio.run(fetch_names(a.ha, os.environ["HA_TOKEN"])), sys.stdout, indent=1)
        sys.stdout.write("\n")
        return

    async def run():
        NAMES.refresh()
        server = await serve()
        log.info("listening on %s, forwarding to %s:%s", LISTEN_PORT, UPSTREAM_HOST, UPSTREAM_PORT)
        async with server:
            await server.serve_forever()

    asyncio.run(run())


if __name__ == "__main__":
    main()
