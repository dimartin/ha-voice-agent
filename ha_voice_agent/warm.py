"""Keep the LLM server's prompt cache warm.

In agent mode the system prompt and tools are byte-identical between requests, so
the server can reuse what it already read. On a CPU, reading them from cold takes
minutes, longer than Home Assistant waits for a voice turn. So the proxy saves that
shared prefix and sends it itself, with a one-word user turn and max_tokens 1:
whenever the prefix changes, at start, and when the server comes back after an
outage. Real requests are never held up by it; on a single-slot server the warm-up
simply queues behind them.
"""
import asyncio
import hashlib
import json
import logging
import os

from aiohttp import ClientError, ClientSession, ClientTimeout

WARM = os.environ.get("WARM", "1") == "1"
WARM_FILE = os.environ.get("WARM_FILE", "/data/warm-prefix.json")
WARM_CHECK_SECONDS = float(os.environ.get("WARM_CHECK_SECONDS", "60"))
WARM_TIMEOUT = float(os.environ.get("WARM_TIMEOUT", "900"))
log = logging.getLogger("voice-filter")


def prefix(body: dict) -> dict | None:
    """The shared part of a request: model, leading system message and tools."""
    msgs = body.get("messages") or []
    if not msgs or msgs[0].get("role") != "system":
        return None
    return {"model": body.get("model"), "messages": [msgs[0]], "tools": body.get("tools") or []}


def key(p: dict) -> str:
    return hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest()


class Warmer:
    def __init__(self, session: ClientSession, upstream: str, path: str = WARM_FILE):
        self.session, self.upstream, self.path = session, upstream, path
        self.prefix, self.key, self.task, self.healthy = None, None, None, False
        try:
            with open(path) as f:
                self.prefix = json.load(f)
            self.key = key(self.prefix)
        except (OSError, ValueError):
            pass

    def note(self, body: dict) -> None:
        """Call with each request as sent upstream; warms when its prefix is new."""
        p = prefix(body)
        if p is None or key(p) == self.key:
            return
        self.prefix, self.key = p, key(p)
        try:
            with open(self.path, "w") as f:
                json.dump(p, f)
        except OSError as e:
            log.warning("warm prefix save failed: %s", e)
        self.schedule("prefix changed")

    def schedule(self, why: str) -> None:
        if self.prefix is None or (self.task and not self.task.done()):
            return
        self.task = asyncio.get_running_loop().create_task(self._warm(why))

    async def _warm(self, why: str) -> None:
        started = self.key
        await self._send(why)
        # The prefix changed while this ran (a proxy start warms the saved prefix, and the
        # first real request after an edit brings a new one): warm the new one too.
        while self.key != started:
            started = self.key
            await self._send("prefix changed during a warm-up")

    async def _send(self, why: str) -> None:
        body = {**self.prefix, "messages": self.prefix["messages"] + [{"role": "user", "content": "hi"}],
                "max_tokens": 1, "stream": False}
        try:
            async with self.session.post(self.upstream + "/v1/chat/completions", json=body,
                                         timeout=ClientTimeout(total=WARM_TIMEOUT)) as r:
                out = await r.json(content_type=None) if r.status == 200 else {}
                t = out.get("timings") or {}
                log.info("warm (%s): status %s, prompt_n=%s cache_n=%s", why, r.status, t.get("prompt_n"), t.get("cache_n"))
        except (ClientError, asyncio.TimeoutError, ValueError) as e:
            log.warning("warm (%s) failed: %r", why, e)

    async def watch(self) -> None:
        """Warm at start and whenever the server comes back after being unreachable."""
        while True:
            try:
                async with self.session.get(self.upstream + "/health", timeout=ClientTimeout(total=10)) as r:
                    ok = r.status == 200
            except (ClientError, asyncio.TimeoutError):
                ok = False
            if ok and not self.healthy:
                self.schedule("server up")
            self.healthy = ok
            await asyncio.sleep(WARM_CHECK_SECONDS)
