# ha-voice-agent

A small OpenAI-compatible proxy that sits between Home Assistant's Assist pipeline and a
local LLM server. It makes voice replies short and speakable, and cuts the prompt Home
Assistant sends roughly in half so a local model answers faster.

Home Assistant still runs speech-to-text, its built-in intents, text-to-speech and every
tool call. The proxy only changes what goes to the model and what comes back.

## What it does

**Speech cleanup (always on).** Replies are rewritten for a text-to-speech voice:
markdown, lists and emoji are removed, and symbols become words ("425°F" → "425 degrees
Fahrenheit", "1¼" → "1 and a quarter", "6–7" → "6 to 7"). The reply is cut at the last
whole sentence that fits in `MAX_WORDS`, and generation upstream stops there. Tool calls
pass through untouched.

**Agent mode (`AGENT_MODE=1`).** Home Assistant's request is rewritten before it reaches
the model:

- its system prompt is replaced by `config/agent-prompt.txt`, plus a compact list of the
  exposed devices grouped by area; Home Assistant's line naming the satellite's area moves
  to the start of the user's turn;
- tool schemas keep their names, parameter names, types, enums and required lists, but
  lose long descriptions (`config/tool-descriptions.json` overrides them by name);
- optional parameters in `DROP_PARAMS` and whole tools in `DROP_TOOLS` are removed.

Everything is derived only from Home Assistant's request, so the rewritten system prompt
and tools are byte-identical between requests, and between satellites, and the LLM
server's prompt cache keeps working.

**A short device list, for large homes (off by default).** With `LAZY_DEVICES_ABOVE=N`,
a home with more than N exposed devices gets area names with device counts in the system
prompt instead of every device. The satellite's own area goes in full at the start of the
user's turn, next to Home Assistant's area line, so the cached prefix stays the same for
every satellite. The model looks up any other area with a `ListDevices(area)` tool, which
is always the last tool and which the proxy answers itself (up to `LOCAL_ROUNDS` rounds),
then sends Home Assistant whatever the model does next; Home Assistant never sees it. The
voice log records `listed` areas. See "Measured" for why it is off on a CPU.

**Misheard device names are corrected.** Speech-to-text mishears names ("cabnets" for
Cabinets). Home Assistant's request lists every exposed name and alias, so a span of the
user's last message that is close to exactly one of them (difflib ratio at least
`NAME_FIX_THRESHOLD`) and not already a name is replaced by it before the model sees the
request, with or without agent mode. Short forms and inflections of a real name ("dining room" for
Dining Room Lamp, "front door locked" for Front Door Lock), spans that start or end with an
ordinary word, and anything under 6 letters are left alone, and a near tie between two names changes
nothing. Each correction goes to the voice log as `corrected` (from, to, score) with the
original text as `heard`. Replaying 119 logged requests from our home against its 270 entity
names changed none of them. This only helps requests that reach the LLM; Home Assistant's
built-in intents match before the proxy sees anything.

**Answers without the LLM.** Kitchen weights-and-measures questions ("how many
teaspoons in a tablespoon?", "a quarter cup in tablespoons") get a fixed answer from a
table in `units.py`. Cups to grams depends on the ingredient, so it still goes to the
model.

**Narrated reasoning is never spoken.** On unclear input a model sometimes writes its
reasoning as the reply ("The user said "Yeah." This is a conversational filler..."). A
reply that opens that way is replaced whole by `NARRATION_REPLY`.

**Every turn ends with something speakable.** An upstream error, no data for
`UPSTREAM_TIMEOUT` seconds, or a reply with nothing to say gets `FALLBACK_REPLY` instead of
silence. The real cause goes to the log and the voice log (`failed`), never to the speaker.

**Action calls need a command.** A small model sometimes turns a misheard fragment into an
action ("the cabinet" became turn-on). A call to a tool listed as an action in
`config/action-tools.json` goes through only if the request is a command: it has a
command word from the same file, or it is not a question and has a number, "on"/"off" or a
word like "timer" or "volume" ("kitchen lights off", "volume 30"). An answer to the
assistant's own question ("Which room?" "The kitchen.") and a follow-up ("and the bedroom
too") count with the request before them. Otherwise the proxy says `BLOCK_REPLY` and logs
the refused call (`blocked`). Reads such as `GetLiveContext` always pass. "I'm cold" is
refused too: asking back is the safe failure. Text the model streamed before a refused call
has already been sent, so the refusal follows it.

**A claimed action needs a tool behind it.** A model can answer a command with "I've
turned off the light" and no tool call, so nothing changes. When the request is a command
and no tool other than a read has succeeded in the turn yet, the reply is held until it
ends. If it matches `claim_phrases` in `config/action-tools.json` and does not end in "?"
(a reply that asks back confirms nothing), the proxy re-asks once with `rerun_note` on the
end of the user's message (the cached prefix is reused). Any tool that is not in `reads`
backs a confirmation when its result has no error, so `HassBroadcast`, the other timer tools
and an `intent_script` count even though the guard's `actions` list leaves them out. A new
read tool must go in `reads`. A tool call from the re-run goes to Home Assistant, and a
reply that claims nothing (a question back, say) is spoken; another claim, an empty reply
or a failed re-run gets `CLAIM_REPLY`. A claim after a tool result, in a request for later
(`later_phrases`: "in 30 minutes", "at 7", "tomorrow"), or streamed before the upstream
connection failed gets `CLAIM_REPLY` with no re-run, so nothing that acts now runs twice
or early. Questions ("did you turn it off?") are never checked. The voice log records
`claim_unbacked`, `rerun` (`act`, `skipped-after-tool`, `skipped-later` or
`skipped-failed`), `rerun_tools`, `rerun_raw` (the re-run's text) and `rerun_failed` (why
the re-run request failed). On Gemma 4
26B-A4B with our agent prompt, 60 runs of 10 commands produced no unbacked claim
(`eval/claim_rate.py` against a llama.cpp server running `gemma-4-26B-A4B-it-UD-Q4_K_M`;
the raw results aren't published), so this
rarely fires; it is there for smaller models.

**Something is said before a tool runs.** Tool calls are held until complete, then any
text the model wrote first ("Let me check the thermostat") is spoken, or `WORKING_PHRASE`
if nothing has been said yet in the turn. The turn after the tool result never gets it.

**The prompt cache stays warm.** On a CPU, reading the system prompt and tools from cold
takes minutes, longer than Home Assistant waits. The proxy saves that shared prefix to
`WARM_FILE` and sends it itself, with a one-word question and `max_tokens: 1`, whenever it
changes, at start, and when the server comes back after an outage. In agent mode the
satellite's area line goes in the user turn rather than the system prompt, so every
satellite shares one cached prefix. The voice log records `prompt_n` and `cache_n` (tokens
read versus reused) when llama.cpp reports them.

**Voice log.** Every turn is appended to `VOICE_LOG` as one JSON line: what was asked,
the raw reply, what was spoken, whether it was cut, and which tools were called.

## Measured

On one home (82 exposed devices, 33 tools, a 9B Qwen-family model on an Apple M5 Pro):

| | Home Assistant's own prompt | agent mode |
|---|---|---|
| prompt size | 7,524 tokens | ~3,600 tokens |
| right action or answer (17 requests × 2 runs) | 34/34 | 33/34 |
| answer length, mean / max | 18 / 29 words | 15 / 30 words |

The one miss was a misheard request that switched on a device. The rule "never turn
anything on or off to answer a question" in `config/agent-prompt.txt` fixed it: 0 of 12
misheard requests triggered an action afterwards, while 6 of 6 real commands and 6 of 6
state questions still called the right tool. The full set was not rerun after that fix.

On CPU only (i7-8700, llama.cpp), reading the full Home Assistant prompt without a cache
took 5 to 10 minutes depending on the model, which is why prompt size matters.

**Short device list, on CPU** (`LAZY_DEVICES_ABOVE`, 2026-11-25: Gemma 4 26B-A4B on
llama.cpp, 10 CPUs of an i7-8700, a real home's agent-mode prompt and 17 bench requests
from one satellite; the raw results aren't published):

| | full list | short list |
|---|---|---|
| cached prefix | 3,318 tokens | 2,767 tokens |
| cold read of the prefix, once per server start | 177 s | 147 s |
| first request with the satellite's own devices (165 tokens re-read) | 3.6 s to first token | 10.8 s |
| "turn on the kitchen counter light" from another room | 5.0 s, one round | 15.2 s, a `ListDevices` round first |
| the other 15 requests | | 14 made the same tool calls, within −0.4 to +0.8 s of the full list (reply length at temperature 0.6; prompt reading was about the same); "play Nina Simone" called `search_media` instead of `play_media` and took 7.0 s instead of 11.3 s; one run each |

On a CPU the full list costs nothing per request, because it sits in the cached prefix,
while the short list re-reads the satellite's devices whenever another satellite spoke last
and adds a model round for every other room. It saves 30 s once per server start. So the
default is the full list (`LAZY_DEVICES_ABOVE=0`). Not measured: a GPU or Apple Silicon
host, where reading is fast and the trade may differ, and homes much larger than ours.

## Run it

```sh
docker compose -f compose.example.yaml up -d --build
```

Then point Home Assistant's OpenAI-compatible conversation integration at
`http://<host>:8011/v1`. The API key it sends is passed through to `UPSTREAM` unchanged.

| variable | default | meaning |
|---|---|---|
| `UPSTREAM` | (required) | base URL of the OpenAI-compatible LLM server |
| `PORT` | `8011` | listen port |
| `MAX_WORDS` | `35` | sentence-boundary cut for spoken replies |
| `AGENT_MODE` | `0` | `1` rewrites Home Assistant's prompt and tools |
| `AGENT_PROMPT_FILE` | `/app/config/agent-prompt.txt` | voice rules for agent mode |
| `TOOL_DESCRIPTIONS_FILE` | `/app/config/tool-descriptions.json` | description overrides by tool name |
| `DROP_PARAMS` | empty | optional tool parameters to remove, comma-separated |
| `DROP_TOOLS` | empty | tools to remove, comma-separated |
| `FALLBACK_REPLY` | `Sorry, I can't answer right now.` | spoken when the upstream fails, times out or says nothing |
| `UPSTREAM_TIMEOUT` | `25` | seconds without data from the upstream before giving up; Home Assistant waits about 30 |
| `BLOCK_REPLY` | `Sorry, I didn't catch that.` | spoken instead of an action call with no command in the request; a reply ending in "?" would reopen the mic |
| `ACTION_TOOLS_FILE` | `/app/config/action-tools.json` | action tools, read tools, command words, claim and later phrases |
| `CLAIM_REPLY` | `Sorry, I couldn't do that.` | spoken instead of a reply that claims an action nothing did |
| `WORKING_PHRASE` | `One moment.` | said before a tool runs if nothing else was; empty turns it off |
| `WARM` | `1` | `0` turns the cache warm-up off; it runs only in agent mode |
| `WARM_FILE` | `/data/warm-prefix.json` | the saved prefix |
| `WARM_CHECK_SECONDS` | `60` | how often the server's `/health` is checked |
| `WARM_TIMEOUT` | `900` | how long a warm-up may take |
| `NARRATION_REPLY` | `Sorry, I didn't catch that.` | spoken instead of a reply that narrates the model's reasoning |
| `VOICE_LOG` | `/data/voice-log.jsonl` | per-turn log |
| `LAZY_DEVICES_ABOVE` | `0` | above this many devices, send area counts and the satellite's own area instead of every device; `0` always sends every device |
| `LOCAL_ROUNDS` | `2` | `ListDevices` rounds the proxy answers in one turn |
| `NAME_FIX_THRESHOLD` | `0.85` | how close a span must be to one device name to be corrected; `0` turns it off |

The files in `config/` are examples. Edit them for your home; Home Assistant's
intent-script tools in particular arrive with descriptions like "Execute Home Assistant
WeatherForecast intent", which the model can't use until you describe them.

## Text-to-speech filter

Replies that never reach the LLM, from Home Assistant's built-in intents and intent
scripts, can still contain text Piper reads badly: "64 °F", "20%", "1 tbsp".
`ha-voice-tts-filter` is a small Wyoming proxy that sits on Piper's port and applies
the same speech rules (`spoken.to_spoken`) to every `synthesize` and
`synthesize-chunk` event, then forwards it to Piper. Everything else passes through
untouched, so Home Assistant needs no change.

```sh
docker build -f Dockerfile.tts -t ha-voice-tts-filter .
```

| variable | default | meaning |
|---|---|---|
| `UPSTREAM_HOST` | `piper` | Piper's host |
| `UPSTREAM_PORT` | `10200` | Piper's port |
| `LISTEN_PORT` | `10200` | the port Home Assistant's Wyoming integration points at |
| `LOG_LEVEL` | `INFO` | `DEBUG` logs every rewrite |

Publish the filter on the port Home Assistant uses for Piper, and leave Piper itself
unpublished. `compose.example.yaml` shows both.

## Speech-to-text filter

Parakeet writes the same request as "sconce two to ten percent" on one run and
"sconce 2-10%" on the next, and Home Assistant's built-in intents match a device name
only as it is registered. In 2026.11 the registered names are an entity's aliases, or its
name when it has none: an entity whose only alias is "Den Sconce Two" does not match
"den sconce 2", although "Den Sconce 2" is its friendly name. Each miss falls
through to the LLM, seconds instead of milliseconds.

`ha-voice-stt-filter` is a Wyoming proxy on the STT server's port. Audio goes up
untouched; the final `transcript` coming back is rewritten in two ways:

- a merged range before a percent is split: "2-10%" becomes "2 to 10%";
- a name's number form follows the registered one: "den sconce 2" becomes
  "den sconce two" when only "Den Sconce Two" is registered (the case of the rest of
  the transcript is kept). A spelling that is itself
  registered is never rewritten, and a name made only of numbers is ignored.

The names come from `NAMES_FILE` (a JSON list), reloaded when it changes:

```sh
HA_TOKEN=... ha-voice-stt-filter export-names --ha wss://<host>:8123/api/websocket > data/stt-names.json
docker build -f Dockerfile.stt -t ha-voice-stt-filter .
```

`eval/stt_replay.py` runs a list of transcripts through Home Assistant's sentence debugger
before and after the filter (it parses, nothing runs) and fails if any transcript that
matched before stops matching or changes target. On 129 transcripts (the cases in issue 1
plus every distinct request in a real voice log), 8 were rewritten and all 8 went from no
device to the right one, including a merged range like "Set den sconce 2-5%."; none broke.
The transcripts and results aren't published.

| variable | default | meaning |
|---|---|---|
| `UPSTREAM_HOST` | `wyoming-whisper` | the STT server's host |
| `UPSTREAM_PORT` | `10300` | its port |
| `LISTEN_PORT` | `10300` | the port Home Assistant's Wyoming STT integration points at |
| `NAMES_FILE` | `/data/stt-names.json` | registered names, from `export-names` |
| `LOG_LEVEL` | `INFO` | every rewrite is logged at INFO |

## Suggest custom sentences

Requests that repeat and always work can skip the LLM: a custom sentence answers the same
words in Home Assistant's own matcher in milliseconds. Each voice log line carries `turn`
(one id for every request of a turn), `first_turn`, `calls` (tool names and arguments) and,
after a tool runs, `results` (`ok`, and the entity ids it acted on). From those,

    ha-voice-suggest /data/voice-log.jsonl --min-count 3 > suggestions.yaml

prints candidate sentences in Home Assistant's `custom_sentences` format, for a person to
review. A turn counts only if it was the first of its conversation, made exactly one action
call, every tool succeeded, the action resolved a real entity, the reply wasn't a question,
and nothing touched is safety-critical: no lock, alarm, valve or cover entity, and none of
"lock", "alarm", "valve", "garage", "door" or "gate" in the request or the call's arguments.
Turns are grouped by the words speech-to-text heard (`heard`, when the proxy corrected a
misheard name, otherwise `user`), because that is what Home Assistant's matcher sees. A
command is suggested once it has `--min-count` such turns or more, and only for built-in
`Hass*` intents. Words that also appear in a turn that failed, was refused or did something
else are not suggested, because the sentence would change what that request does; skipped
candidates and the reason go to stderr. This is the replay check, done from the log: a
suggested sentence is literal, so it can only change requests with the same words. A
candidate that fixes an `area` is marked in the output, since it would act on that area
from every satellite. With
`--ha wss://<host>:8123/api/websocket` and `HA_TOKEN` set, each candidate also goes through
Home Assistant's sentence debugger (it parses without running anything). One the debugger
reports as a match is dropped, with the intent or sentence trigger it matched. One for which
it only names a closest-guess intent is kept, since that request still reaches the LLM. The
certificate is verified; add `--insecure` for an install with a self-signed certificate.
`--ha` without `HA_TOKEN` exits with an error. Turns that only read state
are never suggested; their answers change over time. Nothing is written into Home Assistant.

## Report on repeated and failed requests

The proxy sees only requests that reached the LLM. Home Assistant sees every request but
keeps just the last 10 runs per pipeline, in memory, until it restarts. So:

    HA_TOKEN=... ha-voice-report collect --ha wss://<host>:8123/api/websocket --out runs.jsonl --every 60

snapshots finished pipeline runs into a file (read only: it uses the pipeline debug API),
skipping ones it already has, and

    ha-voice-report report --voice-log /data/voice-log.jsonl --runs runs.jsonl

prints requests repeated on the same device within `--window` seconds (default 60;
pairs closer than `--min-gap`, 3 s, are scripted replays and skipped), requests that
reached the LLM and ended in an error, a question back or a command with no tool call
(command words from `--action-tools`, default `ACTION_TOOLS_FILE`), and the most frequent
requests. Without `--runs` it uses the voice log alone; the voice log names no device, so
those repeats are not split by satellite. `--satellite-log` adds a satellite's own turn log
(JSON lines; `--ts-key`, `--text-key`, `--device-key`). A satellite row within 5 s of a
pipeline run is that run and counts once; the rest are counted and listed as heard by a
satellite but never a pipeline run. A timestamp with no zone is read as local time and a
bare number as epoch seconds, so convert naive UTC or epoch milliseconds first, or every
row lands outside the 5 s. `--json` prints the same as JSON. Nothing is a dashboard; it's
a text report.

## Limits

- Tested against Home Assistant's built-in llama.cpp integration with streaming on. Other
  OpenAI-compatible integrations should work but are untested.
- Agent mode parses the "Static Context" device list from Home Assistant's Assist API
  prompt. If Home Assistant changes that format, agent mode falls back to an empty device
  list; the tools still work.
- The voice log contains what people in your home said. Treat it as private.
- The warm-up refills the server's in-memory cache. Saving the cache to disk with
  llama.cpp's `--slot-save-path`, so a server restart costs nothing, is not built.
- A server that restarts between two `/health` checks (`WARM_CHECK_SECONDS`) is not
  noticed, so its cache stays cold until the prefix next changes or the proxy restarts.
  On a CPU, requests in that window time out and say `FALLBACK_REPLY`.
- The command check is per word, so it errs both ways:
  - It lets through a command word in a non-command ("did you turn it on"), and, because a
    number or a word like "red" makes a statement count as a command, ordinary statements
    such as "it's 72 in here" or "the red one". A follow-up ("me too", "and then he left")
    counts with any command just before it.
  - It refuses "I'm cold", a "yes" to an action the assistant offered ("Want me to turn it
    off?"), and a second clarification ("Which kitchen light?" "The big one."), since it
    looks back only one user turn.
  - Locks and garage doors need a confirmation step, which this is not.

## Develop

```sh
pip install -e ".[dev]"
pytest -q tests
```

Licensed under Apache-2.0.
