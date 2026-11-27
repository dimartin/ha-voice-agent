"""Correct a misheard device name in the user's words before the model sees them.

Speech-to-text mishears names ("cabnets" for Cabinets), and a small model then guesses
or acts on the wrong device. Home Assistant's request already lists every exposed name
and alias (agent.parse_static_context), so a span of the user's words that is close to
exactly one of them, and not already one of them, is replaced by that name.

Rules, all conservative:
- spans of 1 to (longest name + 1) words are compared, ignoring case, spaces and
  punctuation, with difflib's ratio; NAME_FIX_THRESHOLD is the bar;
- a span is replaced only when exactly one name clears the bar;
- a span that is already a name, or overlaps one said correctly, is left alone;
- a span that is a short form or an inflection of a name is left alone: read in order,
  each of its words matches a word of that name, where a heard word matches a name word
  if one starts with the other, or the heard word contains a name word of at least
  INFLECTED_MIN letters ("dining room" for Dining Room Lamp, "front door locked" and
  "back door unlocked" for Front Door Lock, "kitchen speakers" for Kitchen Speaker);
  "speeker" for Speaker and "cabnets" for Cabinets still are corrected, as neither word
  starts with or contains the other;
- a span that starts or ends with an ordinary word (articles, command and question
  words, "light", "lamp"), or is made only of them, is never replaced, so "turn on the
  kitchen light" is not read as a mishearing of some other kitchen name;
- a span or name shorter than MIN_CHARS letters is never part of a correction.
"""
import difflib
import os
import re

THRESHOLD = float(os.environ.get("NAME_FIX_THRESHOLD", "0.85"))
# Spans and names shorter than this collide with ordinary words ("mood" is one letter from
# a player called "moOde"), so neither side of a correction may be shorter.
MIN_CHARS = 6
# A heard word containing a name word this long is an inflection of it ("unlocked" of
# "lock"); shorter name words ("tv", "dot") turn up inside unrelated words.
INFLECTED_MIN = 4

# Words that are never part of a correction: they carry the request, not the name.
ORDINARY = frozenset("""
a an the my our your this that these those it its them all every some any of in on at to for from with by and or
please can could would will you me i we us is are was be do does did turn switch set put make change start stop
open close lock unlock play pause skip next dim brighten raise lower up down off on light lights lamp
what which who how when where why there here now then too also cook bake boil roast heat cool
""".split())


def _key(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _words(text: str) -> list[tuple[int, int, str]]:
    return [(m.start(), m.end(), m.group()) for m in re.finditer(r"[A-Za-z0-9']+", text)]


def _same_word(heard: str, name: str) -> bool:
    return heard.startswith(name) or name.startswith(heard) or (len(name) >= INFLECTED_MIN and name in heard)


def _short_form(heard: list[str], name: list[str]) -> bool:
    """Each heard word, in order, matches a later word of the name (greedy)."""
    j = 0
    for h in heard:
        while j < len(name) and not _same_word(h, name[j]):
            j += 1
        if j == len(name):
            return False
        j += 1
    return True


def names_of(devices: list[dict]) -> list[str]:
    out = []
    for d in devices:
        out += [n.strip() for n in (d.get("names") or "").split(",") if n.strip()]
    return sorted(set(out), key=str.lower)


def correct(text: str, names: list[str], threshold: float = THRESHOLD) -> tuple[str, list[dict]]:
    """Return the text with misheard names replaced, and one record per replacement."""
    if not text or not names or threshold <= 0:
        return text, []
    words = _words(text)
    keys = {n: _key(n) for n in names}
    name_tokens = {n: [t for t in map(_key, n.split()) if t] for n in names}
    longest = max(len(n.split()) for n in names)
    lowered = [w[2].lower() for w in words]

    # Words already covered by a name said correctly.
    taken = set()
    for n in names:
        k = len(n.split())
        for i in range(len(words) - k + 1):
            if _key(" ".join(w[2] for w in words[i:i + k])) == keys[n]:
                taken.update(range(i, i + k))

    found = []
    for size in range(longest + 1, 0, -1):
        for i in range(len(words) - size + 1):
            span = range(i, i + size)
            if taken & set(span):
                continue
            toks = lowered[i:i + size]
            if all(t in ORDINARY for t in toks) or toks[0] in ORDINARY or toks[-1] in ORDINARY:
                continue
            k = _key(" ".join(toks))
            if len(k) < MIN_CHARS:
                continue
            heard = [t for t in map(_key, toks) if t]
            scored = []
            for n in names:
                if _short_form(heard, name_tokens[n]):
                    scored = []  # a short form or inflection of a real name: not a mishearing
                    break
                if len(keys[n]) < MIN_CHARS:
                    continue
                r = difflib.SequenceMatcher(None, k, keys[n]).ratio()
                if r >= threshold:
                    scored.append((r, n))
            if len(scored) != 1:
                continue
            r, n = scored[0]
            found.append((words[i][0], words[i + size - 1][1], n, round(r, 3)))
            taken.update(span)

    out, subs = text, []
    for start, end, n, r in sorted(found, reverse=True):
        subs.append({"from": text[start:end], "to": n, "score": r})
        out = out[:start] + n + out[end:]
    return out, list(reversed(subs))


def apply(body: dict, devices: list[dict], threshold: float = THRESHOLD) -> list[dict]:
    """Correct the last user message in place; earlier turns and tool results stay as sent.
    The same input always gives the same output, so a follow-up request after a tool
    result carries the same corrected text and the server's cache still matches."""
    msgs = body.get("messages") or []
    for i in range(len(msgs) - 1, -1, -1):
        m = msgs[i]
        if m.get("role") == "user":
            if not isinstance(m.get("content"), str):
                return []
            text, subs = correct(m["content"], names_of(devices), threshold)
            if subs:
                msgs[i] = {**m, "content": text}  # a new dict: the caller's copy of the history keeps what was heard
            return subs
    return []
