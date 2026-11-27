"""Turn LLM text into text a TTS voice can read, and cut it at a sentence boundary."""
import re

FRACTIONS = {"¼": "a quarter", "½": "a half", "¾": "three quarters", "⅓": "a third", "⅔": "two thirds", "⅛": "an eighth"}
EMOJI = re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F2FF️‍]")
SENTENCE_END = re.compile(r"(?<=[.!?])[\"')\]]?\s+")
# Home Assistant keeps the mic open after any reply ending in "?". After a false wake the model asks
# "could you repeat that?" and the room's next words start another turn, so these
# questions end with "." instead. A clarifying question like "Which room?" stays open.
REOPENER = re.compile(r"\b(repeat|rephrase|clarify|say (that|it) again|catch that|didn'?t (understand|hear)|"
                      r"not sure what you mean|what'?s up|how can i help|what can i do)\b", re.I)
# A turn whose last sentence is only one of these ends without asking the model.
CANCEL = re.compile(r"(?:^|[.!?,]\s*)(cancel|never ?mind|forget it|nothing|no thanks?|that'?s all)[\s.!?]*$", re.I)

# On unclear input the model sometimes writes its reasoning as the reply ("The user said
# "Yeah." This is a conversational filler... I should..."). A reply that opens this way is
# replaced whole, so none of it is spoken.
NARRATION = re.compile(r"^\W*the user(?:'s|’s)?\s+(?:is|was|said|says|asked|asks|wants|seems|appears|"
                       r"might|may|just|input|speech|request|message|utterance|query)\b", re.I)


def close_reopener(sentence: str) -> str:
    s = sentence.rstrip()
    if s.endswith("?") and REOPENER.search(s):
        return s[:-1] + "." + sentence[len(s):]
    return sentence


def wants_cancel(text) -> bool:
    return bool(text) and bool(CANCEL.search(text.strip()))


THINK_OPEN, THINK_CLOSE = "<think>", "</think>"


def to_spoken(text: str) -> str:
    t = text
    t = re.sub(r"```.*?```", " ", t, flags=re.S)
    t = re.sub(r"`([^`]*)`", r"\1", t)
    t = re.sub(r"(\*\*|__|\*|_)(\S.*?\S|\S)\1", r"\2", t)
    t = re.sub(r"^\s{0,3}#{1,6}\s*", "", t, flags=re.M)
    t = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", t, flags=re.M)
    t = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", t)
    t = EMOJI.sub("", t)
    for sym, word in FRACTIONS.items():
        t = re.sub(rf"(\d)\s*{sym}", rf"\1 and {word}", t)
        t = t.replace(sym, word)
    t = re.sub(r"\s*°\s*F\b", " degrees Fahrenheit", t)
    t = re.sub(r"\s*°\s*C\b", " degrees Celsius", t)
    t = t.replace("°", " degrees")
    t = re.sub(r"(\d)\s*%", r"\1 percent", t)
    # "1 tbsp" is one tablespoon; a full stop after the abbreviation goes only mid-sentence.
    t = re.sub(r"\b(1|one|a|an) (tbsp|tsp)s?\b(?:\.(?=\s+(?-i:[a-z])))?",
               lambda m: f"{m.group(1)} {'tablespoon' if m.group(2).lower() == 'tbsp' else 'teaspoon'}", t, flags=re.I)
    t = re.sub(r"\btbsps?\b(?:\.(?=\s+(?-i:[a-z])))?", "tablespoons", t, flags=re.I)
    t = re.sub(r"\btsps?\b(?:\.(?=\s+(?-i:[a-z])))?", "teaspoons", t, flags=re.I)
    # "6-7" is a range; "2026-11-25" (a date Home Assistant reads out) is left alone.
    t = re.sub(r"(?<![\d–-])(\d+)\s*[–-]\s*(\d+)(?![–-]?\d)", r"\1 to \2", t)
    t = re.sub(r"\s*[—–]\s*", ", ", t)
    t = t.replace(" & ", " and ")
    # A list the model wrote anyway becomes one run of sentences.
    t = re.sub(r"\s*\n+\s*", lambda m: " " if m.string[: m.start()].rstrip()[-1:] in ".!?:," else ". ", t)
    t = re.sub(r"\s{2,}", " ", t)
    return t.strip()


class SentenceCutter:
    """Feed streamed text; get back whole spoken sentences until the word budget is spent."""

    def __init__(self, max_words: int, narration_reply: str | None = "Sorry, I didn't catch that."):
        # narration_reply=None turns the narration check off (turns after a tool result).
        self.max_words = max_words
        self.narration_reply = narration_reply
        self.buf = ""
        self.words = 0
        self.done = False
        self.narrated = False

    def feed(self, text: str) -> str:
        if self.done:
            return ""
        self.buf += text
        # Reasoning models can leak their thinking tags into content. A complete
        # <think>...</think> block is never spoken; an open one holds output until it closes.
        while THINK_OPEN in self.buf:
            start = self.buf.index(THINK_OPEN)
            end = self.buf.find(THINK_CLOSE, start)
            if end < 0:
                return self._sentences(self.buf[:start], hold=self.buf[start:])
            self.buf = self.buf[:start] + self.buf[end + len(THINK_CLOSE):]
        # A stray </think> means the server opened the block in the prompt. Qwen-family
        # models then answer, emit the tag, and answer again: speak the first answer only.
        if THINK_CLOSE in self.buf:
            out = self._sentences(self.buf[: self.buf.index(THINK_CLOSE)])
            out += self.flush()
            self.done = True
            return out
        return self._sentences(self.buf)

    def _sentences(self, text: str, hold: str = "") -> str:
        self.buf = text
        out = []
        while not self.done:
            m = SENTENCE_END.search(self.buf)
            if not m:
                break
            out.append(self._emit(self.buf[: m.end()]))
            self.buf = self.buf[m.end():]
        self.buf += hold
        return "".join(out)

    def narration_tail(self, tools) -> str:
        """The fixed reply for a narrated turn, once the reply is complete. A narrated turn
        that called a tool says nothing: Home Assistant runs the tool and asks again."""
        return self.narration_reply + " " if self.narrated and not tools else ""

    def flush(self) -> str:
        if self.done or not self.buf.strip():
            return ""
        s, self.buf = self.buf, ""
        return self._emit(s)

    def _emit(self, sentence: str) -> str:
        spoken = close_reopener(to_spoken(sentence))
        if not spoken:
            return ""
        if self.narration_reply and not self.words and NARRATION.match(spoken):
            # Reasoning, not an answer: speak none of it. A tool call may still follow in
            # the stream, so the fixed reply waits for the end (narration_tail).
            self.narrated = self.done = True
            return ""
        n = len(spoken.split())
        # Always keep the first sentence, however long; never let a later one cross the budget.
        if self.words and self.words + n > self.max_words:
            self.done = True
            return ""
        self.words += n
        if self.words >= self.max_words:
            self.done = True
        return spoken + " "
