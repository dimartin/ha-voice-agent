from ha_voice_agent.spoken import SentenceCutter, to_spoken


def test_markdown_list_becomes_sentences():
    t = "Buttermilk substitutes:\n\n- **1 cup milk + 1 tablespoon lemon juice** — stir.\n- Yogurt thinned with milk."
    assert to_spoken(t) == "Buttermilk substitutes: 1 cup milk + 1 tablespoon lemon juice, stir. Yogurt thinned with milk."


def test_units_and_symbols():
    assert to_spoken("Roast at 425°F for 1¼ hours, 20% less salt.") == \
        "Roast at 425 degrees Fahrenheit for 1 and a quarter hours, 20 percent less salt."
    assert to_spoken("Boil 6–7 min 🥚") == "Boil 6 to 7 min"


def test_decimal_is_not_a_sentence_end():
    c = SentenceCutter(100)
    assert c.feed("Use 1.") == ""
    assert c.feed("5 cups. Then") == "Use 1.5 cups. "


def test_sentence_that_would_cross_budget_is_dropped():
    c = SentenceCutter(10)
    text = "One two three four five six. Seven eight nine ten eleven. Twelve thirteen. Fourteen."
    out = "".join(c.feed(ch) for ch in text) + c.flush()
    assert out.strip() == "One two three four five six."
    assert c.done


def test_sentences_within_budget_all_survive():
    c = SentenceCutter(10)
    out = c.feed("One two three. Four five six. Seven. ") + c.flush()
    assert out.strip() == "One two three. Four five six. Seven."


def test_first_sentence_always_survives():
    c = SentenceCutter(3)
    assert c.feed("This first sentence is longer than the budget. More.").strip() == \
        "This first sentence is longer than the budget."


def test_flush_emits_unterminated_tail():
    c = SentenceCutter(50)
    assert c.feed("Three teaspoons") == ""
    assert c.flush().strip() == "Three teaspoons"


def stream(cutter, text, size=3):
    return "".join(cutter.feed(text[i:i + size]) for i in range(0, len(text), size)) + cutter.flush()


def test_stray_think_close_keeps_first_answer_only():
    # Seen from Ornith 35B-A3B on llama.cpp with thinking disabled.
    text = "There are 3 teaspoons in a tablespoon. </think>. There are 3 teaspoons in a tablespoon."
    assert stream(SentenceCutter(35), text).strip() == "There are 3 teaspoons in a tablespoon."
    c = SentenceCutter(35)
    assert (c.feed(text) + c.flush()).strip() == "There are 3 teaspoons in a tablespoon."


def test_think_block_is_never_spoken():
    text = "<think>The user wants a temperature. Pork is 145.</think>Cook pork to 145 degrees Fahrenheit."
    assert stream(SentenceCutter(35), text).strip() == "Cook pork to 145 degrees Fahrenheit."


def test_open_think_block_holds_output():
    c = SentenceCutter(35)
    assert c.feed("<think>Let me check. Still thinking. ") == ""
    assert (c.feed("Done.</think>It is sunny. ") + c.flush()).strip() == "It is sunny."


# Replies logged in voice-log.jsonl between 2026-10-26 and 2026-11-10.
LEAKED = [
    "The user is expressing uncertainty about what to watch. This isn't a command or a question that requires a tool.",
    'The user said "Yeah." This is a conversational filler and doesn\'t contain a request. I should respond in a way that invites them.',
    'The user said "Exit." This is likely a command to stop the interaction or close something.',
    'The user\'s input "Yeah, that\'s what I want to do,\'cause I have a whole bunch of" is an incomplete sentence. I should ask for clarification.',
    "The user's speech is fragmented and doesn't contain a clear request or question. It sounds like they are continuing a conversation.",
]


def test_narrated_reasoning_is_replaced_whole():
    for text in LEAKED:
        c = SentenceCutter(35, narration_reply="Sorry, I didn't catch that.")
        out = "".join(c.feed(ch) for ch in text) + c.flush()
        assert out == "", text  # none of the reasoning is spoken
        assert c.narrated and c.done
        assert c.narration_tail([]).strip() == "Sorry, I didn't catch that."


def test_narrated_turn_that_calls_a_tool_says_nothing():
    c = SentenceCutter(35)
    assert c.feed("The user wants the kitchen light on. I will turn it on. ") == ""
    assert c.narrated and c.narration_tail(["HassTurnOn"]) == ""


def test_narration_check_is_off_after_a_tool_result():
    c = SentenceCutter(35, narration_reply=None)
    out = c.feed("The user's request has been completed. ") + c.flush()
    assert out.strip() == "The user's request has been completed."
    assert not c.narrated and c.narration_tail([]) == ""


def test_narration_only_counts_at_the_start():
    c = SentenceCutter(35)
    out = c.feed("Paris. The user said France, so that's the capital. ") + c.flush()
    assert out.strip() == "Paris. The user said France, so that's the capital."
    assert not c.narrated


def test_user_as_a_plain_noun_is_spoken():
    c = SentenceCutter(35)
    out = c.feed("The user manual says to hold the button for five seconds.") + c.flush()
    assert out.strip() == "The user manual says to hold the button for five seconds."
    assert not c.narrated
