"""Answer kitchen weights-and-measures questions without the LLM.

"How many teaspoons in a tablespoon?" or "How much is a quarter cup in tablespoons?"
gets a fixed answer from the table below. Anything else returns None and goes to the
model as before, including cups to grams, which depends on the ingredient.
"""
import re
from fractions import Fraction

TSP = 4.92892159375  # millilitres in a US teaspoon
# name -> (dimension, size in millilitres or grams, plural). US customary volume.
UNITS = {
    "teaspoon": ("volume", TSP, "teaspoons"), "tablespoon": ("volume", 3 * TSP, "tablespoons"),
    "fluid ounce": ("volume", 6 * TSP, "fluid ounces"), "cup": ("volume", 48 * TSP, "cups"),
    "pint": ("volume", 96 * TSP, "pints"), "quart": ("volume", 192 * TSP, "quarts"),
    "gallon": ("volume", 768 * TSP, "gallons"), "millilitre": ("volume", 1.0, "millilitres"),
    "litre": ("volume", 1000.0, "litres"),
    "ounce": ("weight", 28.349523125, "ounces"), "pound": ("weight", 453.59237, "pounds"),
    "gram": ("weight", 1.0, "grams"), "kilogram": ("weight", 1000.0, "kilograms"),
}
ALIASES = {
    "teaspoon": r"teaspoons?|tsps?", "tablespoon": r"tablespoons?|tbsps?",
    "fluid ounce": r"fluid ounces?|fl oz", "cup": r"cups?", "pint": r"pints?", "quart": r"quarts?",
    "gallon": r"gallons?", "millilitre": r"millilit(?:re|er)s?|ml", "litre": r"lit(?:re|er)s?",
    "ounce": r"ounces?|oz", "pound": r"pounds?|lbs?", "gram": r"grams?", "kilogram": r"kilograms?|kilos?|kg",
}
UNIT = "(" + "|".join(ALIASES.values()) + ")"
WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
         "eight": 8, "nine": 9, "ten": 10, "twelve": 12, "sixteen": 16}
PARTS = {"half": Fraction(1, 2), "third": Fraction(1, 3), "quarter": Fraction(1, 4), "eighth": Fraction(1, 8)}
PART = r"(?:half|thirds?|quarters?|eighths?)"
QTY = (rf"((?:\d+(?:\.\d+)?|\d+/\d+|{'|'.join(WORDS)})(?: (?:and )?(?:a )?{PART})?|(?:a )?{PART})")
# "how many teaspoons (are) in a tablespoon", "how many grams is one pound"
HOW_MANY = re.compile(rf"^how many {UNIT} (?:(?:are|is|make|equals?) )?(?:(?:in|to|per) )?(?:{QTY} )?{UNIT}$")
# "how much is a quarter cup in tablespoons", "convert 2 cups to millilitres"
HOW_MUCH = re.compile(rf"^(?:how (?:much|many) is |what is |convert )?{QTY} (?:of an? )?{UNIT} (?:in|to|into) {UNIT}$")
SPOKEN = {Fraction(1, 2): "a half", Fraction(1, 3): "a third", Fraction(2, 3): "two thirds",
          Fraction(1, 4): "a quarter", Fraction(3, 4): "three quarters", Fraction(1, 8): "an eighth"}


def unit_name(word):
    return next(n for n, a in ALIASES.items() if re.fullmatch(a, word))


def quantity(text):
    """'a quarter' -> 1/4, 'one and a half' -> 3/2, '2' -> 2, 'three quarters' -> 3/4."""
    if not text:
        return Fraction(1)
    said_and = "and" in text.split()  # "one and a third" adds; "one third" is a third
    words = [w for w in text.split() if w != "and"]
    whole = None
    if re.fullmatch(r"[\d./]+", words[0]):
        whole, words = Fraction(words[0]), words[1:]
    elif words[0] in WORDS and (len(words) == 1 or words[0] not in ("a", "an")):
        whole, words = Fraction(WORDS[words[0]]), words[1:]
    words = [w for w in words if w not in ("a", "an")]
    if not words:
        return whole
    part = PARTS[words[0].rstrip("s")]
    if said_and:  # "one and a half"
        return (whole or 0) + part
    return (whole or 1) * part  # "three quarters", "one quarter", "a half"


def say(amount, unit):
    """'3 teaspoons', 'a quarter cup', '1 and a half cups', 'about 237 millilitres'."""
    near = Fraction(amount).limit_denominator(8)
    exact = abs(float(near) - amount) <= 1e-9 * amount
    if abs(float(near) - amount) / amount < 0.01:
        whole, frac = divmod(near, 1)
        if not frac:
            return f"{whole} {unit if whole == 1 else UNITS[unit][2]}", exact
        if frac in SPOKEN:
            if not whole:  # "a quarter cup", "three quarters of a cup"
                if SPOKEN[frac].startswith(("a ", "an ")):
                    return f"{SPOKEN[frac]} {unit}", exact
                return f"{SPOKEN[frac]} of {'an' if unit[0] in 'aeiou' else 'a'} {unit}", exact
            return f"{whole} and {SPOKEN[frac]} {UNITS[unit][2]}", exact
    if amount >= 10:
        num = f"{round(amount)}"
    elif amount >= 1:
        num = f"{round(amount, 1):g}"
    else:
        num = f"{amount:.2g}"
    return f"{num} {unit if num == '1' else UNITS[unit][2]}", False


def answer(text):
    """A spoken answer for a weights-and-measures question, or None."""
    # Keep a decimal point ("1.5 cups"); drop every other full stop and punctuation.
    t = re.sub(r"[?!,]|\.(?!\d)|(?<!\d)\.", "", (text or "").lower())
    t = re.sub(r"\s+", " ", t).strip()
    if m := HOW_MANY.match(t):
        want, qty, have = m.groups()
    elif m := HOW_MUCH.match(t):
        qty, have, want = m.groups()
    else:
        return None
    have, want = unit_name(have), unit_name(want)
    # "ounces in a cup" means fluid ounces.
    if "ounce" in (have, want) and "volume" in (UNITS[have][0], UNITS[want][0]):
        have, want = (("fluid ounce" if u == "ounce" else u) for u in (have, want))
    try:
        q = quantity(qty)
    except (KeyError, ValueError, ZeroDivisionError):
        return None
    if not q or have == want or UNITS[have][0] != UNITS[want][0]:
        return None
    result = float(q) * UNITS[have][1] / UNITS[want][1]
    given, _ = say(float(q), have)
    out, exact = say(result, want)
    about = "" if exact else "about "
    if m.re is HOW_MANY:
        if q == 1:
            given = ("an " if have[0] in "aeiou" else "a ") + have
        verb = "is" if out.startswith(("1 ", "a ", "an ")) and " and " not in out else "are"
        return f"There {verb} {about}{out} in {given}."
    return f"{given[0].upper()}{given[1:]} is {about}{out}."
