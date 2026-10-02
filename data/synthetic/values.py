"""Fake PII values rendered in the formats M4 still misses (milestone 1, targeted data).

Every generator takes a seeded random.Random and returns (label, text) pairs, where text is
the exact string the message must contain. Numbers come from reserved or test ranges:
UK Ofcom drama numbers (07700 900xxx, 0113 496 0xxx, ...), NANP 555-01xx, published test
card numbers or random Luhn-valid ones, SSNs in the never-issued 9xx area, NINOs with the
documentation prefix QQ. Nothing here identifies a real person.
"""

from __future__ import annotations

import random

DIGIT_WORDS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
}  # fmt: skip
WORD_DIGITS = {w: d for d, w in DIGIT_WORDS.items()} | {"oh": "0"}

# ---------------------------------------------------------------- phone numbers

UK_LANDLINE = ["020 7946 0", "0113 496 0", "0114 496 0", "0115 496 0", "0116 496 0",
               "0117 496 0", "0118 496 0", "0121 496 0", "0131 496 0", "0141 496 0",
               "0151 496 0", "0161 496 0", "0191 498 0"]  # fmt: skip
NANP_AREAS = ["212", "312", "415", "617", "713", "202", "305", "206", "404", "512",
              "416", "604", "514", "403", "613", "902", "718", "646", "773", "818"]  # fmt: skip


def phone_digits(rng: random.Random, locale: str) -> list[str]:
    """Digit groups of a fake number for a locale (US, CA, UK, IN)."""
    if locale == "UK":
        if rng.random() < 0.5:
            return ["07700", "900", f"{rng.randint(0, 999):03d}"]
        a, b, c = rng.choice(UK_LANDLINE).split()
        return [a, b, c + f"{rng.randint(0, 999):03d}"]
    if locale == "IN":
        return [f"9{rng.randint(7000, 9999)}", f"{rng.randint(0, 99999):05d}"]
    return [rng.choice(NANP_AREAS), "555", f"01{rng.randint(0, 99):02d}"]


def spoken(digits: str, rng: random.Random, oh: float = 0.6, runs: bool = True) -> str:
    """Digits as words, with "oh" for zero and "double"/"triple" for runs."""
    words, i = [], 0
    while i < len(digits):
        d, n = digits[i], 1
        while runs and i + n < len(digits) and digits[i + n] == d and n < 3:
            n += 1
        word = "oh" if d == "0" and rng.random() < oh else DIGIT_WORDS[d]
        if n > 1 and rng.random() < 0.8:
            words.append(f"{'double' if n == 2 else 'triple'} {word}")
            i += n
        else:
            words.append(word)
            i += 1
    return " ".join(words)


def unspeak(text: str) -> str:
    """Inverse of the renderings: the digits a phone/number string stands for."""
    out, mult = [], 1
    for tok in text.replace(",", " ").replace("-", " ").replace(".", " ").replace("/", " ").split():
        t = tok.lower()
        if t in ("double", "triple"):
            mult = 2 if t == "double" else 3
        elif t in WORD_DIGITS:
            out.append(WORD_DIGITS[t] * mult)
            mult = 1
        elif t == "hundred":
            out.append("00")
        elif t == "plus":
            continue
        else:
            out.append("".join(c for c in t if c.isdigit()))
    return "".join(out)


def phone(rng: random.Random, locale: str) -> tuple[str, str, str]:
    """(label, text, format) for a phone number in one of the hard formats."""
    groups = phone_digits(rng, locale)
    fmt = rng.choices(
        ["spoken", "spoken_commas", "split_lines", "unspaced", "mixed", "digit_spaced",
         "intl_spoken", "plain"],
        weights=[3, 3, 2, 2, 2, 1, 2, 1],
    )[0]  # fmt: skip
    if fmt == "spoken":
        text = " ".join(spoken(g, rng) for g in groups)
    elif fmt == "spoken_commas":
        sep = rng.choice([", ", ". ", " - "])
        text = sep.join(spoken(g, rng) for g in groups)
    elif fmt == "split_lines":
        parts = [" ".join(g) if rng.random() < 0.3 else g for g in groups]
        text = "\n".join(parts)
    elif fmt == "unspaced":
        text = "".join(groups)
    elif fmt == "mixed":
        text = " ".join(g if rng.random() < 0.5 else spoken(g, rng) for g in groups)
        if text == " ".join(groups):
            text = " ".join([groups[0]] + [spoken(g, rng) for g in groups[1:]])
    elif fmt == "digit_spaced":
        text = rng.choice([" / ", "  ", " - "]).join(" ".join(g) for g in groups)
    elif fmt == "intl_spoken":
        cc = {"UK": "four four", "IN": "nine one", "US": "one", "CA": "one"}[locale]
        first = groups[0][1:] if locale == "UK" else groups[0]
        text = f"plus {cc}, " + ", ".join(spoken(g, rng) for g in [first, *groups[1:]])
    else:
        text = rng.choice([" ", "-", "."]).join(groups)
        if locale in ("US", "CA") and rng.random() < 0.5:
            text = f"({groups[0]}) {groups[1]}-{groups[2]}"
    return "TELEPHONENUM", text, fmt


# ---------------------------------------------------------------- titles

TITLES = ["Dr.", "Dr", "Prof.", "Rev.", "Revd", "Fr.", "Capt.", "Pvt.", "Sgt.", "Sgt", "Lt.",
          "Cpl.", "Maj.", "Col.", "Cmdr.", "Insp.", "Det.", "Sir", "Dame", "Lord", "Lady",
          "Hon.", "Judge", "Rabbi", "Imam", "Pastor", "Sister", "Brother", "Mx", "Mx.", "Ms",
          "Ms.", "Mrs", "Mrs.", "Mr", "Mr.", "Miss", "Sr.", "Sra.", "Herr", "Frau", "Mme",
          "Shri", "Smt."]  # fmt: skip
STACKED = ["Prof. Dr.", "Dr Dr", "Lt. Col.", "Rev. Dr.", "Revd Dr", "Maj. Gen.", "Prof. Sir",
           "Rt. Hon.", "Dr. Ing."]  # fmt: skip


def title(rng: random.Random) -> tuple[str, str, str]:
    if rng.random() < 0.2:
        t, fmt = rng.choice(STACKED), "stacked"
    else:
        t, fmt = rng.choice(TITLES), "single"
    r = rng.random()
    if r < 0.2:
        t, fmt = t.lower(), fmt + "_lower"
    elif r < 0.27:
        t, fmt = t.upper(), fmt + "_upper"
    return "TITLE", t, fmt


# ---------------------------------------------------------------- identifiers and cards

TEST_CARDS = ["4111111111111111", "4012888888881881", "4000056655665556", "4242424242424242",
              "5555555555554444", "5105105105105100", "2223003122003222", "5200828282828210",
              "378282246310005", "371449635398431", "6011111111111117", "6011000990139424",
              "3530111333300000", "36227206271667"]  # fmt: skip


def luhn_ok(num: str) -> bool:
    total = 0
    for i, c in enumerate(reversed(num)):
        d = int(c) * (2 if i % 2 else 1)
        total += d - 9 if d > 9 else d
    return total % 10 == 0


def random_card(rng: random.Random) -> str:
    prefix = rng.choice(["4", "51", "52", "53", "54", "55"])
    body = prefix + "".join(str(rng.randint(0, 9)) for _ in range(15 - len(prefix)))
    for check in "0123456789":
        if luhn_ok(body + check):
            return body + check
    raise AssertionError("unreachable")


def card(rng: random.Random) -> tuple[str, str, str]:
    num = rng.choice(TEST_CARDS) if rng.random() < 0.5 else random_card(rng)
    if len(num) in (14, 15):  # Amex / Diners: 4-6-5 or 4-6-4
        groups = [num[:4], num[4:10], num[10:]]
    else:
        groups = [num[i : i + 4] for i in range(0, 16, 4)]
    fmt = rng.choice(["spaced", "4-8-4", "contiguous", "dots", "hyphens", "split_lines"])
    if fmt == "spaced":
        text = " ".join(groups)
    elif fmt == "4-8-4" and len(num) == 16:
        text = f"{num[:4]} {num[4:12]} {num[12:]}"
    elif fmt == "contiguous":
        text = num
    elif fmt == "dots":
        text = ".".join(groups)
    elif fmt == "hyphens":
        text = "-".join(groups)
    else:
        fmt = "split_lines"
        half = len(groups) // 2
        text = " ".join(groups[:half]) + "\n" + " ".join(groups[half:])
    return "CREDITCARDNUMBER", text, fmt


def ssn(rng: random.Random) -> tuple[str, str, str]:
    a, b, c = f"{rng.randint(900, 999)}", f"{rng.randint(10, 99)}", f"{rng.randint(1000, 9999)}"
    fmt = rng.choice(["hyphens", "spaced", "contiguous", "spoken"])
    text = {
        "hyphens": f"{a}-{b}-{c}",
        "spaced": f"{a} {b} {c}",
        "contiguous": a + b + c,
        "spoken": ", ".join(spoken(g, rng, oh=0.2, runs=False) for g in (a, b, c)),
    }[fmt]
    return "SOCIALNUM", text, "ssn_" + fmt


def nino(rng: random.Random) -> tuple[str, str, str]:
    d = [f"{rng.randint(0, 99):02d}" for _ in range(3)]
    suffix = rng.choice("ABCD")
    fmt = rng.choice(["spaced", "compact", "lower"])
    if fmt == "spaced":
        text = f"QQ {d[0]} {d[1]} {d[2]} {suffix}"
    elif fmt == "compact":
        text = f"QQ{''.join(d)}{suffix}"
    else:
        text = f"qq{''.join(d)}{suffix.lower()}"
    return "SOCIALNUM", text, "nino_" + fmt


def sin(rng: random.Random) -> tuple[str, str, str]:
    g = [f"{rng.randint(0, 999):03d}" for _ in range(3)]
    fmt = rng.choice(["spaced", "hyphens", "contiguous"])
    sep = {"spaced": " ", "hyphens": "-", "contiguous": ""}[fmt]
    return "SOCIALNUM", sep.join(g), "sin_" + fmt


# ---------------------------------------------------------------- dates of birth and ages

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]  # fmt: skip
ORD = {1: "1st", 2: "2nd", 3: "3rd", 21: "21st", 22: "22nd", 23: "23rd", 31: "31st"}
ORD_WORDS = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth",
             "ninth", "tenth", "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth",
             "sixteenth", "seventeenth", "eighteenth", "nineteenth", "twentieth",
             "twenty-first", "twenty-second", "twenty-third", "twenty-fourth", "twenty-fifth",
             "twenty-sixth", "twenty-seventh", "twenty-eighth"]  # fmt: skip
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
TEENS = ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
         "eighteen", "nineteen"]  # fmt: skip


def number_words(n: int) -> str:
    if n < 10:
        return DIGIT_WORDS[str(n)]
    if n < 20:
        return TEENS[n - 10]
    if n < 100:
        return TENS[n // 10] + ("" if n % 10 == 0 else "-" + DIGIT_WORDS[str(n % 10)])
    return "one hundred" + ("" if n == 100 else " and " + number_words(n - 100))


def dob(rng: random.Random) -> tuple[str, str, str]:
    y, m, d = rng.randint(1935, 2010), rng.randint(1, 12), rng.randint(1, 28)
    fmt = rng.choice(["ddmmyy", "ordinal_apos", "of_the", "spelled", "iso", "dotted",
                      "lower_month", "slashes_short"])  # fmt: skip
    ordn = ORD.get(d, f"{d}th")
    text = {
        "ddmmyy": f"{d:02d}{m:02d}{y % 100:02d}",
        "ordinal_apos": f"{ordn} {MONTHS[m - 1][:3]} '{y % 100:02d}",
        "of_the": f"the {ordn} of the {ORD.get(m, f'{m}th')} {y}",
        "spelled": f"{ORD_WORDS[d - 1]} of {MONTHS[m - 1]}, nineteen {number_words(y % 100)}"
        if y < 2000
        else f"{ORD_WORDS[d - 1]} of {MONTHS[m - 1]}, two thousand and {number_words(y % 100)}",
        "iso": f"{y}-{m:02d}-{d:02d}",
        "dotted": f"{d:02d}.{m:02d}.{y}",
        "lower_month": f"{MONTHS[m - 1][:3].lower()} {ordn} {y % 100:02d}",
        "slashes_short": f"{d}/{m}/{y % 100:02d}",
    }[fmt]
    return "DATE", text, "dob_" + fmt


def age(rng: random.Random) -> tuple[str, str, str]:
    n = rng.choice([rng.randint(1, 17), rng.randint(18, 89), rng.randint(90, 104)])
    fmt = rng.choice(["shorthand_sex", "yo", "y_o", "bare", "spelled"])
    text = {
        "shorthand_sex": f"{n}{rng.choice('MF')}",
        "yo": f"{n}yo",
        "y_o": f"{n} y/o",
        "bare": str(n),
        "spelled": number_words(n).replace("-", rng.choice(["-", " "])),
    }[fmt]
    return "AGE", text, "age_" + fmt


# ---------------------------------------------------------------- names and emails

GIVEN = ["Aaliyah", "Abdirahman", "Adaora", "Agnieszka", "Aiko", "Alasdair", "Amara",
         "Anjali", "Aoife", "Arash", "Astrid", "Babajide", "Beatriz", "Bronwen", "Cassius",
         "Chiamaka", "Cormac", "Dagny", "Dariusz", "Delphine", "Dmitri", "Ebele", "Eilidh",
         "Emeka", "Esperanza", "Farhan", "Fenella", "Folasade", "Gaurav", "Gwendolyn",
         "Hamza", "Hyun-woo", "Ignatius", "Ishaan", "Jacinta", "Jasper", "Joaquín", "Kalinda",
         "Kavya", "Keziah", "Lachlan", "Leilani", "Lorcan", "Mahalia", "Malachy", "Mehmet",
         "Mireille", "Moana", "Nalini", "Niamh", "Nnamdi", "Oisín", "Olamide", "Ottilie",
         "Pankaj", "Perpetua", "Quentin", "Radhika", "Rhiannon", "Rosalind", "Sanjay",
         "Saoirse", "Seren", "Shoshana", "Sunita", "Tadhg", "Temperance", "Thandiwe",
         "Ulrike", "Vikram", "Wiremu", "Xiomara", "Yaw", "Yusuf", "Zainab", "Zbigniew",
         "Hope", "Grant", "Rose", "May", "Bill", "Joy", "Sterling", "Summer", "Hunter",
         "Faith", "Mercy", "Reed"]  # fmt: skip
SURNAMES = ["Abernethy", "Achterberg", "Adeleke", "Agarwal", "Ballantyne", "Bhattacharya",
            "Brennan", "Castellano", "Chukwu", "Dąbrowski", "Dunleavy", "Eze", "Fairweather",
            "Fitzpatrick", "Gallagher", "Gonçalves", "Haddad", "Hollis", "Iyer", "Jankowski",
            "Kaur", "Kerrigan", "Kovačević", "Lindgren", "MacAskill", "Mbeki", "Mensah",
            "Nakagawa", "Nwosu", "O'Driscoll", "Okoro", "Pellegrini", "Quinlan", "Rautenbach",
            "Saunders", "Sithole", "Takahashi", "Tiernan", "Underhill", "Vasconcelos",
            "Wainwright", "Weatherby", "Xu", "Yeboah", "Zielinski", "van der Berg",
            "de la Cruz", "Al-Mansour", "Ní Bhriain", "Featherstonehaugh", "Baker", "Cook",
            "Mason", "Fisher", "Bishop", "King", "Black", "Green", "Young", "Long"]  # fmt: skip


# Names that are also ordinary words: the generator rejects a message where one occurs twice.
WORDLIKE = {n.lower() for n in ["Hope", "Grant", "Rose", "May", "Bill", "Joy", "Sterling",
            "Summer", "Hunter", "Faith", "Mercy", "Reed", "Baker", "Cook", "Mason", "Fisher",
            "Bishop", "King", "Black", "Green", "Young", "Long", "Seren", "Temperance",
            "Brother", "Sister", "Judge", "Lady", "Lord", "Miss", "Sir", "Dame"]}  # fmt: skip


def person(rng: random.Random, with_title: bool) -> list[tuple[str, str, str]]:
    out = [title(rng)] if with_title else []
    given, sur = rng.choice(GIVEN), rng.choice(SURNAMES)
    if rng.random() < 0.15 and given.lower() not in WORDLIKE and sur.lower() not in WORDLIKE:
        given, sur = given.lower(), sur.lower()
    if rng.random() < 0.1:
        given = given[0] + "."
    out += [("GIVENNAME", given, "name"), ("SURNAME", sur, "name")]
    return out


def email(rng: random.Random, given: str, sur: str) -> tuple[str, str, str]:
    g = "".join(c for c in given.lower() if c.isalpha()) or "user"
    s = "".join(c for c in sur.lower() if c.isalpha()) or "mail"
    local = rng.choice([f"{g}.{s}", f"{g[0]}{s}", f"{g}{rng.randint(1, 99)}", f"{s}.{g[0]}"])
    domain = rng.choice(["example.com", "example.org", "example.net", "mail.example.co.uk"])
    if rng.random() < 0.25:
        return "EMAIL", f"{local} at {domain.replace('.', ' dot ')}", "spelled"
    return "EMAIL", f"{local}@{domain}", "plain"


# ---------------------------------------------------------------- look-alikes (not PII)


def _digits(rng: random.Random, n: int) -> str:
    return str(rng.randint(1, 9)) + "".join(str(rng.randint(0, 9)) for _ in range(n - 1))


def lookalike(rng: random.Random) -> str:
    """A business string that resembles PII but is not personal (labelled keep)."""
    kind = rng.choice(["order", "ticket", "tracking", "gift", "serial", "sku", "invoice",
                       "promo", "price", "quantity", "case"])  # fmt: skip
    if kind == "order":
        return rng.choice([f"#{_digits(rng, 8)}", f"ORD-{_digits(rng, 6)}",
                           f"{_digits(rng, 3)}-{_digits(rng, 7)}-{_digits(rng, 7)}"])  # fmt: skip
    if kind == "ticket":
        return rng.choice([f"INC{_digits(rng, 7)}", f"TCK-{_digits(rng, 5)}",
                           f"#{_digits(rng, 4)}-{_digits(rng, 4)}"])  # fmt: skip
    if kind == "tracking":
        return rng.choice([f"1Z{rng.randint(100, 999)}AA1{_digits(rng, 10)}",
                           " ".join(_digits(rng, 4) for _ in range(5)) + f" {_digits(rng, 2)}",
                           f"JD{_digits(rng, 16)}"])  # fmt: skip
    if kind == "gift":  # 16 digits that fail Luhn, so they cannot be card numbers
        while True:
            num = _digits(rng, 16)
            if not luhn_ok(num):
                break
        return rng.choice([" ".join(num[i : i + 4] for i in range(0, 16, 4)), num,
                           "GC-" + "-".join(num[i : i + 4] for i in range(0, 16, 4))])  # fmt: skip
    if kind == "serial":
        return rng.choice([f"SN {_digits(rng, 3)}{rng.choice('ABCDEFGHJK')}{_digits(rng, 6)}",
                           f"S/N {rng.choice('XYZ')}{_digits(rng, 9)}"])  # fmt: skip
    if kind == "sku":
        return f"{rng.choice(['SKU', 'item', 'model'])} {_digits(rng, 3)}-{_digits(rng, 4)}"
    if kind == "invoice":
        return f"INV-{rng.randint(2023, 2026)}-{_digits(rng, 5)}"
    if kind == "promo":
        word = rng.choice(["SPRING", "WELCOME", "SAVE", "LOYAL", "FLASH", "XMAS"])
        return f"{word}{rng.choice([10, 15, 20, 25, 30, 50])}"
    if kind == "price":
        cents = f"{rng.randint(5, 900)}.{rng.randint(0, 99):02d}"
        return rng.choice([f"${cents}", f"£{cents}", f"Rs {rng.randint(100, 50000):,}",
                           f"CA${rng.randint(5, 900)}"])  # fmt: skip
    if kind == "quantity":
        return rng.choice(["two boxes", "three of the six-packs", "a dozen", "four units",
                           "twelve rolls", "both items", "five cartons"])  # fmt: skip
    return f"case {_digits(rng, 2)}-{_digits(rng, 6)}"


# ---------------------------------------------------------------- patterns

PATTERNS = {  # pattern -> share of messages
    "phone": 0.30,
    "title": 0.25,
    "id_card": 0.20,
    "dob_age": 0.15,
    "negative": 0.10,
}


def target_values(rng: random.Random, pattern: str, locale: str) -> list[tuple[str, str, str]]:
    """The values a message of this pattern must contain, in the order they should appear."""
    if pattern == "phone":
        vals = person(rng, with_title=rng.random() < 0.3) + [phone(rng, locale)]
        if rng.random() < 0.25:
            vals.append(phone(rng, locale))
        return vals
    if pattern == "title":
        vals = person(rng, with_title=True)
        if rng.random() < 0.35:
            vals += person(rng, with_title=True)
        return vals
    if pattern == "id_card":
        maker = rng.choice([card, card, ssn, nino, sin]) if locale != "IN" else card
        if locale == "UK" and maker in (ssn, sin):
            maker = nino
        if locale == "CA" and maker in (ssn, nino):
            maker = sin
        if locale == "US" and maker in (nino, sin):
            maker = ssn
        return person(rng, with_title=rng.random() < 0.3) + [maker(rng)]
    if pattern == "dob_age":
        vals = person(rng, with_title=rng.random() < 0.3)
        vals += [dob(rng)] if rng.random() < 0.5 else [age(rng)]
        if rng.random() < 0.3:
            vals.append(age(rng) if vals[-1][0] == "DATE" else dob(rng))
        return vals
    vals = person(rng, with_title=False)  # negative: ordinary PII + look-alikes
    if rng.random() < 0.5:
        vals.append(email(rng, vals[-2][1], vals[-1][1]))
    return vals
