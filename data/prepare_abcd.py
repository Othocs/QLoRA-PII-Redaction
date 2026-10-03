"""ABCD test split -> data/processed/abcd.jsonl: real (human-typed) customer-service chats,
labelled from each conversation's fictional customer profile.

    python data/prepare_abcd.py

ABCD (ASAPP, "Action-Based Conversations Dataset", MIT licence): trained agents chatting with
crowd workers who play a customer from a scenario card. The card holds the customer's
(fictional) details, which they type into the chat; ABCD also ships a delexicalised copy of
every turn with some of those values replaced by tokens (<email>, <phone>, <zip_code>, ...).

Labels (whole word, case-insensitive, every occurrence, longest first):
  customer_name          GIVENNAME + SURNAME (full name and each part alone)
  email                  EMAIL
  phone                  TELEPHONENUM (also its 10 digits with any separators)
  street_address         BUILDINGNUM (leading number) + STREET; the street name alone -> STREET
  city / zip_code        CITY / ZIPCODE
  username, account_id,  IGNORE: personal but outside our 19 labels
  pin_number, password,
  security_answer
Not labelled (kept): order id, state, membership level, products, amounts.

Also labelled: a house number right before a labelled street (a new address), an agent's
self-introduction ("this is Amy from ...") as GIVENNAME, and usernames typed after
"username is / :" that are not on the card (IGNORE).

Documents are the agent and customer turns ("Agent: ..." / "Customer: ..."); the system
"action" turns are templated, not typed by people, and are left out.

Consistency filter: a conversation is dropped if, on any turn, the delexicalised copy has
more <email>/<phone>/<zip_code>/<street_address>/<username>/<account_id>/<pin_number> tokens
than we matched values of that type - i.e. someone retyped a value differently, which would
otherwise count as a false leak. (<name> in ABCD marks product names, so it is not checked.)
"""

from __future__ import annotations

import gzip
import json
import re
import sys
import urllib.request
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pii_gateway.spans import Example, Span, write_examples  # noqa: E402

URL = "https://github.com/asappresearch/abcd/raw/master/data/abcd_v1.1.json.gz"
RAW = Path("data/raw/eval/abcd/abcd_v1.1.json.gz")
OUT = Path("data/processed/abcd.jsonl")
SPEAKER = {"agent": "Agent", "customer": "Customer"}
IGNORED = ("username", "account_id", "pin_number", "password", "security_answer")
CHECKED = {"email": "EMAIL", "phone": "TELEPHONENUM", "zip_code": "ZIPCODE",
           "street_address": "STREET", "username": "IGNORE", "account_id": "IGNORE",
           "pin_number": "IGNORE"}  # fmt: skip


def word_pattern(value: str) -> re.Pattern:
    return re.compile(r"(?<!\w)" + re.escape(value) + r"(?!\w)", re.IGNORECASE)


def phone_pattern(phone: str) -> re.Pattern | None:
    digits = re.sub(r"\D", "", phone)
    if len(digits) < 7:
        return None
    # an opening "(" belongs to the number: "(757) 344-2937"
    return re.compile(r"(?<![\w(])\(?" + r"[\s().-]*".join(digits) + r"(?![\w])")


def targets(scenario: dict) -> list[tuple[re.Pattern, list[tuple[int, str]] | str, str]]:
    """(pattern, how to label a match, delex type) for every value of this conversation.
    `how` is a label, or a list of (group offset, label) parts for the street address."""
    per, order = scenario.get("personal", {}), scenario.get("order", {})
    out = []
    name = (per.get("customer_name") or "").strip()
    if name:
        parts = name.split()
        out.append((word_pattern(name), "NAME_FULL", "name"))
        out.append((word_pattern(parts[0]), "GIVENNAME", "name"))
        if len(parts) > 1:
            out.append((word_pattern(parts[-1]), "SURNAME", "name"))
    if per.get("email"):
        out.append((word_pattern(per["email"]), "EMAIL", "email"))
    if per.get("phone"):
        out.append((word_pattern(per["phone"]), "TELEPHONENUM", "phone"))
        pp = phone_pattern(per["phone"])
        if pp:
            out.append((pp, "TELEPHONENUM", "phone"))
    street = (order.get("street_address") or "").strip()
    if street:
        out.append((word_pattern(street), "STREET_FULL", "street_address"))
        m = re.match(r"(\d+)\s+(.+)", street)
        if m:
            out.append((word_pattern(m.group(2)), "STREET", "street_address"))
    for key, label in (("city", "CITY"), ("zip_code", "ZIPCODE")):
        if order.get(key):
            out.append((word_pattern(str(order[key])), label, key))
    for key in IGNORED:
        if per.get(key):
            out.append((word_pattern(str(per[key])), "IGNORE", key))
    if order.get("order_id") or per.get("order_id"):
        pass  # kept: not personal
    # longest values first so "chloe zhang" wins over "chloe"
    return sorted(out, key=lambda t: -len(t[0].pattern))


AGENT_INTRO = re.compile(r"(?:this is|my name is|name's|I'm|I am)\s+([A-Z][a-z]+)\b"
                         r"(?=\s*(?:from|here|,|\.|!|and|with|at|$))")  # fmt: skip
USERNAME = re.compile(r"user\s?name\s*(?:is|:|=)\s*([A-Za-z0-9_.]*[A-Za-z0-9])", re.I)
NOT_USERNAMES = {
    "first", "not", "the", "correct", "wrong", "incorrect", "still", "now", "also", "just",
    "customer", "agent", "and", "or", "please", "email", "is", "it", "was",
}  # fmt: skip


def label_text(text: str, scenario: dict, speaker: str = "customer") -> tuple[list[Span], Counter]:
    """Spans for every value occurrence in `text`, plus a count per delex type.

    Beyond the scenario values: a house number right before a labelled street (a customer
    typing a *new* address), an agent introducing themselves by name, and a username typed
    after "username is / :" that is not on the scenario card (IGNORE)."""
    spans: list[Span] = []
    found: Counter = Counter()

    def free(a: int, b: int) -> bool:
        return not any(a < s.end and s.start < b for s in spans)

    for pat, how, dtype in targets(scenario):
        for m in pat.finditer(text):
            a, b = m.start(), m.end()
            if not free(a, b):
                continue
            found[dtype] += 1
            value = text[a:b]
            if how == "NAME_FULL":
                parts = list(re.finditer(r"\S+", value))
                spans.append(Span(a + parts[0].start(), a + parts[0].end(), "GIVENNAME"))
                if len(parts) > 1:
                    spans.append(Span(a + parts[-1].start(), a + parts[-1].end(), "SURNAME"))
            elif how == "STREET_FULL":
                num = re.match(r"(\d+)\s+", value)
                if num:
                    spans.append(Span(a, a + len(num.group(1)), "BUILDINGNUM"))
                    spans.append(Span(a + num.end(), b, "STREET"))
                else:
                    spans.append(Span(a, b, "STREET"))
            else:
                spans.append(Span(a, b, how))
    for st in [x for x in spans if x.label == "STREET"]:
        num = re.search(r"(?<![\w])(\d{1,6})\s+$", text[: st.start])
        if num and free(num.start(1), num.end(1)):
            spans.append(Span(num.start(1), num.end(1), "BUILDINGNUM"))
    if speaker == "agent":
        for m in AGENT_INTRO.finditer(text):
            if free(m.start(1), m.end(1)):
                spans.append(Span(m.start(1), m.end(1), "GIVENNAME"))
    for m in USERNAME.finditer(text):
        if m.group(1).lower() not in NOT_USERNAMES and free(m.start(1), m.end(1)):
            spans.append(Span(m.start(1), m.end(1), "IGNORE"))
    return sorted(spans, key=lambda s: s.start), found


def build(convo: dict) -> tuple[Example | None, str]:
    sc = convo["scenario"]
    lines, spans, offset = [], [], 0
    for (speaker, utt), delex in zip(convo["original"], convo["delexed"], strict=True):
        if speaker not in SPEAKER:
            continue
        prefix = f"{SPEAKER[speaker]}: "
        turn_spans, found = label_text(utt, sc, speaker)
        expected = Counter(t[1:-1] for t in re.findall(r"<[a-z_]+>", delex["text"]))
        for dtype in CHECKED:
            if expected[dtype] > found[dtype]:
                return None, f"unmatched <{dtype}>"
        base = offset + len(prefix)
        spans += [
            Span(s.start + base, s.end + base, s.label, utt[s.start : s.end]) for s in turn_spans
        ]
        lines.append(prefix + utt)
        offset += len(prefix) + len(utt) + 1
    text = "\n".join(lines)
    for s in spans:
        assert text[s.start : s.end] == s.text, (convo["convo_id"], s)
    meta = {
        "source": "abcd",
        "flow": sc.get("flow"),
        "subflow": sc.get("subflow"),
        "language": "en",
    }
    return Example(f"abcd-{convo['convo_id']}", text, spans, meta), ""


def main() -> None:
    if not RAW.exists():
        RAW.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, RAW)
    with gzip.open(RAW) as f:
        data = json.load(f)
    kept, dropped = [], Counter()
    for convo in data["test"]:
        ex, why = build(convo)
        if ex is None:
            dropped[why] += 1
        else:
            kept.append(ex)
    write_examples(OUT, kept)
    labels = Counter(s.label for ex in kept for s in ex.spans)
    with_pii = sum(any(s.label != "IGNORE" for s in ex.spans) for ex in kept)
    stats = {"test_conversations": len(data["test"]), "kept": len(kept),
             "dropped": dict(dropped), "labels": dict(labels.most_common()),
             "docs_with_pii": with_pii}  # fmt: skip
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
