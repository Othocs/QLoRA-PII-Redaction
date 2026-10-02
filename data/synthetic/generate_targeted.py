"""Generate targeted support-desk training messages with the DeepSeek API (milestone 1).

    export DEEPSEEK_API_KEY=...          # or put it in .env (gitignored); never committed
    uv run python data/synthetic/generate_targeted.py --n 2000 [--workers 8] [--dry-run]

M4 still misses spelled-out / split phone numbers, titles, odd SSN / NINO / card formats and
compact dates of birth or shorthand ages. This script asks an LLM to write the *carrier text*
around values that Python generates (values.py), so the labels are exact by construction:
every occurrence of a supplied value, in any letter case, is labelled, longest first; the
LLM never labels anything. A message is rejected if a supplied value is missing; if a digit
run, an email or a title + capitalised word appears outside the labelled spans and the
declared look-alikes ("keep"); if a phone is given as the company's, a date of birth as a
header date, or a look-alike number as a payment card; or if it is over 1,200 characters.

Request specs are derived from --seed and the request index, and raw responses are cached
in data/synthetic/raw/ (gitignored), so a re-run resumes without re-paying for requests.
Output: data/synthetic/targeted_2k.jsonl (Example JSON lines, meta.source = "targeted").
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import ssl
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # data/ for sibling packages
from synthetic.values import PATTERNS, WORDLIKE, lookalike, target_values  # noqa: E402

HERE = Path(__file__).parent
RAW = HERE / "raw"
OUT = HERE / "targeted_2k.jsonl"
API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-chat"
PER_REQUEST = 5
MAX_CHARS = 1200
CHANNELS = {"chat": 0.4, "email": 0.4, "agent_note": 0.2}
LOCALES = ["US", "UK", "CA", "IN"]
PLACE = {"US": "the US", "UK": "the UK", "CA": "Canada", "IN": "India"}
STYLES = ["value mid-sentence with no label word before it",
          "value in a signature block or contact line",
          "value given after a correction or false start",
          "value inside a quoted earlier message or forwarded header",
          "terse agent shorthand (cb, ph, verif, NOK, cx)",
          "customer rambling about the problem first, details near the end",
          "short angry chat, lowercase, typos",
          "polite formal email with greeting and sign-off"]  # fmt: skip

SYSTEM = """You write realistic, varied customer-support messages for a training set that \
teaches a model to find personal data. Write like real customers and agents: typos, \
lowercase chat, run-on sentences, forwarded emails with signatures, terse agent notes with \
abbreviations (cust, cb, ph, verif, dob, NOK). Vary products and situations (banking, \
telecoms, retail, travel, utilities, healthcare admin, insurance, software, delivery).

Rules:
- Vary structure and length (from one line up to ~900 characters). Do NOT introduce every \
value with a phrase like "my phone number is" or "my SSN is": follow the style hint, and \
often just drop the value in ("cb on ...", "reach me ...", "txt ...", after a dash, in a \
signature). Never mention the country, locale, these instructions or the word "context".
- Each message must contain every supplied value EXACTLY as given (same characters, same \
case, same punctuation and line breaks), each at least once, each as a standalone word \
(never glued to letters or digits).
- Use the values naturally: a supplied title goes right before (or near) a supplied name; \
a supplied phone number is the person's number; a date is the person's date of birth.
- Do NOT add any other personal data: no other names, nicknames, phone numbers, emails, \
addresses, dates of birth, ages or ID numbers. Staff may sign off with a role or team, \
never a name.
- Include every supplied business string verbatim; they are not personal data. List them, \
and any other business codes or amounts you add, verbatim in "keep".
- At most 1,100 characters per message. English, with at most a short non-English phrase.

Return json only: {"messages": [{"id": "<id>", "text": "<message>", "keep": ["..."]}]}"""


# ---------------------------------------------------------------- request specs


def make_specs(seed: int, req: int) -> list[dict]:
    """The PER_REQUEST message specs of request number `req` (deterministic)."""
    rng = random.Random(f"{seed}-{req}")
    specs = []
    for k in range(PER_REQUEST):
        pattern = rng.choices(list(PATTERNS), weights=list(PATTERNS.values()))[0]
        locale = rng.choice(LOCALES)
        channel = rng.choices(list(CHANNELS), weights=list(CHANNELS.values()))[0]
        values = target_values(rng, pattern, locale)
        n_keep = rng.choice([2, 3]) if pattern == "negative" else rng.choice([0, 1, 1, 2])
        keep = [lookalike(rng) for _ in range(n_keep)]
        vals = [{"label": lab, "text": t, "format": f} for lab, t, f in values]
        specs.append({"id": f"r{req:04d}-{k}", "pattern": pattern, "locale": locale,
                      "channel": channel, "style": rng.choice(STYLES), "keep": keep,
                      "values": vals})  # fmt: skip
    return specs


def user_prompt(specs: list[dict]) -> str:
    lines = ["Write these messages. Values are shown as JSON strings (\\n = line break).", ""]
    for s in specs:
        vals = "; ".join(f"{v['label']}: {json.dumps(v['text'], ensure_ascii=False)}"
                         for v in s["values"])  # fmt: skip
        keep = "; ".join(json.dumps(k, ensure_ascii=False) for k in s["keep"])
        extra = f" Business strings (not personal): {keep}." if keep else ""
        lines.append(f"- id {s['id']}: {s['channel'].replace('_', ' ')}, customer in "
                     f"{PLACE[s['locale']]}, style: {s['style']}. Personal values: {vals}."
                     f"{extra}")  # fmt: skip
    return "\n".join(lines)


# ---------------------------------------------------------------- API


def api_key() -> str:
    key = os.environ.get("DEEPSEEK_API_KEY")
    env = Path(".env")
    if not key and env.exists():
        for line in env.read_text().splitlines():
            if line.strip().startswith("DEEPSEEK_API_KEY="):
                key = line.split("=", 1)[1].strip().strip("'\"")
    if not key:
        sys.exit("DEEPSEEK_API_KEY is not set (environment or .env)")
    return key


def ssl_context() -> ssl.SSLContext:
    """uv's standalone Python ships no CA bundle: use certifi, else the system bundle."""
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        system = Path("/etc/ssl/cert.pem")
        return ssl.create_default_context(cafile=str(system) if system.exists() else None)


SSL_CTX = ssl_context()


def call(key: str, specs: list[dict], temperature: float, retries: int = 4) -> dict:
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": user_prompt(specs)}],
        "response_format": {"type": "json_object"},
        "temperature": temperature,
        "max_tokens": 4000,
    }).encode()  # fmt: skip
    for attempt in range(retries):
        req = urllib.request.Request(API_URL, data=body, method="POST", headers={
            "Content-Type": "application/json", "Authorization": f"Bearer {key}"})  # fmt: skip
        try:
            with urllib.request.urlopen(req, timeout=180, context=SSL_CTX) as r:
                return json.loads(r.read())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            if isinstance(getattr(e, "reason", None), ssl.SSLError):  # not transient
                raise SystemExit(f"TLS error talking to DeepSeek: {e.reason}") from e
            code = getattr(e, "code", None)
            if code in (400, 401, 402, 403):  # bad request / key / balance: retrying won't help
                raise SystemExit(f"DeepSeek API error {code}: {getattr(e, 'reason', e)}") from e
            time.sleep(2 ** (attempt + 1))
    raise RuntimeError("DeepSeek API: retries exhausted")


def fetch(key: str, seed: int, req: int, temperature: float) -> dict | None:
    """Cached raw response for request `req`, calling the API if needed."""
    path = RAW / f"{seed}-{req:04d}.json"
    if path.exists():
        return json.loads(path.read_text())
    try:
        resp = call(key, make_specs(seed, req), temperature)
    except RuntimeError as e:
        print(f"request {req}: {e}", file=sys.stderr)
        return None
    path.write_text(json.dumps(resp, ensure_ascii=False))
    return resp


# ---------------------------------------------------------------- labelling and validation

DIGIT_RUN = re.compile(r"\d(?:[\d \-./]*\d)?")
EMAILISH = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+|\b\w+ at \w+ dot \w+")
# Semantic checks the labels cannot carry: a supplied phone given as the company's number, a
# date of birth used as a message header date, a look-alike number presented as a payment card.
COMPANY_PHONE = re.compile(
    r"\b(?:call|reach|contact|ring|phone|text)\s+us\b|\bus (?:at|on)\b"
    r"|\bour (?:number|line|helpline|team)\b",
    re.IGNORECASE,
)
HEADER_DATE = re.compile(r"(?:Sent|Date|Dated|dated|Received|Posted)\s*:?\s*$")
TITLE_NAME = re.compile(r"\b(?:Mr|Mrs|Ms|Mx|Dr|Prof|Rev|Capt|Sgt|Sir|Dame)\.?\s+[A-Z][a-z]+")


def _occurrences(text: str, value: str, flags: int = 0) -> list[int]:
    pat = re.compile(r"(?<![\w])" + re.escape(value) + r"(?![\w])", flags)
    return [m.start() for m in pat.finditer(text)]


def label_message(text: str, values: list[dict]) -> tuple[list[dict] | None, str]:
    """Spans for every occurrence of every supplied value (longest first, no overlaps).

    Rejects (None, reason) when a value is missing, or when a value that could also be an
    ordinary word or number ("May", "Long", a bare age "45") occurs more than once, since
    the second occurrence might not be the person's.
    """
    spans: list[dict] = []
    seen: set[str] = set()
    for v in sorted(values, key=lambda v: -len(v["text"])):
        val = v["text"]
        if val in seen:  # the same title or name supplied twice: labelled once, everywhere
            continue
        seen.add(val)
        ambiguous = val.lower() in WORDLIKE or (val.isdigit() and len(val) <= 3)
        if ambiguous and len(_occurrences(text, val, re.IGNORECASE)) != 1:
            return None, f"ambiguous {v['label']}"
        if not _occurrences(text, val):  # the exact form must be there at least once
            return None, f"missing value {v['label']}"
        hits = [i for i in _occurrences(text, val, re.IGNORECASE)  # "fr. aaliyah" too
                if not any(i < s["end"] and s["start"] < i + len(val) for s in spans)]  # fmt: skip
        for i in hits:
            spans.append({"start": i, "end": i + len(val), "label": v["label"],
                          "text": text[i : i + len(val)]})  # fmt: skip
    return sorted(spans, key=lambda s: s["start"]), ""


def validate(text: str, values: list[dict], keep: list[str]) -> tuple[dict | None, str]:
    """(record fields, "") for an accepted message, else (None, reason)."""
    if not text or len(text) > MAX_CHARS:
        return None, "length"
    spans, why = label_message(text, values)
    if spans is None:
        return None, why
    covered = bytearray(len(text))
    for s in spans:
        covered[s["start"] : s["end"]] = b"\x01" * (s["end"] - s["start"])
    ok_keep = []
    for k in dict.fromkeys(k for k in keep if isinstance(k, str) and k.strip() and k in text):
        i = text.find(k)
        if any(covered[i : i + len(k)]):
            continue  # a "look-alike" that overlaps a value is not a look-alike: drop it
        ok_keep.append(k)
    kept = bytearray(covered)
    for k in ok_keep:
        for i in _occurrences(text, k) or [text.find(k)]:
            kept[i : i + len(k)] = b"\x01" * len(k)

    def outside(m: re.Match) -> bool:
        return any(not kept[i] for i in range(m.start(), m.end()) if not text[i].isspace())

    for sp in spans:
        before = text[max(0, sp["start"] - 40) : sp["start"]]
        if sp["label"] == "TELEPHONENUM" and COMPANY_PHONE.search(before):
            return None, "phone given as the company's"
        if sp["label"] == "DATE" and HEADER_DATE.search(before):
            return None, "date used as a header date"
    for k in ok_keep:
        before = text[max(0, text.find(k) - 30) : text.find(k)]
        if (
            sum(c.isdigit() for c in k) >= 12
            and "gift" not in before.lower()
            and re.search(r"\bcard\b", before, re.IGNORECASE)
        ):
            return None, "look-alike presented as a card"
    if any(outside(m) for m in DIGIT_RUN.finditer(text) if sum(c.isdigit() for c in m[0]) >= 7):
        return None, "unlabelled digit run"
    if any(outside(m) for m in EMAILISH.finditer(text)):
        return None, "unlabelled email"
    if any(outside(m) for m in TITLE_NAME.finditer(text)):
        return None, "unlabelled title + name"
    return {"text": text, "spans": spans, "keep": ok_keep}, ""


# ---------------------------------------------------------------- contamination

EVAL_SETS = [Path("data/support_desk") / f"support_desk_{n}.jsonl"
             for n in ("val", "hard", "fresh", "300")]  # fmt: skip
NAME_LABELS = {"GIVENNAME", "SURNAME", "TITLE"}  # common names may recur; values may not


def word_ngrams(text: str, n: int = 8) -> set[tuple[str, ...]]:
    w = re.findall(r"\w+", text.lower())
    return {tuple(w[i : i + n]) for i in range(len(w) - n + 1)}


def eval_fingerprints(paths=EVAL_SETS) -> tuple[set, set]:
    """8-word sequences and non-name gold values of the support-desk evaluation sets."""
    grams, values = set(), set()
    for p in paths:
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            grams |= word_ngrams(r["text"])
            values |= {s["text"].lower() for s in r["spans"] if s["label"] not in NAME_LABELS}
    return grams, values


def contaminated(rec: dict, grams: set, values: set) -> bool:
    return bool(word_ngrams(rec["text"]) & grams) or any(
        s["text"].lower() in values for s in rec["spans"] if s["label"] not in NAME_LABELS
    )


# ---------------------------------------------------------------- main


def collect(resp: dict, specs: list[dict], reasons: Counter) -> list[dict]:
    try:
        content = json.loads(resp["choices"][0]["message"]["content"])
        msgs = {m.get("id"): m for m in content.get("messages", []) if isinstance(m, dict)}
    except (KeyError, IndexError, TypeError, json.JSONDecodeError):
        reasons["unparseable response"] += len(specs)
        return []
    out = []
    for s in specs:
        m = msgs.get(s["id"])
        if not m or not isinstance(m.get("text"), str):
            reasons["missing message"] += 1
            continue
        rec, why = validate(m["text"].strip(), s["values"], s["keep"] + (m.get("keep") or []))
        if rec is None:
            reasons[why] += 1
            continue
        rec["meta"] = {"source": "targeted", "pattern": s["pattern"], "channel": s["channel"],
                       "locale": s["locale"], "formats": [v["format"] for v in s["values"]],
                       "keep": rec.pop("keep"), "spec": s["id"]}  # fmt: skip
        out.append(rec)
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--n", type=int, default=2000, help="accepted messages to keep")
    ap.add_argument("--seed", type=int, default=2027)  # 2026 = the first pilot's prompt
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.1)
    ap.add_argument("--max-requests", type=int, default=800)
    ap.add_argument("--dry-run", action="store_true", help="print one request's prompt; no API")
    args = ap.parse_args(argv)
    if args.dry_run:
        print(SYSTEM, "\n\n", user_prompt(make_specs(args.seed, 0)))
        return
    key = api_key()
    RAW.mkdir(parents=True, exist_ok=True)
    accepted, reasons, usage = [], Counter(), Counter()
    grams, values = eval_fingerprints()
    req = 0
    with ThreadPoolExecutor(args.workers) as pool:
        while len(accepted) < args.n and req < args.max_requests:
            need = max(1, (args.n - len(accepted)) // PER_REQUEST + 1)
            batch = list(range(req, min(req + min(need, 4 * args.workers), args.max_requests)))
            req = batch[-1] + 1
            resps = list(pool.map(lambda r: fetch(key, args.seed, r, args.temperature), batch))
            if all(r is None for r in resps):
                sys.exit("every request in the batch failed; stopping (see errors above)")
            for r, resp in zip(batch, resps, strict=True):
                if resp is None:
                    continue
                for k, v in (resp.get("usage") or {}).items():
                    if isinstance(v, int):
                        usage[k] += v
                for rec in collect(resp, make_specs(args.seed, r), reasons):
                    if contaminated(rec, grams, values):
                        reasons["overlaps an evaluation set"] += 1
                    else:
                        accepted.append(rec)
            print(f"requests {req}, accepted {len(accepted)}, rejected {sum(reasons.values())}",
                  file=sys.stderr)  # fmt: skip
    accepted = accepted[: args.n]
    with open(OUT, "w", encoding="utf-8") as f:
        for i, rec in enumerate(accepted, 1):
            f.write(json.dumps({"id": f"tgt-{i:04d}", **rec}, ensure_ascii=False) + "\n")
    print(json.dumps({"accepted": len(accepted), "requests": req, "rejected": dict(reasons),
                      "patterns": Counter(r["meta"]["pattern"] for r in accepted),
                      "usage_tokens": dict(usage)}, indent=2))  # fmt: skip


if __name__ == "__main__":
    main()
