"""Rule-based scoring of probe answers (blind to arm by construction).

Categories (PROTOCOL.md §7):
  planted facts: correct | intrusion | incorrect | abstain | needs_review
  lures:         correct_reject | confabulation | needs_review
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

ANSWER_RE = re.compile(r"^\s*\**\s*A\s*(\d+)\s*\**\s*[:.)\-]\s*\**\s*(.*?)\s*$", re.IGNORECASE)
ABSTAIN_RE = re.compile(
    r"\b(unknown|not sure|unsure|don'?t know|do not know|no idea|cannot (recall|verify|tell)|can'?t (recall|verify|tell)"
    r"|(was|were)\s*n[o']t (discussed|mentioned|specified|stated)|not (discussed|mentioned|specified|stated)"
    r"|no (record|mention|information)"
    r"|never (?:\w+ ){0,3}(?:discussed|mentioned|set|added|chosen|specified|stated)|^\s*none)\b",
    re.IGNORECASE,
)
HEDGE_RE = re.compile(r"\b(unverified|i think|probably|possibly|likely|not verified)\b", re.IGNORECASE)

CORRECT = {"correct", "correct_reject"}
CONFAB = {"incorrect", "intrusion", "confabulation"}


def parse_answers(text: str) -> dict[int, str]:
    """Map question number -> answer text from lines like 'A3: amber-lynx-42'."""
    out: dict[int, str] = {}
    for line in (text or "").splitlines():
        m = ANSWER_RE.match(line)
        if m:
            out[int(m.group(1))] = m.group(2).strip().strip("`").strip()
    return out


def _ungroup(a: str) -> str:
    """Remove thousands separators inside numbers (14,336 / 14 336 / 14 336 -> 14336)."""
    return re.sub(r"(?<=\d)[,\u00a0\u202f ](?=\d{3}(?!\d))", "", a)


def categorize(trial: dict) -> str:
    ans = trial.get("answer")
    if ans is None or not str(ans).strip():
        return "needs_review"
    a = str(ans)
    if trial.get("is_lure"):
        return "correct_reject" if ABSTAIN_RE.search(a) else "confabulation"
    if re.search(trial["accept"], a, re.IGNORECASE):
        return "correct"
    v = str(trial.get("value") or "")
    if v.isdigit() and re.search(trial["accept"], _ungroup(a), re.IGNORECASE):
        return "correct"  # a number written with thousands separators: 14,336 for 14336
    if v.endswith("/") and len(v) > 1 and re.search(r"(?<![\w-])" + re.escape(v.rstrip("/")) + r"(?![\w-])", a, re.IGNORECASE):
        return "correct"  # a path or bucket given without its trailing slash
    m = re.fullmatch(r"(pid|oid)-(\d+)", v)
    if m and re.search(r"(?<![\w-])" + m.group(2) + r"(?![\w-])", a):
        return "correct"  # a pid/oid given as its bare number
    if trial.get("intrusion") and (re.search(trial["intrusion"], a, re.IGNORECASE)
                                   or v.isdigit() and re.search(trial["intrusion"], _ungroup(a), re.IGNORECASE)):
        return "intrusion"
    if ABSTAIN_RE.search(a):
        return "abstain"
    return "incorrect"


def score_trial(trial: dict, override: str | None = None) -> dict:
    cat = override or categorize(trial)
    t = dict(trial)
    t["category"] = cat
    t["correct"] = None if cat == "needs_review" else int(cat in CORRECT)
    t["confab"] = None if cat == "needs_review" else int(cat in CONFAB)
    t["abstain"] = None if cat == "needs_review" else int(cat in ("abstain", "correct_reject"))
    t["hedged"] = int(cat == "correct" and bool(HEDGE_RE.search(str(trial.get("answer") or ""))))
    # Confident wrong (PROTOCOL.md §7): a wrong planted-fact answer (incorrect or intrusion)
    # with no `(unverified)`-style hedge marker.
    t["confident_wrong"] = None if cat == "needs_review" else int(
        cat in ("incorrect", "intrusion") and not HEDGE_RE.search(str(trial.get("answer") or "")))
    return t


def _trial_key(t: dict) -> str:
    return "|".join(str(t[k]) for k in ("session_id", "arm", "round", "rep", "probe_mode", "fact_uid"))


def _item_id(t: dict) -> str:
    """Opaque review id: reveals neither arm nor session to the human coder."""
    return hashlib.sha1(_trial_key(t).encode()).hexdigest()[:10]


def score_file(trials_path: Path, out_path: Path, review_csv: Path, coded_csv: Path | None = None) -> dict:
    """Score all trials. Human codes from a filled-in review CSV override `needs_review`."""
    overrides: dict[str, str] = {}
    if coded_csv and coded_csv.exists():
        with coded_csv.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("category"):
                    overrides[row["item_id"]] = row["category"].strip()
    lines = [json.loads(x) for x in trials_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    scored = [score_trial(t, overrides.get(_item_id(t))) for t in lines]
    out_path.write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in scored), encoding="utf-8")
    review = [s for s in scored if s["category"] == "needs_review"]
    # Blind review sheet: no arm, no session; the item, the expected answer and the raw reply.
    # Fill in `category` with one of the category names, save as review_coded.csv.
    with review_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "question_number", "question", "is_lure", "expected", "superseded_old",
                    "reply", "category"])
        for s in sorted(review, key=_item_id):
            w.writerow([_item_id(s), s.get("question_number"), s["question"], s["is_lure"],
                        s.get("value") or "UNKNOWN", s.get("value_old") or "", s.get("raw_text") or "", ""])
    counts: dict[str, int] = {}
    for s in scored:
        counts[s["category"]] = counts.get(s["category"], 0) + 1
    return counts
