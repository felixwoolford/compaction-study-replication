"""Fact and probe generation.

Each fact binds a slot ("the staging bucket") to an unguessable value
("amber-lynx-42"). Fact vocabulary is disjoint from filler vocabulary
(filler.py), and stimuli.py asserts no value or slot leaks into filler.
"""

from __future__ import annotations

import random
import re
from dataclasses import asdict, dataclass, field

# Fact-only vocabulary. Filler must never use these words.
ADJ = [
    "amber", "brisk", "cobalt", "dusky", "ember", "fallow", "gilded", "hollow",
    "indigo", "jaunty", "kelp", "lucid", "mossy", "nimble", "ochre", "pallid",
    "quartz", "russet", "sable", "tawny", "umber", "vivid", "wistful", "zephyr",
]
ANIMAL = [
    "lynx", "heron", "marten", "ibis", "okapi", "tapir", "vole", "wren",
    "gecko", "egret", "dingo", "bison", "civet", "kudu", "lemur", "newt",
    "oriole", "puffin", "quokka", "serval", "tern", "urchin", "walrus", "yak",
]

# Slots: what a fact is about. `key` is used in config files for tool_output plants.
SLOTS = [
    ("staging bucket", "staging_bucket", "codename"),
    ("metrics port", "metrics_port", "port"),
    ("retry limit for the uploader", "uploader_retry_limit", "number"),
    ("exporter config path", "exporter_config_path", "filepath"),
    ("release codename", "release_codename", "codename"),
    ("cache directory", "cache_directory", "dirpath"),
    ("feature flag for the new importer", "importer_feature_flag", "identifier"),
    ("test fixture seed", "fixture_seed", "number"),
    ("webhook port", "webhook_port", "port"),
    ("queue name for background jobs", "job_queue_name", "codename"),
    ("helper function name for deduplication", "dedup_helper_name", "identifier"),
    ("batch size for the indexer", "indexer_batch_size", "number"),
    ("archive bucket", "archive_bucket", "codename"),
    ("admin port", "admin_port", "port"),
    ("lockfile path", "lockfile_path", "filepath"),
    ("migration label", "migration_label", "identifier"),
    ("timeout for the health check in seconds", "healthcheck_timeout_s", "number"),
    ("default branch name for the mirror", "mirror_branch", "codename"),
    ("schema version tag", "schema_version_tag", "identifier"),
    ("tracing service name", "tracing_service", "codename"),
    ("debug port", "debug_port", "port"),
    ("max workers for the scraper", "scraper_max_workers", "number"),
    ("snapshot directory", "snapshot_directory", "dirpath"),
    ("parser module name", "parser_module", "identifier"),
]

SOURCES = ("user", "assistant_text", "tool_output", "thinking", "superseded")

UNKNOWN_SOURCE = "lure"


@dataclass
class Fact:
    uid: str
    slot: str
    key: str
    kind: str
    source: str  # one of SOURCES, or "lure"
    position: str  # "early" | "late" | "none" (lures)
    value: str | None
    value_old: str | None = None  # superseded facts: the replaced value
    question: str = ""
    accept: str = ""  # regex, case-insensitive
    intrusion: str | None = None  # regex for value_old
    # filled in by stimuli.py: entry ids that carry the fact
    entry_ids: list[str] = field(default_factory=list)
    # 0: planted before the first compaction; r: planted in round r's work (Study 3
    # round details). A fact is probed only from the round it was planted in on.
    round_planted: int = 0

    @property
    def is_lure(self) -> bool:
        return self.source == UNKNOWN_SOURCE

    def to_dict(self) -> dict:
        return asdict(self)


def _value(rng: random.Random, kind: str) -> str:
    if kind == "codename":
        return f"{rng.choice(ADJ)}-{rng.choice(ANIMAL)}-{rng.randint(11, 98)}"
    if kind == "port":
        while True:
            p = rng.randint(20000, 59999)
            if p % 100:
                return str(p)
    if kind == "number":
        # Three digits, not round: filler only ever uses numbers below 100.
        while True:
            n = rng.randint(113, 987)
            if n % 5:
                return str(n)
    if kind == "filepath":
        return f"/srv/{rng.choice(ANIMAL)}/{rng.choice(ADJ)}-{rng.randint(2, 9)}.conf"
    if kind == "dirpath":
        return f"/srv/{rng.choice(ANIMAL)}/{rng.choice(ADJ)}-{rng.randint(2, 9)}"
    if kind == "identifier":
        return f"{rng.choice(ADJ)}_{rng.choice(ANIMAL)}_v{rng.randint(2, 9)}"
    raise ValueError(kind)


def value_regex(value: str) -> str:
    """Match the value as a whole token (so '37' does not match '137')."""
    return r"(?<![\w-])" + re.escape(value) + r"(?![\w-])"


def question_for(slot: str, source: str) -> str:
    if source == "thinking":
        return f"What did you pick for the {slot}?"
    return f"What is the {slot}?"


def make_facts(rng: random.Random, session_id: str, facts_per_source: dict[str, int],
               n_lures: int, n_late: int) -> list[Fact]:
    """Draw slots without replacement and assign sources, values and probes."""
    n_needed = sum(facts_per_source.values()) + n_lures + n_late
    if n_needed > len(SLOTS):
        raise ValueError(f"need {n_needed} slots, only {len(SLOTS)} defined")
    slots = rng.sample(SLOTS, n_needed)
    used_values: set[str] = set()

    def fresh(kind: str) -> str:
        while True:
            v = _value(rng, kind)
            if v not in used_values:
                used_values.add(v)
                return v

    facts: list[Fact] = []
    it = iter(slots)
    for source in SOURCES:
        for _ in range(facts_per_source.get(source, 0)):
            slot, key, kind = next(it)
            v = fresh(kind)
            f = Fact(uid=f"{session_id}:{key}", slot=slot, key=key, kind=kind, source=source,
                     position="early", value=v, question=question_for(slot, source),
                     accept=value_regex(v))
            if source == "superseded":
                f.value_old = fresh(kind)
                f.intrusion = value_regex(f.value_old)
            facts.append(f)
    for i in range(n_late):
        slot, key, kind = next(it)
        v = fresh(kind)
        source = ("user", "tool_output")[i % 2]
        facts.append(Fact(uid=f"{session_id}:{key}", slot=slot, key=key, kind=kind, source=source,
                          position="late", value=v, question=question_for(slot, source),
                          accept=value_regex(v)))
    for _ in range(n_lures):
        slot, key, kind = next(it)
        facts.append(Fact(uid=f"{session_id}:{key}", slot=slot, key=key, kind=kind,
                          source=UNKNOWN_SOURCE, position="none", value=None,
                          question=question_for(slot, "user")))
    return facts


def forbidden_filler_tokens() -> set[str]:
    """Tokens filler must never contain: the value vocabulary and every slot key.

    Full slot phrases ("staging bucket") are checked per session in stimuli.py;
    ordinary words like "config" or "test" stay allowed so filler can look real.
    """
    return set(ADJ) | set(ANIMAL) | {key for _, key, _ in SLOTS}
