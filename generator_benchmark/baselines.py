"""Answer baselines that need no model: how well a memoriser would do on each test level.

most_frequent: always write the most common training answer (of the same relation when known).
nearest: write the answer of the most similar training question (word overlap).
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

from . import tasks


def norm(text: str, casefold: bool = True) -> str:
    out = re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))
    return out.casefold() if casefold else out


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.casefold()))


def evaluate(train: list[dict], test: list[dict]) -> dict[str, dict[str, float]]:
    overall = Counter(i["target"] for i in train).most_common(1)[0][0]
    by_relation: dict[str, Counter] = {}
    for item in train:
        relation = item["fact"].get("relation")
        if relation:
            by_relation.setdefault(relation, Counter())[item["target"]] += 1
    train_tokens = [(_tokens(i["card_text"]), i["target"]) for i in train]
    out: dict[str, dict[str, list[int]]] = {}
    for item in test:
        relation = item["fact"].get("relation")
        guess = by_relation[relation].most_common(1)[0][0] if relation in by_relation and item["level"] != "L2R" else overall
        tokens = _tokens(item["card_text"])
        nearest = max(train_tokens, key=lambda t: len(tokens & t[0]) / max(1, len(tokens | t[0])))[1]
        expected = norm(item["expected"], item["casefold"])
        row = out.setdefault(item["level"], {"n": [], "most_frequent": [], "nearest": []})
        row["n"].append(1)
        row["most_frequent"].append(int(norm(guess, item["casefold"]) == expected))
        row["nearest"].append(int(norm(nearest, item["casefold"]) == expected))
    return {level: {"n": len(r["n"]), "most_frequent": sum(r["most_frequent"]) / len(r["n"]),
                    "nearest": sum(r["nearest"]) / len(r["n"])} for level, r in sorted(out.items())}


def all_tasks() -> dict[str, dict]:
    data = tasks.build_all()
    return {name: evaluate(d["train"], d["test"]) for name, d in data.items() if name != "select"}


if __name__ == "__main__":
    import json
    print(json.dumps(all_tasks(), indent=1))
