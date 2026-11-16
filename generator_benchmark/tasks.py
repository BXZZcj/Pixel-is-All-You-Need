"""Benchmark items and splits. Items are plain dicts; canvases are rendered from them on demand (render.py).

Item fields:
  item_id, task, split ("train"/"test"), level, mode ("write"/"select"), card_text,
  prefix (answer text already on the input card), target (answer text on the target card),
  expected (string the judge's transcription must match), casefold (bool),
  options, correct_index (select only), fact (provenance: relation, entity, answer), original_prefix (transition)

Test levels: copy; L1 paraphrase; L2R reverse; L2F related fact; L3 two-hop; L4 untrained fact;
L5A in-card reasoning (trained families, new names); L5B in-card reasoning (never-trained family);
A1 arithmetic new expressions; A2 never-trained arithmetic form; I1 instruction new words; I2 never-trained rule.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from typing import Any, Iterable

from . import cards, kb

SEED = 20261030
HOLDOUT_FRACTION = 0.25
GENERIC_PROMPT = "Complete the task shown on the card. Keep everything else on the card unchanged."

QA_TEMPLATES = {
    "capital": (["What is the capital of {e}?", "Which city is the capital of {e}?"],
                ["Name the capital city of {e}."]),
    "continent": (["On which continent is {e}?", "{E} is located on which continent?"],
                  ["Name the continent where {e} is found."]),
    "language": (["What is the main language of {e}?", "Which language do most people in {e} speak?"],
                 ["People in {e} mostly speak which language?"]),
    "symbol": (["What is the chemical symbol of {e}?", "Which symbol stands for {e} in chemistry?"],
               ["In the periodic table, what is the symbol for {e}?"]),
    "author": (["Who wrote {e}?", "Who is the author of {e}?"], ["{E} was written by which author?"]),
    "antonym": (["What is the opposite of {e}?", "Give the antonym of the word {e}."],
                ["Which word means the opposite of {e}?"]),
}
REVERSE_TEMPLATES = {
    "capital": "{a} is the capital of which country?",
    "symbol": "Which chemical element has the symbol {a}?",
    "antonym": "What is the opposite of {a}?",
}
TWO_HOP = {
    "continent": "On which continent is the city of {cap}?",
    "language": "Which language do most people in the city of {cap} speak?",
}
COPY_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
COPY_COMPLETE_TEXT = "Copy this text into the answer box:\n{s}"
COPY_TRANSITION_TEXT = "Continue copying this text into the answer box, adding exactly one more character:\n{s}"
SELECT_INSTRUCTION = "\nDraw a red rectangle around the correct option."


def _cap(text: str) -> str:
    return text[0].upper() + text[1:]


def _item_id(task: str, level: str, card_text: str, prefix: str, mode: str) -> str:
    digest = hashlib.sha256(json.dumps([task, level, card_text, prefix, mode]).encode()).hexdigest()[:20]
    return f"e5x_{digest}"   # the prefix is part of the frozen ids (and so of every per-item seed)


def make_item(*, task: str, split: str, level: str, card_text: str, target: str, expected: str | None = None,
              prefix: str = "", casefold: bool = True, fact: dict | None = None, mode: str = "write",
              options: list[str] | None = None, correct_index: int | None = None,
              original_prefix: str = "") -> dict[str, Any]:
    return {"item_id": _item_id(task, level, card_text, prefix, mode), "task": task, "split": split, "level": level,
            "mode": mode, "card_text": card_text, "prefix": prefix, "target": target,
            "expected": target if expected is None else expected, "casefold": casefold, "options": options,
            "correct_index": correct_index, "fact": fact or {}, "original_prefix": original_prefix}


def mentions(text: str, name: str) -> bool:
    """Whole-word, case-insensitive occurrence ("the " prefixes on entity names are ignored)."""
    name = name[4:] if name.startswith("the ") else name
    return re.search(rf"(?<![A-Za-z]){re.escape(name)}(?![A-Za-z])", text, re.IGNORECASE) is not None


RELATED_TEXT = " | ".join(f"{q} {a}" for q, a, _ in kb.RELATED)


def holdout_split(values: list, rng: random.Random) -> tuple[list, list]:
    """Hold out a fraction of the rows; a row whose entity or answer occurs in the related-fact training questions
    can never be held out, since it would then appear in training."""
    protected = [v for v in values if any(isinstance(x, str) and mentions(RELATED_TEXT, x) for x in v)]
    free = [v for v in values if v not in protected]
    rng.shuffle(free)
    k = max(1, round(len(values) * HOLDOUT_FRACTION))
    return free[k:] + protected, free[:k]


def build_facts(seed: int = SEED) -> dict[str, list[dict]]:
    """Facts by relation; `held` marks entities that never appear in training."""
    rng = random.Random(seed)
    train_c, held_c = holdout_split(kb.COUNTRIES, rng)
    train_el, held_el = holdout_split(kb.ELEMENTS, rng)
    train_b, held_b = holdout_split(kb.BOOKS, rng)
    train_a, held_a = holdout_split(kb.ANTONYMS, rng)
    facts: dict[str, list[dict]] = {k: [] for k in QA_TEMPLATES}
    for held, rows in ((False, train_c), (True, held_c)):
        for country, capital, continent, language in rows:
            facts["capital"].append({"entity": country, "answer": capital, "held": held, "country": country})
            facts["continent"].append({"entity": country, "answer": continent, "held": held, "country": country})
            if language:
                facts["language"].append({"entity": country, "answer": language, "held": held, "country": country})
    for held, rows in ((False, train_el), (True, held_el)):
        for element, symbol in rows:
            facts["symbol"].append({"entity": element, "answer": symbol, "held": held})
    for held, rows in ((False, train_b), (True, held_b)):
        for book, author in rows:
            facts["author"].append({"entity": book, "answer": author, "held": held})
    for held, rows in ((False, train_a), (True, held_a)):
        for word, antonym in rows:
            facts["antonym"].append({"entity": word, "answer": antonym, "held": held})
    return facts


def qa_items(seed: int = SEED) -> tuple[list[dict], list[dict]]:
    facts = build_facts(seed)
    train, test = [], []
    capital_of = {f["country"]: f["answer"] for f in facts["capital"]}
    for relation, rows in facts.items():
        train_t, para_t = QA_TEMPLATES[relation]
        for f in rows:
            meta = {"relation": relation, "entity": f["entity"], "answer": f["answer"]}
            def card(t):
                return t.format(e=f["entity"], E=_cap(f["entity"]))
            if f["held"]:
                test.append(make_item(task="qa", split="test", level="L4", card_text=card(train_t[0]),
                                      target=f["answer"], fact=meta))
                continue
            for t in train_t:
                train.append(make_item(task="qa", split="train", level="train", card_text=card(t),
                                       target=f["answer"], fact=meta))
            test.append(make_item(task="qa", split="test", level="L1", card_text=card(para_t[0]),
                                  target=f["answer"], fact=meta))
            if relation in REVERSE_TEMPLATES:
                test.append(make_item(task="qa", split="test", level="L2R",
                                      card_text=REVERSE_TEMPLATES[relation].format(a=f["answer"]),
                                      target=f["entity"], fact={**meta, "answer": f["entity"], "entity": f["answer"]}))
            if relation in TWO_HOP:
                test.append(make_item(task="qa", split="test", level="L3",
                                      card_text=TWO_HOP[relation].format(cap=capital_of[f["country"]]),
                                      target=f["answer"], fact={**meta, "via_capital": capital_of[f["country"]]}))
    for question, answer, related in kb.RELATED:
        train.append(make_item(task="qa", split="train", level="train", card_text=question, target=answer,
                               fact={"relation": "related", "answer": answer}))
        test.append(make_item(task="qa", split="test", level="L2F", card_text=related, target=answer,
                              fact={"relation": "related", "answer": answer}))
    return train, test


BIG_TEMPLATES = {  # train-only extra phrasings for qa_big; none equals a test template (checked in validate)
    "capital": ["Capital of {e}?", "Q: What is the capital city of {e}? A:", "Tell me the capital of {e}.",
                "Which city is the seat of government of {e}?", "The capital of {e} is which city?",
                "What city serves as the capital of {e}?", "Write the capital of {e}.", "{E}: capital city?"],
    "continent": ["Which continent is {e} on?", "Q: On what continent is {e}? A:", "Tell me the continent of {e}.",
                  "{E} belongs to which continent?", "Write the continent of {e}.", "{E}: which continent?",
                  "What continent contains {e}?", "{E} lies on which continent?"],
    "language": ["Which language is spoken most in {e}?", "Q: What language do people in {e} speak? A:",
                 "Tell me the main language of {e}.", "Write the main language of {e}.", "{E}: main language?",
                 "What is the most common language in {e}?", "Most people in {e} speak what language?",
                 "The main language of {e} is what?"],
    "symbol": ["Chemical symbol of {e}?", "Q: What is the symbol for {e}? A:", "Tell me the chemical symbol of {e}.",
               "Write the chemical symbol of {e}.", "{E}: chemical symbol?", "How is {e} written as a chemical symbol?",
               "What letters stand for {e}?", "The chemical symbol of {e} is what?"],
    "author": ["Author of {e}?", "Q: Who wrote the book {e}? A:", "Tell me who wrote {e}.", "Write the author of {e}.",
               "{E}: who is the author?", "Which writer created {e}?", "{E} is a book by whom?",
               "The author of {e} is who?"],
    "antonym": ["Opposite of {e}?", "Q: What is the antonym of {e}? A:", "Tell me the opposite of {e}.",
                "Write the opposite of {e}.", "{E}: opposite word?", "What word is the reverse of {e}?",
                "What is the contrary of {e}?", "The opposite of {e} is what?"],
}
EXTRA_RELATIONS = {  # relation: (table, templates)
    "month_after": ("MONTH_AFTER", ["Which month comes after {e}?", "What month follows {e}?", "Month after {e}?",
                                    "Q: Which month is after {e}? A:", "After {e} comes which month?",
                                    "Write the month that follows {e}."]),
    "day_after": ("DAY_AFTER", ["Which day comes after {e}?", "What day follows {e}?", "Day after {e}?",
                                "Q: Which day is after {e}? A:", "After {e} comes which day?",
                                "Write the day that follows {e}."]),
    "number_word": ("NUMBER_WORDS", ["Write the number {e} as a word.", "How do you spell {e} in words?",
                                     "Number {e} in words?", "Q: What is {e} written as a word? A:",
                                     "Spell out the number {e}.", "Write {e} in letters."]),
    "past_tense": ("PAST_TENSE", ["What is the past tense of {e}?", "Past tense of {e}?", "Write the past tense of {e}.",
                                  "Q: What is the past form of {e}? A:", "Yesterday I ___ (to {e}). Fill in the verb.",
                                  "Give the simple past of {e}."]),
    "plural": ("PLURALS", ["What is the plural of {e}?", "Plural of {e}?", "Write the plural of {e}.",
                           "Q: What is more than one {e} called? A:", "One {e}, two what?", "Give the plural form of {e}."]),
    "animal_young": ("ANIMAL_YOUNG", ["What is a baby {e} called?", "Baby {e}?", "Write the name of a young {e}.",
                                      "Q: What do you call a young {e}? A:", "A young {e} is called what?",
                                      "Name the baby of a {e}."]),
    "animal_sound": ("ANIMAL_SOUND", ["What sound does a {e} make?", "Sound of a {e}?", "Write the sound a {e} makes.",
                                      "Q: What noise does a {e} make? A:", "A {e} says what?",
                                      "Name the sound of a {e}."]),
}


def qa_big_items(seed: int = SEED) -> tuple[list[dict], list[dict]]:
    """QA training enlarged to ~3k items (more phrasings + seven extra relations). Test = only the L1 paraphrase of
    each extra-relation fact (level L1X); evaluate qa_big arms on tasks "qa qa_big" to get the original QA test too."""
    base_train, base_test = qa_items(seed)
    facts = build_facts(seed)
    train = list(base_train)
    for relation, rows in facts.items():
        for f in rows:
            if f["held"]:
                continue
            meta = {"relation": relation, "entity": f["entity"], "answer": f["answer"]}
            for t in BIG_TEMPLATES[relation]:
                train.append(make_item(task="qa_big", split="train", level="train",
                                       card_text=t.format(e=f["entity"], E=_cap(f["entity"])), target=f["answer"], fact=meta))
    test = []
    for relation, (table, templates) in EXTRA_RELATIONS.items():
        for entity, answer in getattr(kb, table):
            meta = {"relation": relation, "entity": entity, "answer": answer}
            for t in templates[:-1]:
                train.append(make_item(task="qa_big", split="train", level="train",
                                       card_text=t.format(e=entity, E=_cap(entity)), target=answer, fact=meta))
            test.append(make_item(task="qa_big", split="test", level="L1X",
                                  card_text=templates[-1].format(e=entity, E=_cap(entity)), target=answer, fact=meta))
    return train, test


def copy_items(n_train: int = 10000, n_test: int = 200, seed: int = SEED) -> dict[str, list[dict]]:
    rng = random.Random(seed + 1)
    strings: set[str] = set()
    while len(strings) < n_train + n_test:
        strings.add("".join(rng.choice(COPY_ALPHABET) for _ in range(rng.choice((4, 5, 6, 7, 8)))))
    ordered = sorted(strings)
    rng.shuffle(ordered)
    test_s, train_s = ordered[:n_test], ordered[n_test:]
    out: dict[str, list[dict]] = {"copy_complete": [], "copy_transition": []}
    for split, values in (("train", train_s), ("test", test_s)):
        level = "train" if split == "train" else "copy"
        for s in values:
            out["copy_complete"].append(make_item(task="copy_complete", split=split, level=level,
                                                  card_text=COPY_COMPLETE_TEXT.format(s=s), target=s, casefold=False))
            k = int(hashlib.sha256(s.encode()).hexdigest(), 16) % len(s)
            out["copy_transition"].append(make_item(task="copy_transition", split=split, level=level,
                                                    card_text=COPY_TRANSITION_TEXT.format(s=s), prefix=s[:k],
                                                    target=s[:k + 1], casefold=False, original_prefix=s[:k]))
    return out


def arithmetic_items(n_train: int = 1000, n_test: int = 200, seed: int = SEED) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed + 2)

    def sample() -> tuple[str, int]:
        kind = rng.choice(("add", "sub", "mul", "paren"))
        if kind == "add":
            a, b = rng.randint(10, 99), rng.randint(10, 99)
            return f"What is {a} + {b}?", a + b
        if kind == "sub":
            a, b = sorted((rng.randint(10, 99), rng.randint(10, 99)), reverse=True)
            return f"What is {a} - {b}?", a - b
        if kind == "mul":
            a, b = rng.randint(2, 19), rng.randint(2, 9)
            return f"What is {a} x {b}?", a * b
        a, b, c = rng.randint(2, 15), rng.randint(2, 15), rng.randint(2, 9)
        return f"What is ({a} + {b}) x {c}?", (a + b) * c

    seen: dict[str, int] = {}
    while len(seen) < n_train + n_test:
        question, value = sample()
        seen.setdefault(question, value)
    questions = list(seen)
    rng.shuffle(questions)
    train = [make_item(task="arithmetic", split="train", level="train", card_text=q, target=str(seen[q]))
             for q in questions[n_test:]]
    test = [make_item(task="arithmetic", split="test", level="A1", card_text=q, target=str(seen[q]))
            for q in questions[:n_test]]
    held: dict[str, int] = {}
    while len(held) < n_test // 2:  # never-trained form: three-term sums
        a, b, c = rng.randint(10, 49), rng.randint(10, 49), rng.randint(10, 49)
        held[f"What is {a} + {b} + {c}?"] = a + b + c
    test += [make_item(task="arithmetic", split="test", level="A2", card_text=q, target=str(v)) for q, v in held.items()]
    return train, test


INSTRUCTION_RULES = {
    "reverse": ("Write the word {w} backwards.", lambda w: w[::-1]),
    "first": ("Write the first letter of the word {w}.", lambda w: w[0]),
    "last": ("Write the last letter of the word {w}.", lambda w: w[-1]),
    "upper": ("Write the word {w} in capital letters.", lambda w: w.upper()),
    "twice": ("Write the word {w} two times with no space.", lambda w: w + w),
}
HELD_RULE = ("Write the word {w} without its first letter.", lambda w: w[1:])


def instruction_items(seed: int = SEED) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed + 3)
    words = sorted(set(w for w in kb.WORDS if len(w) * 2 <= 21))
    rng.shuffle(words)
    k = max(1, round(len(words) * HOLDOUT_FRACTION))
    test_w, train_w = words[:k], words[k:]
    train, test = [], []
    for name, (template, fn) in INSTRUCTION_RULES.items():
        casefold = name != "upper"
        for w in train_w:
            train.append(make_item(task="instruction", split="train", level="train", card_text=template.format(w=w),
                                   target=fn(w), casefold=casefold, fact={"rule": name}))
        for w in test_w:
            test.append(make_item(task="instruction", split="test", level="I1", card_text=template.format(w=w),
                                  target=fn(w), casefold=casefold, fact={"rule": name}))
    template, fn = HELD_RULE
    for w in test_w:
        test.append(make_item(task="instruction", split="test", level="I2", card_text=template.format(w=w),
                              target=fn(w), fact={"rule": "drop_first"}))
    return train, test


def _context_sample(family: str, names: list[str], rng: random.Random) -> tuple[str, str]:
    if family == "compare":
        a, b, c = rng.sample(names, 3)
        adj, top, bottom = rng.choice(kb.ADJECTIVES)
        if rng.random() < 0.5:
            return f"{a} is {adj} than {b}. {b} is {adj} than {c}. Who is the {top}?", a
        return f"{a} is {adj} than {b}. {b} is {adj} than {c}. Who is the {bottom}?", c
    if family == "attribute":
        a, b = rng.sample(names, 2)
        c1, c2 = rng.sample(kb.COLORS, 2)
        thing = rng.choice(kb.OBJECTS)
        if rng.random() < 0.5:
            return f"{a} has a {c1} {thing}. {b} has a {c2} {thing}. What color is the {thing} of {a}?", c1
        return f"{a} has a {c1} {thing}. {b} has a {c2} {thing}. What color is the {thing} of {b}?", c2
    if family == "count":
        a, b = rng.sample(names, 2)
        x, y = rng.randint(2, 9), rng.randint(2, 9)
        thing = rng.choice(("apples", "coins", "stamps", "marbles"))
        return f"{a} has {x} {thing}. {b} gives {a} {y} more {thing}. How many {thing} does {a} have now?", str(x + y)
    if family == "location":  # never trained
        a, b = rng.sample(names, 2)
        r1, r2, r3 = rng.sample(kb.ROOMS, 3)
        return f"{a} is in the {r1}. {b} is in the {r2}. {a} walks to the {r3}. Where is {a} now?", r3
    raise ValueError(family)


def context_items(n_train: int = 1000, n_test: int = 200, seed: int = SEED) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed + 4)
    names = list(kb.NAMES)
    rng.shuffle(names)
    k = max(3, round(len(names) * HOLDOUT_FRACTION))
    test_names, train_names = names[:k], names[k:]

    def collect(pool, families, n, level, split):
        rows: dict[str, str] = {}
        while len(rows) < n:
            family = rng.choice(families)
            question, answer = _context_sample(family, pool, rng)
            rows.setdefault(question, answer)
        return [make_item(task="context", split=split, level=level, card_text=q, target=a) for q, a in rows.items()]

    trained = ("compare", "attribute", "count")
    train = collect(train_names, trained, n_train, "train", "train")
    test = collect(test_names, trained, n_test, "L5A", "test") + collect(test_names, ("location",), n_test // 2, "L5B", "test")
    return train, test


def to_select(item: dict, pool: Iterable[str], rng: random.Random) -> dict:
    """Same question as a three-option select item; distractors are other answers of the same type."""
    target = item["target"]
    fold = str.casefold if item["casefold"] else str
    candidates = sorted({p for p in pool if fold(p) != fold(target) and cards.option_fits(p)})
    if target.isdigit():
        value = int(target)
        candidates = sorted({str(value + d) for d in (-10, -2, -1, 1, 2, 10) if value + d >= 0})
    if len(candidates) < 2 or not cards.option_fits(target):
        raise ValueError(f"no distractors for {target!r}")
    options = rng.sample(candidates, 2)
    correct = rng.randrange(3)
    options.insert(correct, target)
    return make_item(task="select", split=item["split"], level=item["level"],
                     card_text=item["card_text"] + SELECT_INSTRUCTION, target=target, casefold=item["casefold"],
                     fact=item["fact"], mode="select", options=options, correct_index=correct)


def answer_type(item: dict) -> str:
    """Distractors share the answer type; a reverse question's answer is the relation's entity, not its answer."""
    relation = item["fact"].get("relation", "qa")
    return f"{relation}_reverse" if item["level"] == "L2R" else relation


def select_items(qa_train, qa_test, arith_train, arith_test, ctx_train, ctx_test, n_train: int = 1000,
                 seed: int = SEED) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed + 5)
    by_relation: dict[str, set[str]] = {}
    for it in qa_train + qa_test:
        by_relation.setdefault(answer_type(it), set()).add(it["target"])
    context_pool = {it["target"] for it in ctx_train + ctx_test}

    def convert(items, pool_of):
        out = []
        for it in items:
            try:
                out.append(to_select(it, pool_of(it), rng))
            except ValueError:
                continue
        return out

    qa_pool = lambda it: by_relation.get(answer_type(it), set())
    ctx_pool = lambda it: context_pool
    num_pool = lambda it: []
    train = convert(qa_train, qa_pool) + convert(arith_train, num_pool) + convert(ctx_train, ctx_pool)
    rng.shuffle(train)
    train = train[:n_train]
    test = convert(qa_test, qa_pool) + convert(arith_test, num_pool) + convert(ctx_test, ctx_pool)
    return train, test


SHAPES = ("circle", "square", "triangle", "star")
COLORS6 = ("red", "blue", "green", "yellow", "purple", "orange")


def _scene(rng: random.Random, n: int, *, sizes: bool = False) -> list[dict]:
    """n non-overlapping shapes with distinct colours, spread left to right in the picture area."""
    from .cards import SCENE_BOX
    x0, y0, x1, y1 = SCENE_BOX
    slots = sorted(rng.sample(range(n * 2), n))  # columns, so left/right is unambiguous
    width = (x1 - x0) / (n * 2)
    colors = rng.sample(COLORS6, n)
    out = []
    for slot, color in zip(slots, colors):
        r = rng.choice((28, 40, 56)) if sizes else 44
        cx = int(x0 + width * (slot + 0.5))
        cy = int(rng.uniform(y0 + 60, y1 - 60))
        out.append({"shape": rng.choice(SHAPES), "color": color, "x": cx, "y": cy, "r": r})
    return out


def _visual_sample(family: str, rng: random.Random, names: list[str]):
    """Returns (question, answer, scene) or None when the scene does not make the question unambiguous."""
    if family == "color":
        scene = _scene(rng, rng.randint(2, 4))
        shapes = [o["shape"] for o in scene]
        target = rng.choice(scene)
        if shapes.count(target["shape"]) != 1:
            return None
        return f"What color is the {target['shape']} in the picture?", target["color"], scene
    if family == "shape":
        scene = _scene(rng, rng.randint(2, 4))
        target = rng.choice(scene)
        return f"What shape is {target['color']} in the picture?", target["shape"], scene
    if family == "count":
        n = rng.randint(2, 6)
        scene = _scene(rng, n)
        shape = rng.choice(SHAPES)
        k = sum(o["shape"] == shape for o in scene)
        return f"How many {shape}s are in the picture?", str(k), scene
    if family == "relation":
        scene = _scene(rng, 3)
        i = rng.randrange(3)
        if rng.random() < 0.5 and i > 0:
            return f"What color is the shape just left of the {scene[i]['color']} one?", scene[i - 1]["color"], scene
        if i < 2:
            return f"What color is the shape just right of the {scene[i]['color']} one?", scene[i + 1]["color"], scene
        return None
    if family == "owner":  # interleaved: the binding is in the text, the property in the picture
        scene = _scene(rng, 3)
        a, b, c = rng.sample(names, 3)
        owners = dict(zip((a, b, c), scene))
        who = rng.choice((a, b, c))
        text = " ".join(f"{n} owns the {o['color']} shape." for n, o in owners.items())
        return f"{text} What shape does {who} own?", owners[who]["shape"], scene
    if family == "largest":  # never trained
        scene = _scene(rng, 3, sizes=True)
        radii = sorted(o["r"] for o in scene)
        if radii[-1] == radii[-2]:
            return None
        return "What color is the largest shape in the picture?", max(scene, key=lambda o: o["r"])["color"], scene
    raise ValueError(family)


def visual_items(n_train: int = 1000, n_test: int = 200, seed: int = SEED) -> tuple[list[dict], list[dict]]:
    """Picture-based questions: the information is drawn on the card, not written. Trained families: color, shape,
    count, relation, owner (text+picture). Test: L6A trained families, new scenes; L6I owner with unseen names;
    L6B 'largest' (never trained)."""
    rng = random.Random(seed + 7)
    names = list(kb.NAMES)
    rng.shuffle(names)
    k = max(3, round(len(names) * HOLDOUT_FRACTION))
    test_names, train_names = names[:k], names[k:]

    def collect(families, n, level, split, pool):
        out, seen = [], set()
        while len(out) < n:
            sample = _visual_sample(rng.choice(families), rng, pool)
            if sample is None:
                continue
            question, answer, scene = sample
            key = (question, json.dumps(scene))
            if key in seen:
                continue
            seen.add(key)
            item = make_item(task="visual", split=split, level=level, card_text=question, target=answer,
                             fact={"family": next(f for f in ("owner", "largest", "relation", "count", "shape", "color")
                                                  if (f == "owner" and "owns" in question) or (f == "largest" and "largest" in question)
                                                  or (f == "relation" and "just" in question) or (f == "count" and "How many" in question)
                                                  or (f == "shape" and question.startswith("What shape is")) or f == "color")})
            item["scene"] = scene
            item["item_id"] = _item_id("visual", level, question + json.dumps(scene), "", "write")
            out.append(item)
        return out

    trained = ("color", "shape", "count", "relation")
    train = collect(trained + ("owner",), n_train, "train", "train", train_names)
    test = (collect(trained, n_test, "L6A", "test", train_names) + collect(("owner",), n_test // 2, "L6I", "test", test_names)
            + collect(("largest",), n_test // 2, "L6B", "test", train_names))
    return train, test


def select_qa_items(qa_train, qa_test, seed: int = SEED) -> tuple[list[dict], list[dict]]:
    """QA only, as three-option select items (no rendering burden); same questions as the write QA arm."""
    rng = random.Random(seed + 6)
    pools: dict[str, set[str]] = {}
    for it in qa_train + qa_test:
        pools.setdefault(answer_type(it), set()).add(it["target"])
    out = []
    for items in (qa_train, qa_test):
        converted = []
        for it in items:
            try:
                converted.append({**to_select(it, pools[answer_type(it)], rng), "task": "select_qa"})
            except ValueError:
                continue
        out.append(converted)
    return out[0], out[1]


RENDER_INSTRUCTION = "\nWrite this answer in the answer box: {answer}"


def item_seed(item_id: str) -> int:
    """The generation seed of an item (the same for every model and condition)."""
    return int(hashlib.sha256(item_id.encode()).hexdigest()[:8], 16)


def render_items(sources: dict[str, list[dict]], per_task: int = 300, seed: int = SEED) -> tuple[list[dict], list[dict]]:
    """Render skill (think-then-draw decoupling): the card is unchanged and the answer is given in the prompt.
    Train = training items of the reasoning tasks with the answer in the prompt; no test split of its own (the
    generator is evaluated with --route render / --route bridge on the normal test sets)."""
    rng = random.Random(seed + 8)
    train = []
    for name, items in sorted(sources.items()):
        chosen = rng.sample(items, min(per_task, len(items)))
        for it in chosen:
            row = {**it, "task": "render", "prompt": GENERIC_PROMPT + RENDER_INSTRUCTION.format(answer=it["target"])}
            row["item_id"] = "r" + it["item_id"]
            train.append(row)
    rng.shuffle(train)
    return train, []


def build_all(seed: int = SEED) -> dict[str, dict[str, list[dict]]]:
    """task -> {"train": [...], "test": [...]}"""
    qa_train, qa_test = qa_items(seed)
    copies = copy_items(seed=seed)
    arith_train, arith_test = arithmetic_items(seed=seed)
    instr_train, instr_test = instruction_items(seed)
    ctx_train, ctx_test = context_items(seed=seed)
    sel_train, sel_test = select_items(qa_train, qa_test, arith_train, arith_test, ctx_train, ctx_test, seed=seed)
    data = {"qa": (qa_train, qa_test), "arithmetic": (arith_train, arith_test), "instruction": (instr_train, instr_test),
            "context": (ctx_train, ctx_test), "select": (sel_train, sel_test), "qa_big": qa_big_items(seed),
            "select_qa": select_qa_items(qa_train, qa_test, seed), "visual": visual_items(seed=seed)}
    # text-channel control for training: the question is also given in the prompt (not native-pixel; tests whether the
    # DiT can learn to use encoder knowledge when the question reaches the encoder as text)
    data["qa_textq"] = ([{**i, "task": "qa_textq", "item_id": "t" + i["item_id"],
                          "prompt": f"{GENERIC_PROMPT}\nTask: {i['card_text']}"} for i in qa_train], [])
    data["render"] = render_items({"qa": qa_train, "arithmetic": arith_train, "instruction": instr_train,
                                   "context": ctx_train, "visual": data["visual"][0]}, seed=seed)
    out = {name: {"train": a, "test": b} for name, (a, b) in data.items()}
    for name, items in copies.items():
        out[name] = {"train": [i for i in items if i["split"] == "train"], "test": [i for i in items if i["split"] == "test"]}
    return out


def validate(data: dict[str, dict[str, list[dict]]]) -> dict[str, Any]:
    """Raise on any leakage or malformed item; return counts."""
    counts: dict[str, Any] = {}
    all_train_text = {i["card_text"] for d in data.values() for i in d["train"]}
    for task, d in data.items():
        ids = [i["item_id"] for i in d["train"] + d["test"]]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{task}: duplicate item ids")
        card_key = lambda i: i["card_text"] + json.dumps(i.get("scene"))  # visual items: same text, new picture
        train_text = {card_key(i) for i in d["train"]}
        for item in d["test"]:
            if card_key(item) in train_text:
                raise ValueError(f"{task}: test card is in train: {item['card_text']!r}")
        levels: dict[str, int] = {}
        for item in d["test"]:
            levels[item["level"]] = levels.get(item["level"], 0) + 1
        counts[task] = {"train": len(d["train"]), "test": levels}
    # Untrained facts: neither the entity nor the question may occur on any training card of the QA/select tasks.
    trained_qa_text = " \n ".join(i["card_text"] for t in ("qa", "select", "qa_big", "select_qa") for i in data[t]["train"])
    for item in data["qa"]["test"]:
        if item["level"] == "L4":
            entity = item["fact"]["entity"]
            if mentions(trained_qa_text, entity):
                raise ValueError(f"held-out entity {entity!r} occurs in training cards")
    for item in data["qa"]["test"] + data["qa_big"]["test"] + data["select_qa"]["test"]:
        if item["card_text"] in all_train_text:
            raise ValueError(f"QA test question appears in another task's training set: {item['card_text']!r}")
    return counts
