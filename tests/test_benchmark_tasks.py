import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generator_benchmark import cards, kb, render, tasks  # noqa: E402

DATA = tasks.build_all()


def box_only_diff(item):
    a = np.asarray(render.render_input(item)).astype(int)
    b = np.asarray(render.render_target(item)).astype(int)
    return np.abs(a - b).sum(axis=2) > 0


class DataTest(unittest.TestCase):
    def test_validate_passes(self):
        counts = tasks.validate(DATA)
        for level in ("L1", "L2R", "L2F", "L3", "L4"):
            self.assertGreater(counts["qa"]["test"][level], 20, level)
        self.assertEqual(counts["copy_complete"]["train"], 10000)

    def test_deterministic(self):
        again = tasks.build_all()
        self.assertEqual(DATA["qa"]["train"], again["qa"]["train"])
        self.assertEqual(DATA["select"]["test"], again["select"]["test"])

    def test_held_entity_not_in_training(self):
        facts = tasks.build_facts()
        held = {f["entity"] for rows in facts.values() for f in rows if f["held"]}
        train_cards = {i["card_text"] for i in DATA["qa"]["train"] + DATA["select"]["train"]}
        self.assertGreater(len(held), 30)
        for entity in held:
            for card in train_cards:
                self.assertFalse(tasks.mentions(card, entity), (entity, card))

    def test_two_hop_uses_trained_facts(self):
        for item in (i for i in DATA["qa"]["test"] if i["level"] == "L3"):
            capital = item["fact"]["via_capital"]
            self.assertTrue(any(i["target"] == capital for i in DATA["qa"]["train"]), capital)
            self.assertTrue(any(i["fact"].get("relation") == item["fact"]["relation"] and i["target"] == item["target"]
                                for i in DATA["qa"]["train"]), item["card_text"])

    def test_reverse_answer_correct(self):
        rows = {r: {f["answer"]: f["entity"] for f in fs} for r, fs in tasks.build_facts().items()}
        for item in (i for i in DATA["qa"]["test"] if i["level"] == "L2R"):
            relation = item["fact"]["relation"]
            self.assertEqual(rows[relation][item["fact"]["entity"]], item["target"])

    def test_arithmetic_answers(self):
        for item in DATA["arithmetic"]["train"][:200] + DATA["arithmetic"]["test"]:
            expression = item["card_text"][len("What is "):-1].replace("x", "*")
            self.assertEqual(eval(expression), int(item["target"]), item["card_text"])  # noqa: S307

    def test_context_names_disjoint(self):
        words = lambda items: {w.strip(".?") for i in items for w in i["card_text"].split()} & set(kb.NAMES)
        self.assertFalse(words(DATA["context"]["train"]) & words(DATA["context"]["test"]))

    def test_select_options(self):
        for item in DATA["select"]["train"][:300] + DATA["select"]["test"][:300]:
            self.assertEqual(len(set(item["options"])), 3)
            self.assertEqual(item["options"][item["correct_index"]], item["target"])

    def test_all_cards_render_and_only_answer_region_changes(self):
        for task, d in DATA.items():
            for item in d["train"][:40] + d["test"][:40]:
                diff = box_only_diff(item)
                self.assertTrue(diff.any(), item["card_text"])
                ys, xs = np.nonzero(diff)
                if item["mode"] == "write":
                    x0, y0, x1, y1 = cards.ANSWER_INNER_ROI
                else:
                    x0, y0, x1, y1 = cards.OPTION_ROIS[item["correct_index"]]
                self.assertTrue(xs.min() >= x0 and xs.max() < x1 and ys.min() >= y0 and ys.max() < y1, item["card_text"])

    def test_every_card_fits(self):
        for task, d in DATA.items():
            for item in d["train"] + d["test"]:
                if task.startswith("copy") and item["split"] == "train":
                    continue  # same layout; sampled in the render test
                render.render_input(item)
                if item["mode"] == "write":
                    self.assertLessEqual(len(item["target"]), cards.ANSWER_MAX_CHARS)

    def test_transition_changes_one_cell(self):
        item = next(i for i in DATA["copy_transition"]["train"] if i["prefix"])
        ys, xs = np.nonzero(box_only_diff(item))
        k = len(item["prefix"])
        self.assertGreaterEqual(xs.min(), cards.char_cell(k)[0] - 2)
        self.assertLess(xs.max(), cards.char_cell(k + 1)[0] + 2)


if __name__ == "__main__":
    unittest.main()
