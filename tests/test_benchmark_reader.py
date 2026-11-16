import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generator_benchmark import cards  # noqa: E402
from generator_benchmark.reader import read_answer  # noqa: E402


class ReaderTest(unittest.TestCase):
    def test_reads_rendered_answers(self):
        for answer in ("Madrid", "X7R8T", "29", "riahc", "New Delhi", "Ulaanbaatar", "H2O", "bluewhalebluewhale"):
            self.assertEqual(read_answer(cards.write_card("Question?", answer)), answer)

    def test_blank_box(self):
        self.assertEqual(read_answer(cards.write_card("Question?")), "")


if __name__ == "__main__":
    unittest.main()
