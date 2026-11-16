"""The ten pixel-space tasks (CPU, no model): the true final canvases read back to the answer, the answer box stays
empty before the answer, frames change, and the scorers give full marks to the true frames and none to a blank one."""
import random
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pixel_models import canvas as Q, draw_tasks as D, maze_tasks as W, rl_draw as R, text_tasks as T  # noqa: E402


class TextTasks(unittest.TestCase):
    def test_answers_read_back(self):
        for spec, px in (("expr:4", 192), ("mul:4x4", 192), ("hop:3", 192), ("wp:3", 256)):
            task = T.Task(spec, px, 0)
            geo = task.geo
            ex = [task.sample(random.Random(1)) for _ in range(30)] + [e for v in task.test_sets(20).values() for e in v]
            last = np.stack([Q.render(geo, T.frame_codes(geo, e.rows[-1], e.answer)) for e in ex])
            self.assertEqual([g.strip() for g in Q.read(geo, last)], [e.answer for e in ex], spec)
            before = np.stack([Q.render(geo, T.frame_codes(geo, e.rows[-1], None)) for e in ex])
            self.assertTrue(all(g == "" for g in Q.read(geo, before)), spec)
            self.assertFalse(any(e.rows[t] == e.rows[t + 1] for e in ex for t in range(len(e.rows) - 1)), spec)

    def test_test_sets_fixed(self):
        a, b = T.Task("expr:4", 192, 0).test_sets(16), T.Task("expr:4", 192, 3).test_sets(16)
        self.assertEqual([e.prompt for e in a["k4"]], [e.prompt for e in b["k4"]])   # independent of the seed

    def test_detour_demonstrations(self):
        task = T.Task("expr:4", 192, 0)
        task.detour = 0.5
        rng = random.Random(0)
        ex = [task.sample(rng) for _ in range(50)]
        self.assertGreater(sum(len(e.rows) > 1 + len(e.steps) for e in ex), 10)   # a step written again on a new row


class DrawTasks(unittest.TestCase):
    SPECS = ("trace:6", "tracesum:7", "countpaint", "countpaint:tallysel", "countpaint:arithcheck:sel",
             "tracesum:7:detour", "countpaint:tallysel:detour", "countpaint:arithcheck:sel:detour")

    def test_true_frames_solve(self):
        for spec in self.SPECS:
            task = R.make_task(spec, 192)
            geo = task.geo
            for name, ex in task.test_sets(12).items():
                starts = [D.render(sc, geo, 0) for sc in ex]
                fin = np.stack([D.render(sc, geo, D.n_frames(sc)) for sc in ex])
                ms = R.score(task, geo, ex, starts, fin, "video")
                self.assertTrue(all(m["solved"] for m in ms), (spec, name))
                blank = np.stack(starts)
                self.assertFalse(any(m["solved"] for m in R.score(task, geo, ex, starts, blank, "video")), (spec, name))

    def test_direct_target_has_no_steps(self):
        task = D.Task("countpaint", 192)
        geo = task.geo
        for sc in task.test_sets(8)["k3"]:
            fin = D.render(sc, geo, D.n_frames(sc), with_trace=False)
            self.assertEqual(D.read_row(geo, fin[None], geo.q_rows - 1), [""])
            self.assertEqual(Q.read(geo, fin[None])[0].strip(), sc.answer)

    def test_clean_shares_detour_items(self):
        a, b = D.Task("countpaint:tallysel:detour", 192), D.Task("countpaint:tallysel:clean", 192)
        ra, rb = a.test_sets(4)["k3"], b.test_sets(4)["k3"]
        self.assertEqual([s.answer for s in ra], [s.answer for s in rb])


class Mazes(unittest.TestCase):
    def test_true_frames_solve(self):
        for spec in ("maze:5", "maze:5:sum", "maze:5:sum:detour"):
            task = W.Task(spec, 192)
            geo = task.geo
            for name, ex in task.test_sets(20).items():
                for sc in ex:
                    fin = W.render(sc, geo, W.n_frames(sc))
                    m = W.score_maze(sc, geo, fin, Q.read(geo, fin[None])[0])
                    self.assertEqual(m["legal"], 1.0, (spec, name))
                    self.assertEqual(W.score_maze(sc, geo, W.render(sc, geo, 0), "")["exact"], 0.0)


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "needs torch")
class Model(unittest.TestCase):
    def test_sizes_and_shapes(self):
        from pixel_models import model as P
        P.S = 192
        expect = {"XS": 7.4, "S": 32.5, "M": 57.7, "B": 129.7}   # non-embedding parameters, millions
        for size, m_ in expect.items():
            net = P.build(size)
            n = sum(p.numel() for n_, p in net.named_parameters() if n_ != "pos") / 1e6
            self.assertAlmostEqual(n, m_, delta=0.1, msg=size)
        x = torch.zeros(2, 1, 192, 192)
        self.assertEqual(tuple(P.build("XS")(x, x, torch.rand(2)).shape), (2, 1, 192, 192))


if __name__ == "__main__":
    unittest.main()
