"""Mazes and Sum Mazes (Sec. 4.1, Appendix E).

  maze:N        an n x n grid (3 <= n <= N) of free cells and holes with a start S and a goal G, under "Path from S to G:".
                Each frame inks one more cell of the path and writes its move (R, D, L or U) on the note row; the last
                frame writes all moves in the answer box. The default demonstration is the breadth-first shortest path
                with ties broken in the fixed order R, D, L, U.
  maze:N:sum    Sum Mazes: a digit 1-9 in every free cell but S and G, under "Sum from S to G:". Each frame moves one
                cell, writes the move on the note row and updates the running total; the last frame writes the total.

Demonstrations for Sec. 7: :detour - a legal path that is longer than a shortest one (the non-optimal demonstrations);
:canon:as-detour - the canonical shortest demonstrations with the test items of the matching :detour spec (the optimal
SFT arm). Scoring (score_maze): exact - the demonstrated path; valid - any shortest path (on Sum Mazes, with that
path's own total).
"""

from __future__ import annotations

import argparse
import collections
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from . import canvas as Q

INK, GRAY, HOLE = 0, 120, 90
MOVES = {"R": (0, 1), "D": (1, 0), "L": (0, -1), "U": (-1, 0)}
MAZE_Q = "Path from S to G:"
SUM_Q = "Sum from S to G:"
CELL = 22            # maze cell, px
AREA_Y = (16, 128)   # the maze: rows 1-7 of the canvas (the note row below it is free)
_G = None


def text_canvas(rows: dict, geo: Q.Geo, answer: str | None = None) -> np.ndarray:
    global _G
    if _G is None:
        _G = Q.glyphs()
    m = np.zeros((geo.rows, geo.cols), np.uint8)
    for r, s in rows.items():
        a = np.frombuffer(s[:geo.cols].encode("ascii"), np.uint8) - 32
        m[r, :len(a)] = a
    if answer:
        m[geo.q_rows + 1:] = Q.wrap_rows(answer.rjust(geo.cols), Q.ANS_ROWS, geo.cols, False)
    t = _G[m].transpose(0, 2, 1, 3).reshape(geo.px, geo.px).copy()
    t[geo.line_y, 4:geo.px - 4] = 0
    return t


# ------------------------------------------------------------------------------------------------------------- maze
@dataclass
class MazeScene:
    kind: str
    N: int
    holes: set
    S: tuple
    G: tuple
    path: list      # the cells of the demonstrated path, S first (canon / rand: a shortest path; detour: a longer legal one)
    moves: str
    answer: str
    text: dict
    shortest: int = 0    # the length of a shortest path (BFS)
    demo: str = "canon"
    vals: dict | None = None   # maze:N:sum - a digit 1-9 in every free cell but S and G; the answer is the path's total


def _bfs(N: int, holes: set, G: tuple) -> dict:
    dist, q = {G: 0}, collections.deque([G])
    while q:
        r, c = q.popleft()
        for dr, dc in MOVES.values():
            n = (r + dr, c + dc)
            if 0 <= n[0] < N and 0 <= n[1] < N and n not in holes and n not in dist:
                dist[n] = dist[(r, c)] + 1
                q.append(n)
    return dist


MAX_MOVES = 20      # the note row holds 24 characters; demonstrations and rollouts stay within 20 moves
MAX_FRAMES = MAX_MOVES + 2


def _canon(S, G, d):
    """the shortest path with ties broken R, D, L, U (a single deterministic demonstrator)"""
    path, moves, cur = [S], "", S
    while cur != G:
        for m, (dr, dc) in MOVES.items():
            n = (cur[0] + dr, cur[1] + dc)
            if d.get(n, 10 ** 9) == d[cur] - 1:
                path.append(n); moves += m; cur = n
                break
    return path, moves


def _detour(rng, S, G, d, beta=1.0, slack=8):
    """a legal path that is NOT a shortest one: a self-avoiding walk drawn toward G (weights exp(-beta * distance change)),
    kept when it reaches G in more moves than the shortest path (and at most shortest + slack, MAX_MOVES)"""
    for _ in range(300):
        path, moves, cur, seen = [S], "", S, {S}
        while cur != G and len(moves) < MAX_MOVES:
            nb = [(m, (cur[0] + dr, cur[1] + dc)) for m, (dr, dc) in MOVES.items() if (cur[0] + dr, cur[1] + dc) in d and (cur[0] + dr, cur[1] + dc) not in seen]
            if not nb:
                break
            m, n = rng.choices(nb, weights=[math.exp(-beta * (d[n_] - d[cur])) for _, n_ in nb])[0]
            path.append(n); moves += m; cur = n; seen.add(n)
        if cur == G and d[S] < len(moves) <= min(MAX_MOVES, d[S] + slack):
            return path, moves
    return None


def path_cells(sc, moves: str):
    """the cells a string of moves visits from S (S first), or None when it leaves the grid, enters a hole or has a
    character that is not a move"""
    cur, path = sc.S, [sc.S]
    for m in moves:
        if m not in MOVES:
            return None
        cur = (cur[0] + MOVES[m][0], cur[1] + MOVES[m][1])
        if not (0 <= cur[0] < sc.N and 0 <= cur[1] < sc.N) or cur in sc.holes:
            return None
        path.append(cur)
    return path


def path_total(sc, path: list) -> int:
    """maze:N:sum - the total of the digits in the cells of a path (S and G hold none)"""
    return sum(sc.vals.get(c, 0) for c in path)


def maze_scene(rng: random.Random, N: int, p: float = 0.3, demo: str = "canon", total: bool = False) -> MazeScene:
    cells = [(r, c) for r in range(N) for c in range(N)]
    while True:
        S, G = rng.sample(cells, 2)
        holes = {x for x in cells if x != S and x != G and rng.random() < p}
        d = _bfs(N, holes, G)
        if d.get(S, 0) < N:   # long enough that the path is not a straight hop
            continue
        if demo == "canon":
            pm = _canon(S, G, d)
        else:
            pm = _detour(rng, S, G, d)
        if pm is not None and len(pm[1]) <= MAX_MOVES:
            break
    path, moves = pm
    if not total:
        return MazeScene("maze", N, holes, S, G, path, moves, moves, {0: MAZE_Q}, d[S], demo)
    # maze:N:sum: a digit in every free cell; each frame moves one cell and updates the running total at the end of the
    # note row; the answer is the total of the walked path
    vals = {c: rng.randint(1, 9) for c in cells if c not in holes and c != S and c != G}
    sc = MazeScene("maze", N, holes, S, G, path, moves, "", {0: SUM_Q}, d[S], demo, vals)
    sc.answer = str(path_total(sc, path))
    return sc


def answer_written(geo: Q.Geo, imgs: np.ndarray) -> list:
    """per canvas (grey uint8): is there text in the answer rows (a free-length rollout stops there)"""
    return [bool(g.strip()) for g in Q.read(geo, imgs)]


def _maze_origin(N: int, geo: Q.Geo) -> tuple:
    return (geo.px - N * CELL) // 2, AREA_Y[0] + (AREA_Y[1] - AREA_Y[0] - N * CELL) // 2


def _centre(sc: MazeScene, geo: Q.Geo, cell: tuple) -> tuple:
    x0, y0 = _maze_origin(sc.N, geo)
    return x0 + cell[1] * CELL + CELL // 2, y0 + cell[0] * CELL + CELL // 2


def render_maze(sc: MazeScene, geo: Q.Geo, frame: int) -> np.ndarray:
    L, done = len(sc.moves), min(frame, len(sc.moves))
    rows = dict(sc.text)
    if done:
        rows[geo.q_rows - 1] = sc.moves[:done]
        if sc.vals is not None:   # moves on the left, the running total right-aligned in the last 4 cells
            rows[geo.q_rows - 1] = sc.moves[:done].ljust(geo.cols - 4) + str(path_total(sc, sc.path[:done + 1])).rjust(4)
    txt = text_canvas(rows, geo, sc.answer if frame > L else None)
    im = Image.new("L", (geo.px, geo.px), 255)
    d = ImageDraw.Draw(im)
    x0, y0 = _maze_origin(sc.N, geo)
    for r in range(sc.N):
        for c in range(sc.N):
            d.rectangle((x0 + c * CELL, y0 + r * CELL, x0 + (c + 1) * CELL, y0 + (r + 1) * CELL),
                        fill=HOLE if (r, c) in sc.holes else 255, outline=GRAY)
    pts = [_centre(sc, geo, p) for p in sc.path[:done + 1]]
    if len(pts) > 1:
        d.line(pts, fill=INK, width=3)
    a = np.array(im)
    marks = [("S", sc.S), ("G", sc.G)] + ([(str(v), c) for c, v in sc.vals.items()] if sc.vals is not None else [])
    for ch, cell in marks:
        cx, cy = _centre(sc, geo, cell)
        y, x = cy - Q.CH // 2, cx - Q.CW // 2
        np.minimum(a[y:y + Q.CH, x:x + Q.CW], _G[ord(ch) - 32], out=a[y:y + Q.CH, x:x + Q.CW])
    return np.minimum(a, txt)


def score_maze(sc: MazeScene, geo: Q.Geo, fin: np.ndarray, got: str) -> dict:
    """legal: the written moves walk from S to G through free cells (any length); valid: legal and as short as a shortest
    path (any of them); exact: the demonstrated moves; extra: moves beyond the shortest (legal paths); path_iou: the new dark
    pixels of the drawing area against the demonstrated path's"""
    got = got.strip()
    extra_keys = {}
    if sc.vals is not None:
        # maze:N:sum - the moves are on the note row, the answer is the total. legal: a legal walk from S to G AND the
        # written total is that walk's own total; valid: legal and as short as a shortest path
        moves = written_moves(sc, geo, fin)
        path = path_cells(sc, moves)
        walk = path is not None and path[-1] == sc.G and len(moves) > 0
        sum_ok = walk and got == str(path_total(sc, path))
        legal = bool(sum_ok)
        extra_keys = {"walk": float(walk), "sum_ok": float(sum_ok)}
        answer, got = sc.moves if got == sc.answer else None, moves   # exact: the demonstrated moves and their total
    else:
        path = path_cells(sc, got)
        legal = path is not None and path[-1] == sc.G and len(got) > 0
        answer = sc.answer
    start, y0, y1 = render_maze(sc, geo, 0), AREA_Y[0], AREA_Y[1]
    want = (render_maze(sc, geo, len(sc.moves))[y0:y1] < 60) & ~(start[y0:y1] < 60)
    have = (fin[y0:y1] < 60) & ~(start[y0:y1] < 60)
    shortest = sc.shortest or len(sc.moves)
    return {"exact": float(got == answer), "legal": float(legal), "valid": float(legal and len(got) == shortest),
            "extra": float(len(got) - shortest) if legal else float("nan"),
            "path_iou": float((want & have).sum() / max(1, (want | have).sum())), **extra_keys}


def read_row(geo: Q.Geo, img: np.ndarray, row: int) -> str:
    """the text of one row of one canvas (nearest glyph per cell, canvas.read's rule for illegible cells)"""
    G = Q.glyphs().astype(np.float32)
    a = img[row * Q.CH:(row + 1) * Q.CH].astype(np.float32)
    cells = a.reshape(Q.CH, geo.cols, Q.CW).transpose(1, 0, 2)
    d = ((cells[:, None] - G[None]) ** 2).mean((-1, -2))
    best = d.argmin(-1)
    two = np.partition(d, 1, axis=-1)
    bad = (best != 0) & (two[..., 0] > 0.7 * two[..., 1])
    return "".join(Q.ILLEGIBLE if bad[c] else chr(32 + best[c]) for c in range(geo.cols)).strip()


def written_moves(sc: MazeScene, geo: Q.Geo, fin: np.ndarray) -> str:
    """maze:N:sum - the moves the model wrote: the first word of the note row of its last frame"""
    return (read_row(geo, fin, geo.q_rows - 1).split() or [""])[0]


def own_scene(sc: MazeScene, geo: Q.Geo, fin: np.ndarray, got: str) -> MazeScene:
    """the scene with the model's OWN solution as its demonstration (RFT re-draws a solved episode as clean frames): the
    moves it wrote (maze: the answer; maze:N:sum: the note row), their path and, for the sum, that path's total"""
    import copy
    sc2 = copy.copy(sc)
    moves = written_moves(sc, geo, fin) if sc.vals is not None else got.strip()
    sc2.path, sc2.moves = path_cells(sc, moves), moves
    sc2.answer = str(path_total(sc, sc2.path)) if sc.vals is not None else moves
    return sc2


# ------------------------------------------------------------------------------------------------------------ common
def n_frames(sc) -> int:
    return len(sc.moves) + 1


def render(sc, geo: Q.Geo, frame: int) -> np.ndarray:
    return render_maze(sc, geo, frame)


def pair(sc, geo: Q.Geo, rng: random.Random):
    t = rng.randint(0, n_frames(sc) - 1)
    return render(sc, geo, t), render(sc, geo, t + 1)


class Task:
    def __init__(self, spec: str, px: int):
        self.spec, self.geo = spec, Q.Geo(px)
        name, _, arg = spec.partition(":")
        # maze:5 | maze:5:sum | maze:5:sum:detour | maze:5:sum:canon:as-detour
        parts = arg.split(":")
        K, flags = int(parts[0]), parts[1:]
        assert all(f in ("canon", "detour", "sum", "as-detour") for f in flags), spec
        self.name, self.demo = name, next((f for f in flags if f in ("canon", "detour")), "canon")
        self.total, self.test_spec, test_demo = "sum" in flags, spec, self.demo
        if "as-detour" in flags:
            assert spec.endswith(f":{self.demo}:as-detour"), spec
            self.test_spec, test_demo = spec[: -len(f":{self.demo}:as-detour")] + ":detour", "detour"
        if name != "maze":
            raise ValueError(spec)
        self.train = lambda r: maze_scene(r, r.randint(3, K), demo=self.demo, total=self.total)
        self.sets = {f"n{n}": (lambda nn: lambda r: maze_scene(r, nn, demo=test_demo, total=self.total))(n) for n in range(3, K + 1)}

    def test_sets(self, n: int) -> dict:
        return {k: [f(random.Random(f"world-{self.test_spec}-{k}-{i}")) for i in range(n)] for k, f in self.sets.items()}


if __name__ == "__main__":
    main()


def main() -> None:
    """renders test scenes of every maze spec and checks that the true final frames score as solved"""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, help="write a contact sheet per spec here")
    a = ap.parse_args()
    geo, bad = Q.Geo(192), 0
    for spec in ("maze:5", "maze:5:sum", "maze:5:sum:detour"):
        t = Task(spec, 192)
        for name, ex in t.test_sets(200).items():
            fin = np.stack([render(sc, geo, n_frames(sc)) for sc in ex])
            got = Q.read(geo, fin)
            sc_ = [score_maze(sc, geo, f, g) for sc, f, g in zip(ex, fin, got)]
            ok = np.mean([s_["exact"] for s_ in sc_])
            bad += ok < 1
            print(f"{spec:18s} {name}: true final frames exact {ok:.3f}, frames mean {np.mean([n_frames(s_) for s_ in ex]):.1f}")
            blank = np.stack([render(sc, geo, 0) for sc in ex[:50]])
            assert np.mean([score_maze(sc, geo, f, "")["exact"] for sc, f in zip(ex, blank)]) == 0, "a blank canvas scored"
        if a.out:
            a.out.mkdir(parents=True, exist_ok=True)
            ex = list(t.test_sets(4).values())[-1]
            sheet = np.concatenate([np.concatenate([render(sc, geo, f) for f in np.linspace(0, n_frames(sc), 7).astype(int)], 1) for sc in ex], 0)
            Image.fromarray(sheet).save(a.out / f"{spec.replace(':', '_')}.png")
    print("check:", "OK" if bad == 0 else f"{bad} sets not solved by the true frames")


if __name__ == "__main__":
    main()
