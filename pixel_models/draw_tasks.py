"""The drawing tasks of Sec. 4.1 (Curve Tracing, Trace Then Sum, Count Then Paint, Compute Then Paint) and their
single-task trainer for Sec. 6 and the start models of Sec. 7.

  Curve Tracing      K curves connect letters on the left to numbers on the right and cross each other ("where does C
                     end?"); each frame inks a quarter of the asked curve, the last frame writes the number.
  Trace Then Sum     "B+E=?": B's curve is inked in two frames, the note "B=4" is written, E's curve is inked, "E=7"
                     is written, and the last frame writes the sum.
  Count Then Paint   three irregular regions in distinct greys, each with a letter and 1-6 dots; "paint the most dots".
                     Each frame circles one dot and updates the count ("A1", "A2", "A3", "A3 B1", ...); the last frames
                     write the chosen letter, box the label with that letter and paint the boxed region.
  Compute Then Paint "paint where dots = 7-2": the result is written first, the dots are counted as above, then one
                     frame per region marks + or -, and the region marked + is chosen, boxed and painted.

Regimes (--arm): direct (the problem canvas -> the final canvas in one generation) or video (frame by frame). Training
pairs are rendered on the fly by DataLoader workers. --chg-weight W weighs the pixels a step changes by 1 + W in the
loss (most of a frame is copied from its input), --cf-paint the counterfactual letters of Count Then Paint.

usage: python -m pixel_models.draw_tasks train --task trace:6 --arm video --size S --out DIR
Checkpoints (model, EMA, optimizer, step) every --save-every steps; a restarted run resumes on any GPU count.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from . import model as P
from . import canvas as Q

F_STEPS = 4          # drawn frames between the problem and the answer
GRAY, INK = 120, 0   # the base drawing / the inked chain of thought


@dataclass
class Scene:
    kind: str
    text: dict           # row -> string (question, labels)
    answer: str
    paths: list          # polylines (lists of (x, y)) drawn in gray in every frame
    trace: list          # the polyline inked in over the frames (Curve Tracing)
    marks: list          # filled dots (x, y, r) drawn in every frame
    ink_width: int = 3
    steps: list = None   # the chain of thought, one action per frame: ("ink", polyline index, fraction) | ("note", row, text)
    traces: list = None  # the polylines the ink actions refer to
    regions: list = None # countpaint: the block polygons, their greys, the dots (x, y), the labels (letter, x, y), the answer
    fills: list = None
    dots: list = None
    labels: list = None
    chosen: int = -1

    def __post_init__(self):
        if self.steps is None:   # Curve Tracing: the one trace inked a quarter per frame
            self.traces = [self.trace]
            self.steps = [("ink", 0, (i + 1) / F_STEPS) for i in range(F_STEPS)]


# ------------------------------------------------------------------------------------------------------------- tasks
def _smooth(points, n=64):
    """Catmull-Rom through the points (sorted by x), n samples"""
    pts = [points[0]] + points + [points[-1]]
    out = []
    for i in range(1, len(pts) - 2):
        p0, p1, p2, p3 = (np.array(pts[j], float) for j in (i - 1, i, i + 1, i + 2))
        for t in np.linspace(0, 1, n // (len(points) - 1), endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out.append(np.array(points[-1], float))
    return [(float(x), float(y)) for x, y in out]


def trace_scene(rng: random.Random, geo: Q.Geo, k: int, waypoints: int, note_row: bool = False, width: int = 1) -> Scene:
    rows = list(range(1, geo.q_rows - (1 if note_row else 0)))   # drawing rows (row 0: the question; last: notes)
    k = min(k, len(rows))
    starts = sorted(rng.sample(rows, k))
    ends = rng.sample(rows, k)
    letters = [chr(65 + i) for i in range(k)]
    numbers = rng.sample(range(1, 10), k)
    y_of = lambda r: r * Q.CH + Q.CH / 2
    x0, x1 = Q.CW + 2, geo.px - Q.CW - 2
    paths = []
    for s, e in zip(starts, ends):
        xs = np.linspace(x0, x1, waypoints + 2)
        y_hi = (geo.q_rows - (1 if note_row else 0)) * Q.CH - 3      # curves stay above the note row
        ys = [y_of(s)] + [rng.uniform(Q.CH + 3, y_hi) for _ in range(waypoints)] + [y_of(e)]
        paths.append(_smooth(list(zip(xs, ys))))
    text = {0: None}
    q = rng.randrange(k)
    text[0] = f"where does {letters[q]} end?"
    for i, s in enumerate(starts):
        text.setdefault(s, [" "] * geo.cols)
        text[s][0] = letters[i]
    for i, e in enumerate(ends):
        text.setdefault(e, [" "] * geo.cols)
        text[e][geo.cols - 1] = str(numbers[i])
    text = {r: (v if isinstance(v, str) else "".join(v)) for r, v in text.items()}
    sc = Scene("trace", text, str(numbers[q]), paths, paths[q], [])
    sc.path_width = width
    return sc


def tracesum_scene(rng: random.Random, geo: Q.Geo, k: int, waypoints: int, draw: bool = True, width: int = 1) -> Scene:
    """two lookups along crossing curves and their sum; the notes are written in the last drawing row"""
    sc = trace_scene(rng, geo, k, waypoints, note_row=True, width=width)
    letters = [r[0] for r in (sc.text.get(row, " ") for row in range(1, geo.q_rows)) if r and r[0].isalpha()]
    starts = [row for row in range(1, geo.q_rows) if sc.text.get(row, " ")[:1].isalpha()]
    num_at = {row: sc.text[row][geo.cols - 1] for row in range(1, geo.q_rows) if sc.text.get(row, " ")[-1:].isdigit()}
    a, b = rng.sample(range(len(sc.paths)), 2)
    end_row = lambda i: int(round((sc.paths[i][-1][1] - Q.CH / 2) / Q.CH))
    va, vb = int(num_at[end_row(a)]), int(num_at[end_row(b)])
    la, lb = sc.text[starts[a]][0], sc.text[starts[b]][0]
    sc.text[0] = f"{la}+{lb}=?"
    nr = geo.q_rows - 1
    n1, n2 = f"{la}={va}", f"{la}={va} {lb}={vb}"
    ink = lambda i: [("ink", i, 0.5), ("ink", i, 1.0)] if draw else []
    sc.kind, sc.answer = ("tracesum" if draw else "tracenote"), str(va + vb)
    sc.traces = sc.paths
    sc.steps = ink(a) + [("note", nr, n1)] + ink(b) + [("note", nr, n2)]
    return sc


TRACE_DETOUR_P = 0.5   # tracesum:K:detour - the chance that a demonstration first inks a curve that is not asked


def tracesum_detour_scene(rng: random.Random, geo: Q.Geo, k: int, waypoints: int, p: float = TRACE_DETOUR_P) -> Scene:
    """Trace Then Sum for RL (Sec. 7): free length (an episode ends where the sum is written), and with probability p the
    demonstration first inks, in two frames, a curve that is neither of the two asked (the non-optimal demonstrations).
    Test scenes have no detour; a solution is minimal when it uses n_min frames. With p = 0 the rng is consumed exactly
    as by tracesum_scene."""
    sc = tracesum_scene(rng, geo, k, waypoints)
    base = list(sc.steps)
    sc.base_steps, sc.n_min = base, len(base) + 1
    if p and rng.random() < p:
        a, b = base[0][1], base[3][1]   # the two asked curves: ink(a) x2, note, ink(b) x2, note
        wrong = rng.choice([i for i in range(len(sc.paths)) if i not in (a, b)])
        sc.steps = [("ink", wrong, 0.5), ("ink", wrong, 1.0)] + base
    return sc


CP_FILLS = (125, 160, 195, 225)   # block greys (the dots, the labels and the paint are black)
DOT_R = 2.5
CP_Q = "paint the most dots"


def _blob(rng: random.Random, cx: float, cy: float, R: float):
    """an irregular star-shaped block: r(theta) = R (1 + sum_k a_k cos(k theta + phi_k)); returns r(.)"""
    (k2, a2, p2), (k3, a3, p3), (k4, a4, p4), (k5, a5, p5) = [(k, rng.uniform(-0.15, 0.15), rng.uniform(0, 2 * math.pi)) for k in (2, 3, 4, 5)]
    cos = math.cos

    def r(th):   # the terms added one by one in a fixed order: the same floats, so the same scenes, on every machine
        t = 0 + a2 * cos(k2 * th + p2)
        t += a3 * cos(k3 * th + p3)
        t += a4 * cos(k4 * th + p4)
        t += a5 * cos(k5 * th + p5)
        return R * (1 + t)
    return r


def _poly(cx, cy, r, grow=0.0, n=60):
    out = []
    for i in range(n):
        th = 2 * math.pi * i / n
        rr = r(th) + grow
        out.append((cx + rr * math.cos(th), cy + rr * math.sin(th)))
    return out


def _blob_np(rng: random.Random, cx: float, cy: float, R: float):
    """_blob with numpy (vectorized over theta), for Compute Then Paint"""
    ks = [(k, rng.uniform(-0.15, 0.15), rng.uniform(0, 2 * math.pi)) for k in (2, 3, 4, 5)]
    return lambda th: R * (1 + sum(a * np.cos(k * np.asarray(th, float) + ph) for k, a, ph in ks))


_TH = 2 * np.pi * np.arange(60) / 60


def _poly_np(cx, cy, r, grow=0.0):
    rr = r(_TH) + grow
    return list(zip((cx + rr * np.cos(_TH)).tolist(), (cy + rr * np.sin(_TH)).tolist()))


def _mask(px: int, poly) -> np.ndarray:
    im = Image.new("L", (px, px), 0)
    ImageDraw.Draw(im).polygon(poly, fill=1)
    return np.asarray(im).astype(bool)


def countpaint_scene(rng: random.Random, geo: Q.Geo, k: int = 3, cmin: int = 1, cmax: int = 6) -> Scene:
    """k blocks in the drawing rows 1..q_rows-2 (the last drawing row holds the counts), dots inside, a unique maximum"""
    x0, x1, y0, y1 = 2, geo.px - 2, Q.CH + 2, (geo.q_rows - 1) * Q.CH - 2
    while True:
        blocks, taken = [], np.zeros((geo.px, geo.px), bool)
        for _ in range(400):
            if len(blocks) == k:
                break
            R = rng.uniform(21, 28)
            cx, cy = rng.uniform(x0 + R, x1 - R), rng.uniform(y0 + R, y1 - R)
            r = _blob(rng, cx, cy, R)
            outer = _poly(cx, cy, r, 3)
            xs, ys = [p[0] for p in outer], [p[1] for p in outer]
            if min(xs) < x0 - 3 or max(xs) > x1 + 3 or min(ys) < y0 - 3 or max(ys) > y1 + 3:
                continue
            mo = _mask(geo.px, outer)
            if (mo & taken).any():
                continue
            taken |= mo
            blocks.append((cx, cy, r))
        if len(blocks) < k:
            continue
        while True:
            counts = [rng.randint(cmin, cmax) for _ in range(k)]
            if counts.count(max(counts)) == 1:
                break
        dots, labels, ok = [], [], True
        for j, (cx, cy, r) in enumerate(blocks):
            inside = lambda x, y, m: math.hypot(x - cx, y - cy) <= r(math.atan2(y - cy, x - cx)) - m
            for _ in range(200):   # the label: its whole cell inside the block, near the centre
                lx, ly = round(cx - Q.CW / 2 + rng.uniform(-6, 6)), round(cy - Q.CH / 2 + rng.uniform(-6, 6))
                if all(inside(x, y, 2) for x in (lx, lx + Q.CW) for y in (ly, ly + Q.CH)):
                    break
            else:
                ok = False; break
            labels.append((chr(65 + j), lx, ly))
            mine = []
            for _ in range(2000):
                if len(mine) == counts[j]:
                    break
                x, y = cx + rng.uniform(-30, 30), cy + rng.uniform(-30, 30)
                if not inside(x, y, 5):
                    continue
                if lx - 6 <= x <= lx + Q.CW + 6 and ly - 2 <= y <= ly + Q.CH + 2:   # clear of the letter
                    continue
                if any(math.hypot(x - u, y - v) < 8 for u, v in mine):
                    continue
                mine.append((x, y))
            if len(mine) < counts[j]:
                ok = False; break
            dots += mine
        if ok:
            break
    regions = [_poly(cx, cy, r) for cx, cy, r in blocks]
    best = counts.index(max(counts))
    nr = geo.q_rows - 1
    notes = [" ".join(f"{chr(65 + j)}{counts[j]}" for j in range(i + 1)) for i in range(k)]
    steps = [("note", nr, t) for t in notes] + [("answer",), ("paint", best)]
    sc = Scene("countpaint", {0: CP_Q}, chr(65 + best), [], [], [], steps=steps, traces=[],
               regions=regions, fills=rng.sample(CP_FILLS, k), dots=dots, labels=labels, chosen=best)
    sc.counts = counts
    return sc


def countpaint_tally_scene(rng: random.Random, geo: Q.Geo, select: bool = False, k: int = 3, cmin: int = 1, cmax: int = 6) -> Scene:
    """Count Then Paint frame by frame: each frame circles the next dot of the current region (top to bottom) and raises
    its running count ("A1" -> "A2" -> "A3" -> "A3 B1" ...); then the letter, the box around the chosen label (select)
    and the painting. (k, cmin, cmax = 4, 2, 8: the variant of the joint scaling ladder)"""
    sc = countpaint_scene(rng, geo, k, cmin, cmax)
    nr = geo.q_rows - 1
    steps, done_notes, start = [], [], 0
    for j, c in enumerate(sc.counts):
        mine = sorted(range(start, start + c), key=lambda i: (round(sc.dots[i][1]), sc.dots[i][0]))
        start += c
        for m, i in enumerate(mine, 1):
            steps.append(("tally", nr, " ".join(done_notes + [f"{chr(65 + j)}{m}"]), i))
        done_notes.append(f"{chr(65 + j)}{c}")
    sel = [("select", sc.chosen)] if select else []   # tallysel: box the chosen block's label, then fill the boxed block
    sc.steps = steps + [("answer",)] + sel + [("paint", sc.chosen)]
    sc.level = "tallysel" if select else "tally"
    return sc


DETOUR_P = 0.5   # countpaint:...:detour - the chance that a demonstration boxes a wrong block before the right one


def countpaint_detour_scene(rng: random.Random, geo: Q.Geo, select: bool = True, p: float = DETOUR_P) -> Scene:
    """Count Then Paint for RL (Sec. 7): the letter is written last, so an episode of free length ends where the answer
    is written, and with probability p the demonstration first boxes a wrong block and the next frame moves the box to
    the right one (a legal, visible, useless step: the non-optimal demonstrations). Test scenes have no detour; a
    solution is minimal when it uses n_min frames."""
    sc = countpaint_tally_scene(rng, geo, select)
    base = [st for st in sc.steps if st[0] != "answer"]
    sc.steps, sc.base_steps, sc.n_min = list(base), base, len(base) + 1
    if select and p and rng.random() < p and len(sc.labels) > 1:
        i = next(j for j, st in enumerate(sc.steps) if st[0] == "select")
        wrong = rng.choice([b for b in range(len(sc.labels)) if b != sc.chosen])
        sc.steps = sc.steps[:i] + [("select", wrong)] + sc.steps[i:]
    sc.level = sc.level + "-detour"
    return sc


def computepaint_scene(rng: random.Random, geo: Q.Geo, free: bool = False, p: float = 0.0) -> Scene:
    """Compute Then Paint (Appendix E): "paint where dots = 7-2". The first frame writes 7-2=5, each following frame
    circles one dot and updates the count ("7-2=5 A1" ... "7-2=5 A5 B1 C3"); after the counts one frame per region
    appends + if its count equals the result and - otherwise ("... A- B+ C-"), then the letter copies the region
    marked +, a frame boxes its label and the last frame paints it.
    free (Sec. 7): the letter is written last (an episode of free length ends where the answer is written), and with
    probability p the demonstration first boxes a wrong region (the non-optimal demonstrations)."""
    sc = countpaint_arith_scene(rng, geo)
    pre, nr = sc.steps[0][2], geo.q_rows - 1
    steps, done, start = [("note", nr, pre)], [pre], 0
    for j, c in enumerate(sc.counts):
        mine = sorted(range(start, start + c), key=lambda i: (round(sc.dots[i][1]), sc.dots[i][0]))
        start += c
        for m, i in enumerate(mine, 1):
            steps.append(("tally", nr, " ".join(done + [f"{chr(65 + j)}{m}"]), i))
        done.append(f"{chr(65 + j)}{c}")
    full, marks = " ".join(done), []
    for j, c in enumerate(sc.counts):
        marks.append(f"{chr(65 + j)}{'+' if j == sc.chosen else '-'}")
        steps.append(("note", nr, full + " " + " ".join(marks)))
    assert len(steps[-1][2]) <= geo.cols, steps[-1][2]
    tail = [("select", sc.chosen), ("paint", sc.chosen)]
    sc.level = "arithcheck"
    if not free:
        sc.steps = steps + [("answer",)] + tail
        return sc
    base = steps + tail
    sc.steps, sc.base_steps, sc.n_min = list(base), base, len(base) + 1
    if p and rng.random() < p:
        wrong = rng.choice([b for b in range(len(sc.labels)) if b != sc.chosen])
        i = len(steps)
        sc.steps = sc.steps[:i] + [("select", wrong)] + sc.steps[i:]
    sc.level = sc.level + "-detour"
    return sc


def countpaint_arith_scene(rng: random.Random, geo: Q.Geo) -> Scene:
    """the scene of Compute Then Paint: 3 non-overlapping blocks of 1-6 dots with distinct counts under "paint where
    dots = 9-4"; the steps write 9-4=5, the counts, the letter of the block with 5 dots, box its label and paint it"""
    x0, x1, y0, y1 = 2, geo.px - 2, Q.CH + 2, (geo.q_rows - 1) * Q.CH - 2
    px = geo.px

    def fits(cx, cy, r):
        o = _poly_np(cx, cy, r, 3)
        return min(p[0] for p in o) >= x0 - 3 and max(p[0] for p in o) <= x1 + 3 and min(p[1] for p in o) >= y0 - 3 and max(p[1] for p in o) <= y1 + 3

    while True:
        blocks = []
        grown = []   # the placed blocks' masks, grown by 3 px
        for _ in range(400):
            if len(blocks) == 3:
                break
            R = rng.uniform(21, 28)
            cx, cy = rng.uniform(x0 + R, x1 - R), rng.uniform(y0 + R, y1 - R)
            r = _blob_np(rng, cx, cy, R)
            if not fits(cx, cy, r):
                continue
            m = _mask(px, _poly_np(cx, cy, r, 3))
            if any((m & g_).sum() / m.sum() > 0 for g_ in grown):
                continue
            blocks.append((cx, cy, r)); grown.append(m)
        if len(blocks) < 3:
            continue
        others = lambda j, g: np.any([_mask(px, _poly_np(*blocks[k], g)) for k in range(3) if k != j], axis=0)
        dot_ok = [_mask(px, _poly_np(*blocks[j], -5)) & ~others(j, 5) for j in range(3)]
        lab_ok = [_mask(px, _poly_np(*blocks[j], -2)) & ~others(j, 3) for j in range(3)]
        counts = rng.sample(range(1, 7), 3)
        dots, labels, ok = [], [], True
        for j, (cx, cy, r) in enumerate(blocks):
            for _ in range(300):
                lx, ly = round(cx - Q.CW / 2 + rng.uniform(-14, 14)), round(cy - Q.CH / 2 + rng.uniform(-14, 14))
                if 0 <= ly and ly + Q.CH <= px and 0 <= lx and lx + Q.CW <= px and lab_ok[j][ly:ly + Q.CH, lx:lx + Q.CW].all():
                    break
            else:
                ok = False; break
            labels.append((chr(65 + j), lx, ly))
            mine = []
            for _ in range(3000):
                if len(mine) == counts[j]:
                    break
                x, y = cx + rng.uniform(-34, 34), cy + rng.uniform(-34, 34)
                if not (0 <= x < px and 0 <= y < px) or not dot_ok[j][int(round(y)) % px, int(round(x)) % px]:
                    continue
                if lx - 6 <= x <= lx + Q.CW + 6 and ly - 2 <= y <= ly + Q.CH + 2:
                    continue
                if any(math.hypot(x - u, y - v) < 8 for u, v in mine):
                    continue
                mine.append((x, y))
            if len(mine) < counts[j]:
                ok = False; break
            dots += mine
        if ok:
            break
    nr = geo.q_rows - 1
    best = rng.randrange(3)
    tgt = counts[best]
    while True:
        u, op = rng.randint(1, 9), rng.choice("+-")
        v = tgt - u if op == "+" else u - tgt
        if 1 <= v <= 9:
            break
    q = f"paint where dots = {u}{op}{v}"
    pre = f"{u}{op}{v}={tgt}"
    notes = [" ".join([pre] + [f"{chr(65 + j)}{counts[j]}" for j in range(i + 1)]) for i in range(3)]
    steps = [("note", nr, pre)] + [("note", nr, t) for t in notes] + [("answer",), ("select", best), ("paint", best)]
    sc = Scene("countpaint", {0: q}, chr(65 + best), [], [], [], steps=steps, traces=[],
               regions=[_poly_np(cx, cy, r) for cx, cy, r in blocks], fills=rng.sample(CP_FILLS, 3), dots=dots, labels=labels,
               chosen=best)
    sc.counts, sc.level = counts, "arith"
    return sc


def cp_targets(sc: Scene, px: int) -> list:
    """the masks a painting is scored against: the regions"""
    return [_mask(px, p) for p in sc.regions]


def n_frames(sc: Scene) -> int:
    """frames after the problem: one per step, plus the answer frame unless a step writes the answer (countpaint)"""
    return len(sc.steps) + (0 if any(s[0] == "answer" for s in sc.steps) else 1)


def read_row(geo: Q.Geo, imgs: np.ndarray, row: int) -> list[str]:
    """the text of one row of each canvas (nearest glyph per cell, canvas.read's rule for illegible cells)"""
    G = Q.glyphs().astype(np.float32)
    a = imgs[:, row * Q.CH:(row + 1) * Q.CH].astype(np.float32)
    cells = a.reshape(len(imgs), Q.CH, geo.cols, Q.CW).transpose(0, 2, 1, 3)
    d = ((cells[:, :, None] - G[None, None]) ** 2).mean((-1, -2))
    best = d.argmin(-1)
    two = np.partition(d, 1, axis=-1)
    bad = (best != 0) & (two[..., 0] > 0.7 * two[..., 1])
    return ["".join(Q.ILLEGIBLE if bad[i, c] else chr(32 + best[i, c]) for c in range(geo.cols)).strip() for i in range(len(imgs))]


def paint_ious(sc: Scene, geo: Q.Geo, start: np.ndarray, fin: np.ndarray) -> list[float]:
    """IoU of the newly painted (dark) pixels of the drawing rows with each block (pixels dark in the problem excluded)"""
    y0, y1 = Q.CH, (geo.q_rows - 1) * Q.CH
    old = start < 60
    tally, sel = any(st[0] == "tally" for st in sc.steps), any(st[0] == "select" for st in sc.steps)
    if tally or sel:   # the rings drawn while counting and the box around the chosen label are not painting
        im = Image.new("L", (geo.px, geo.px), 0)
        dr = ImageDraw.Draw(im)
        if tally:
            for x, y in sc.dots:
                dr.ellipse((x - 6, y - 6, x + 6, y + 6), fill=1)
        if sel:
            for L, lx, ly in sc.labels:
                dr.rectangle((lx - 3, ly - 1, lx + Q.CW + 2, ly + Q.CH), fill=1)
        old = old | np.asarray(im).astype(bool)
    new = (fin < 60) & ~old
    new[:y0] = False; new[y1:] = False
    out = []
    for m in cp_targets(sc, geo.px):
        gt = m & ~old
        out.append(float((new & gt).sum() / max(1, (new | gt).sum())))
    return out


# ----------------------------------------------------------------------------------------------------------- render
def _cut(poly, frac):
    """the polyline from its start up to the given fraction of its length"""
    if frac >= 1:
        return poly
    seg = [math.dist(poly[i], poly[i + 1]) for i in range(len(poly) - 1)]
    want, out, run = frac * sum(seg), [poly[0]], 0.0
    for i, s in enumerate(seg):
        if run + s >= want:
            u = (want - run) / max(s, 1e-9)
            out.append((poly[i][0] + u * (poly[i + 1][0] - poly[i][0]), poly[i][1] + u * (poly[i + 1][1] - poly[i][1])))
            return out
        run += s
        out.append(poly[i + 1])
    return out


_G = None


def render(sc: Scene, geo: Q.Geo, frame: int, with_trace: bool = True) -> np.ndarray:
    """frame 0: the problem; 1..F_STEPS: the trace inked in up to frame / F_STEPS; F_STEPS + 1: the full trace + the answer"""
    global _G
    if _G is None:
        _G = Q.glyphs()
    n = len(sc.steps)
    # with_trace=False: the direct arm's target, without the chain of thought (countpaint keeps the letter and the paint)
    done = sc.steps[:frame] if with_trace else [s for s in sc.steps[:frame] if s[0] in ("answer", "paint")]
    rows = dict(sc.text)
    for act in done:
        if act[0] in ("note", "tally"):
            rows[act[1]] = act[2]
    m = np.zeros((geo.rows, geo.cols), np.uint8)
    for r, s in rows.items():
        a = np.frombuffer(s[:geo.cols].encode("ascii"), np.uint8) - 32
        m[r, :len(a)] = a
    if frame > n or any(act[0] == "answer" for act in done):
        m[geo.q_rows + 1:] = Q.wrap_rows(sc.answer.rjust(geo.cols), Q.ANS_ROWS, geo.cols, False)
    txt = _G[m].transpose(0, 2, 1, 3).reshape(geo.px, geo.px).copy()
    txt[geo.line_y, 4:geo.px - 4] = 0
    if sc.kind == "countpaint":
        im = Image.new("L", (geo.px, geo.px), 255)
        d = ImageDraw.Draw(im)
        for poly, g in zip(sc.regions, sc.fills):
            d.polygon(poly, fill=g)
        d = ImageDraw.Draw(im)
        for x, y in sc.dots:
            d.ellipse((x - DOT_R, y - DOT_R, x + DOT_R, y + DOT_R), fill=INK)
        a = np.array(im)
        for L, x, y in sc.labels:
            np.minimum(a[y:y + Q.CH, x:x + Q.CW], _G[ord(L) - 32], out=a[y:y + Q.CH, x:x + Q.CW])
        im = Image.fromarray(a)
        d = ImageDraw.Draw(im)
        last_sel = max((i for i, act in enumerate(done) if act[0] == "select"), default=None)
        for i, act in enumerate(done):
            if act[0] == "tally":   # the counted dot: a ring around it
                x, y = sc.dots[act[3]]
                d.ellipse((x - 4.5, y - 4.5, x + 4.5, y + 4.5), outline=INK, width=1)
            elif act[0] == "select" and i == last_sel:   # the chosen block: a box around its label (only the last one:
                # a detour demo boxes a wrong block first and then moves the box)
                L, lx, ly = sc.labels[act[1]]
                d.rectangle((lx - 2, ly, lx + Q.CW + 1, ly + Q.CH - 1), outline=INK, width=1)
        for act in done:
            if act[0] == "paint":
                d.polygon(sc.regions[act[1]], fill=INK)
        return np.minimum(np.array(im), txt)
    im = Image.new("L", (geo.px, geo.px), 255)
    d = ImageDraw.Draw(im)
    for poly in sc.paths:
        d.line(poly, fill=GRAY, width=getattr(sc, "path_width", 1))
    for x, y, r in sc.marks:
        d.ellipse((x - r, y - r, x + r, y + r), fill=INK)
    inked = {}
    for act in done:
        if act[0] == "ink":
            inked[act[1]] = max(inked.get(act[1], 0.0), act[2])
    for i, frac in inked.items():
        part = _cut(sc.traces[i], frac)
        if len(part) > 1:
            d.line(part, fill=INK, width=sc.ink_width)
    return np.minimum(np.asarray(im), txt)


class Task:
    """training scenes and fixed test sets of one drawing task. The spec strings name the tasks of the paper:

      trace:6                          Curve Tracing (2-6 curves in training; test sets k2, k4, k6, k8)
      tracesum:7                       Trace Then Sum (3-7 curves; test sets k3, k5, k7)
      countpaint:tallysel              Count Then Paint, frame by frame (one dot per frame, box the label, paint)
      countpaint                       Count Then Paint, direct answering (the same scenes without the counting steps;
                                       evaluate it with --items countpaint:tallysel for the frame-by-frame test items)
      countpaint:tallysel:hard         Count Then Paint in the joint scaling ladder (4 regions of 2-8 dots)
      countpaint:arithcheck:sel        Compute Then Paint
    and for the RL comparison of Sec. 7 (free length, solved = right in the fewest frames):
      tracesum:7:detour / :clean                         non-optimal (start, non-optimal SFT) / optimal demonstrations
      countpaint:tallysel:detour / :clean                idem, Count Then Paint
      countpaint:arithcheck:sel:detour / :clean          idem, Compute Then Paint
    A :clean spec draws the test items of its :detour spec. Test items are drawn from Random(f"draw-{spec}-{set}-{i}")."""

    def __init__(self, spec: str, px: int):
        self.spec, self.geo = spec, Q.Geo(px)
        self.free, self.per_frame = False, 1
        name, _, arg = spec.partition(":")
        self.name = name
        if name == "trace":
            K = int(arg)
            self.train = lambda r: trace_scene(r, self.geo, r.randint(2, K), r.randint(2, 4))
            self.sets = {f"k{k}": (lambda kk: lambda r: trace_scene(r, self.geo, kk, 3))(k) for k in (2, 4, 6, 8) if k <= self.geo.q_rows - 1}
        elif name == "tracesum" and arg.partition(":")[2] in ("detour", "clean"):
            # the test items are those of tracesum:K itself (p = 0 consumes the rng exactly as tracesum_scene)
            K, pd = int(arg.partition(":")[0]), (TRACE_DETOUR_P if arg.endswith("detour") else 0.0)
            self.free = True
            self.test_spec = f"tracesum:{K}"
            self.train = lambda r: tracesum_detour_scene(r, self.geo, r.randint(3, K), r.randint(2, 4), pd)
            self.sets = {f"k{k}": (lambda kk: lambda r: tracesum_detour_scene(r, self.geo, kk, 3, 0.0))(k) for k in (3, 5, 7) if k <= self.geo.q_rows - 2}
        elif name == "tracesum":
            K = int(arg)
            self.train = lambda r: tracesum_scene(r, self.geo, r.randint(3, K), r.randint(2, 4))
            self.sets = {f"k{k}": (lambda kk: lambda r: tracesum_scene(r, self.geo, kk, 3))(k) for k in (3, 5, 7) if k <= self.geo.q_rows - 2}
        elif name == "countpaint" and not arg:   # direct answering: the letter and the painting, no counting steps
            self.train = lambda r: countpaint_scene(r, self.geo, 3)
            self.sets = {"k3": lambda r: countpaint_scene(r, self.geo, 3)}
        elif name == "countpaint" and arg == "tallysel:hard":
            self.train = lambda r: countpaint_tally_scene(r, self.geo, True, 4, 2, 8)
            self.sets = {"k4": lambda r: countpaint_tally_scene(r, self.geo, True, 4, 2, 8)}
            self.reuse = 8   # a scene takes ~0.1 s to place: train_joint draws 8 transitions from each
        elif name == "countpaint" and arg in ("tallysel:detour", "tallysel:clean"):
            self.free = True
            pd = DETOUR_P if arg.endswith("detour") else 0.0
            self.test_spec = spec.replace(":clean", ":detour")
            self.train = lambda r: countpaint_detour_scene(r, self.geo, True, pd)
            self.sets = {"k3": lambda r: countpaint_detour_scene(r, self.geo, True, 0.0)}
        elif name == "countpaint" and arg == "tallysel":
            self.train = lambda r: countpaint_tally_scene(r, self.geo, True)
            self.sets = {"k3": lambda r: countpaint_tally_scene(r, self.geo, True)}
        elif name == "countpaint" and arg == "arithcheck:sel":
            self.train = lambda r: computepaint_scene(r, self.geo)
            self.sets = {"arith": lambda r: computepaint_scene(r, self.geo)}
        elif name == "countpaint" and arg in ("arithcheck:sel:detour", "arithcheck:sel:clean"):
            self.free = True
            pd = DETOUR_P if arg.endswith("detour") else 0.0
            self.test_spec = spec.replace(":clean", ":detour")
            self.train = lambda r: computepaint_scene(r, self.geo, True, pd)
            self.sets = {"arith": lambda r: computepaint_scene(r, self.geo, True, 0.0)}
        else:
            raise ValueError(spec)

    def test_sets(self, n: int) -> dict:
        ts = getattr(self, "test_spec", self.spec)   # :clean shares the :detour test items
        return {k: [f(random.Random(f"draw-{ts}-{k}-{i}")) for i in range(n)] for k, f in self.sets.items()}


def counterfactual(sc: Scene, j: int) -> Scene:
    """the same canvas, but the written letter names region j whatever the counts, and the steps after the letter
    (box, paint) follow it (Appendix E): ties the painted region to the written letter, rather than to what looks like
    the most dots"""
    import copy
    c = copy.copy(sc)
    c.answer, c.chosen = chr(65 + j), j
    c.steps = [(st[0], j) if st[0] in ("select", "paint") else st for st in sc.steps]
    return c


def pair(sc: Scene, geo: Q.Geo, arm: str, rng: random.Random, cf: float = 0.0, per_frame: int = 1):
    nf = n_frames(sc)
    if arm == "direct":
        a, b = render(sc, geo, 0), render(sc, geo, nf, with_trace=False)
    else:
        t = rng.randint(0, nf - 1)
        if cf and sc.kind == "countpaint":
            ans = next((i for i, st in enumerate(sc.steps) if st[0] == "answer"), None)
            if ans is not None and t > ans and rng.random() < cf:   # an input that already shows the letter
                sc = counterfactual(sc, rng.randrange(len(sc.regions)))
        e = t + 1 if per_frame <= 1 else min(nf, t + rng.randint(1, per_frame))   # free steps: 1..per_frame at once
        a, b = render(sc, geo, t), render(sc, geo, e)
    return a, b


# ------------------------------------------------------------------------------------------------------------ train
def train_main(a) -> None:
    import torch
    from torch.utils.data import DataLoader, IterableDataset
    geo = Q.Geo(a.px)
    P.S = a.px
    # multi-GPU (torch.distributed.run): data parallel with the same global batch --batch (each of the k ranks draws
    # --batch / k pairs from its own data stream). Rank 0 alone evaluates, saves and logs.
    import torch.distributed as dist
    world = int(os.environ.get("WORLD_SIZE", "1")); rank = int(os.environ.get("RANK", "0"))
    if world > 1:
        # a long timeout: the other ranks wait at a barrier while rank 0 evaluates
        dist.init_process_group("nccl", timeout=datetime.timedelta(hours=3)); torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    a.out.mkdir(parents=True, exist_ok=True)
    task = Task(a.task, a.px)
    model, ema = P.build(a.size).to(dev), P.build(a.size).to(dev)
    ema.load_state_dict(model.state_dict()); ema.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=a.wd)
    step, ck = 0, a.out / "ckpt.pt"
    if ck.exists():
        s = torch.load(ck, map_location=dev)
        model.load_state_dict(s["model"]); ema.load_state_dict(s["ema"]); opt.load_state_dict(s["opt"]); step = s["step"]
        print(f"resumed at step {step}", flush=True)
    elif a.init:   # start from another run's EMA weights (fresh optimizer, the warm-up again): Trace Then Sum and
        # Compute Then Paint start from a Curve Tracing / Count Then Paint model (Appendix F)
        w = torch.load(a.init, map_location=dev)["ema"]
        model.load_state_dict(w); ema.load_state_dict(w)
        print(f"initialised from {a.init}", flush=True)
    net = model
    if world > 1:
        assert a.batch % world == 0, "--batch must divide by the GPU count"
        net = torch.nn.parallel.DistributedDataParallel(model, device_ids=[torch.cuda.current_device()])
    tests = task.test_sets(a.n_test)
    if rank == 0:
        (a.out / "config.json").write_text(json.dumps({"trainer": "draw_tasks", **{k: str(v) for k, v in vars(a).items()}, "tests": {k: len(v) for k, v in tests.items()},
                                                   "params": sum(p.numel() for p in model.parameters()), "world": world}, indent=1))

    class Pairs(IterableDataset):
        def __iter__(self):
            wi = torch.utils.data.get_worker_info()
            rng = random.Random(f"{a.seed}-{step}-{wi.id if wi else 0}" + (f"-r{rank}" if world > 1 else ""))
            while True:
                sc = task.train(rng)
                x, y = pair(sc, geo, a.arm, rng, a.cf_paint, task.per_frame)
                yield torch.from_numpy(x), torch.from_numpy(y)

    loader = iter(DataLoader(Pairs(), batch_size=a.batch // world, num_workers=a.workers, persistent_workers=True, prefetch_factor=4))
    to_f = lambda t: (t.to(dev, non_blocking=True).float() / 127.5 - 1)[:, None]

    @torch.no_grad()
    def rollout(scs):
        cur = to_f(torch.from_numpy(np.stack([render(sc, geo, 0) for sc in scs])))
        n_fr = [n_frames(sc) if a.arm == "video" else 1 for sc in scs]   # per scene (tally: one frame per dot)
        g = torch.Generator(device=dev).manual_seed(7)
        ts = torch.linspace(0, 1, a.sample_steps + 1, device=dev)
        final = cur.clone()
        u8 = lambda t_: ((t_ + 1) * 127.5)[:, 0].round().clamp(0, 255).byte().cpu().numpy()  # noqa: E731
        done = [False] * len(scs)
        for f in range(max(n_fr) + (6 if task.free else 0)):   # free-length tasks: until the answer is written
            x = torch.randn(cur.shape, device=dev, generator=g)
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                for k in range(a.sample_steps):
                    xh = ema(cur, x, ts[k].repeat(len(scs))).float()
                    x = x + (ts[k + 1] - ts[k]) * (xh - x) / (1 - ts[k]).clamp(min=0.05)
            cur = x.float().clamp(-1, 1)
            if task.free:
                for j, g_ in enumerate(Q.read(geo, u8(cur))):
                    if g_.strip() and not done[j]:
                        final[j], done[j] = cur[j], True
                if all(done):
                    break
            else:
                for j, nf_ in enumerate(n_fr):
                    if nf_ == f + 1:
                        final[j] = cur[j]
        if task.free:
            for j in range(len(scs)):
                if not done[j]:
                    final[j] = cur[j]
        return u8(final)

    def evaluate():
        ema.eval(); res = {"step": step, "weights": "ema"}; show = []
        y0, y1 = Q.CH, geo.q_rows * Q.CH          # the drawing area
        for name, ex in tests.items():
            ok, iou, cp = 0, [], {"iou": [], "iou80": [], "pick": [], "counts": [], "solved": []}
            for i in range(0, len(ex), a.eval_batch):
                chunk = ex[i:i + a.eval_batch]
                fin = rollout(chunk)
                got = [g.strip() for g in Q.read(geo, fin)]
                ok += sum(g == sc.answer for g, sc in zip(got, chunk))
                if task.name == "countpaint":   # the painting (IoU with the true block), the letter, the written counts
                    notes = read_row(geo, fin, geo.q_rows - 1)
                    for f, sc, g, nt in zip(fin, chunk, got, notes):
                        ious = paint_ious(sc, geo, render(sc, geo, 0), f)
                        cp["iou"].append(ious[sc.chosen]); cp["iou80"].append(ious[sc.chosen] >= 0.8)
                        cp["pick"].append(int(np.argmax(ious)) == sc.chosen and max(ious) >= 0.5)
                        want = [st[2] for st in sc.steps if st[0] in ("note", "tally")][-1]
                        cnt = nt == want if a.arm != "direct" else True
                        cp["counts"].append(nt == want)
                        cp["solved"].append(g == sc.answer and ious[sc.chosen] >= 0.8 and cnt)
                elif a.arm != "direct":   # the inked trace: IoU of the dark strokes with the ground truth's
                    for f, sc in zip(fin, chunk):
                        want = render(sc, geo, len(sc.steps) + 1)[y0:y1] < 60
                        got = f[y0:y1] < 60
                        iou.append((want & got).sum() / max(1, (want | got).sum()))
                if i == 0:
                    show.append(np.concatenate([np.concatenate([render(sc, geo, 0), f], 0) for sc, f in zip(chunk[:6], fin[:6])], 1))
            r = {"exact": ok / len(ex)}
            if iou:
                r["trace_iou"] = float(np.mean(iou))
            if cp["iou"]:
                r.update({k: float(np.mean(v)) for k, v in cp.items() if not (k == "counts" and a.arm == "direct")})
            res[name] = {k: round(v, 4) for k, v in r.items()}
        ema.train()
        with (a.out / "evals.jsonl").open("a") as f:
            f.write(json.dumps(res) + "\n")
        print(json.dumps(res), flush=True)
        (a.out / "samples").mkdir(exist_ok=True)
        Image.fromarray(np.concatenate(show[-2:], 0)).save(a.out / "samples" / f"s{step:07d}.png")

    model.train(); t0 = time.time(); s0 = step
    while step < a.steps:
        x1, y = next(loader)
        x1, y = to_f(x1), to_f(y)
        lr = a.lr * min(1.0, (step + 1) / a.warmup)
        if a.decay_from and step >= a.decay_from:   # linear cool-down to 0 at --steps
            lr *= max(0.0, (a.steps - step) / max(1, a.steps - a.decay_from))
        for g_ in opt.param_groups:
            g_["lr"] = lr
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            if a.chg_weight:   # the pixels this step changes (|target - input| > 0.1, dilated 2 px), before the context noise
                w = 1 + a.chg_weight * torch.nn.functional.max_pool2d(((y - x1).abs() > 0.1).float(), 5, stride=1, padding=2)
            if a.arm == "video" and a.ctx_noise:
                x1 = x1 + a.ctx_noise * torch.randn_like(x1)     # generated frames are never perfect
            t = torch.sigmoid(torch.randn(len(x1), device=dev))
            e = torch.randn_like(y)
            xt = t[:, None, None, None] * y + (1 - t[:, None, None, None]) * e
            xh = net(x1, xt, t).float()
            lmap = ((xh - xt) / (1 - t).clamp(min=0.05)[:, None, None, None] - (y - e)) ** 2
            # --chg-weight: most of a frame is copied from its input; the pixels the step changes carry the reasoning
            loss = (lmap * w).sum() / w.sum() if a.chg_weight else lmap.mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if torch.isfinite(gnorm):
            opt.step()
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.mul_(a.ema).add_(pm.detach(), alpha=1 - a.ema)
        step += 1
        if step % 200 == 0 and rank == 0:
            print(json.dumps({"step": step, "loss": round(float(loss), 6), "sec_per_step": round((time.time() - t0) / (step - s0), 4)}), flush=True)
        if (step % a.save_every == 0 or step == a.steps) and rank == 0:
            torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "step": step}, str(ck) + ".tmp")
            os.replace(str(ck) + ".tmp", ck)
        if step % a.eval_every == 0 or step == a.steps:
            if rank == 0:
                evaluate()
            if world > 1:
                dist.barrier()
    if rank == 0:
        (a.out / "TRAIN_DONE").write_text(str(step))
    if world > 1:
        dist.destroy_process_group()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--task", required=True, help="see Task")
    t.add_argument("--arm", choices=("direct", "video"), required=True, help="direct answering, or frame by frame")
    t.add_argument("--size", choices=sorted(P.SIZES), default="S")
    t.add_argument("--px", type=int, default=192)
    t.add_argument("--steps", type=int, default=40000)
    t.add_argument("--batch", type=int, default=256)
    t.add_argument("--eval-batch", type=int, default=64)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--wd", type=float, default=0.0)
    t.add_argument("--warmup", type=int, default=1000)
    t.add_argument("--ema", type=float, default=0.999)
    t.add_argument("--ctx-noise", type=float, default=0.03)
    t.add_argument("--sample-steps", type=int, default=10, help="Euler steps per frame in the training-time evaluation")
    t.add_argument("--n-test", type=int, default=128)
    t.add_argument("--eval-every", type=int, default=2000)
    t.add_argument("--save-every", type=int, default=1000)
    t.add_argument("--workers", type=int, default=12)
    t.add_argument("--init", type=Path, help="start from the EMA weights of another run (its ckpt.pt)")
    t.add_argument("--decay-from", type=int, default=0, help="linear lr cool-down from this step to 0 at --steps")
    t.add_argument("--cf-paint", type=float, default=0.0,
                   help="frame by frame, countpaint: this share of the pairs after the letter use a random letter (counterfactual)")
    t.add_argument("--chg-weight", type=float, default=0.0,
                   help="extra loss weight on the pixels a step changes (1 + W on them): the written counts are ~1%% of a frame")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    train_main(a)


if __name__ == "__main__":
    main()
