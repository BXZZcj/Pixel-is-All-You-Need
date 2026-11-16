"""The canvas of Sec. 4.1 and the answer reader of Appendix G.

Every problem is drawn on a grey PX x PX canvas (PX = 192; 256 for Word Problems) with a grid of 8 x 16 px character
cells aligned with the model's 8 x 8 patches: PX // 8 columns and PX // 16 rows (24 x 12 at 192 px). The question and
the written steps occupy the upper rows, a horizontal rule follows, and the last two rows are the answer box. Text is
English ASCII in a fixed monospaced font (Liberation Mono).

Answers are read from pixels with no judge model: each cell of the answer rows is matched to the nearest glyph of the
font, and a cell about as near to a second glyph is illegible (it never equals a gold character).
"""

from __future__ import annotations

import textwrap
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import model as P

CW, CH, FONT_PX, ANS_ROWS = 8, 16, 13, 2
ILLEGIBLE = "�"


class Geo:
    def __init__(self, px: int):
        assert px % CH == 0
        self.px, self.cols, self.rows = px, px // CW, px // CH
        self.q_rows = self.rows - 1 - ANS_ROWS               # rows above the rule
        self.line_y = self.q_rows * CH + CH // 2
        self.ans_y = (self.q_rows + 1) * CH


@lru_cache(None)
def glyphs() -> np.ndarray:
    """(95, CH, CW) uint8: every printable ASCII character in its cell; index 0, the space, is the blank cell.
    Pillow's basic layout is pinned: the raqm engine (used when libfribidi is installed) gives fractional advances and
    would draw different canvases on different machines."""
    f = ImageFont.truetype(str(P.ASSETS / "LiberationMono-Regular.ttf"), FONT_PX, layout_engine=ImageFont.Layout.BASIC)
    out = []
    for c in range(32, 127):
        im = Image.new("L", (CW, CH), 255)
        ImageDraw.Draw(im).text((0, 1), chr(c), fill=0, font=f)
        out.append(np.asarray(im))
    return np.stack(out)


def wrap_rows(text: str, rows: int, cols: int, words=True):
    """(rows, cols) glyph indices of text wrapped at word (or character) boundaries, or None if it does not fit"""
    lines = textwrap.wrap(text, cols) if words else [text[i:i + cols] for i in range(0, len(text), cols)]
    if len(lines) > rows:
        return None
    m = np.zeros((rows, cols), np.uint8)
    for r, l in enumerate(lines):
        a = np.frombuffer(l.encode("ascii"), np.uint8) - 32
        m[r, :len(a)] = a
    return m


def render(geo: Geo, m: np.ndarray) -> np.ndarray:
    """the canvas (uint8) of a (rows, cols) grid of glyph indices, with the rule above the answer box"""
    img = glyphs()[m].transpose(0, 2, 1, 3).reshape(geo.px, geo.px).copy()
    img[geo.line_y, 4:geo.px - 4] = 0
    return img


def read(geo: Geo, imgs: np.ndarray) -> list[str]:
    """imgs (B, px, px), 0..255 -> the answer written in the answer box of each canvas"""
    G = glyphs().astype(np.float32)
    B = len(imgs)
    a = imgs[:, geo.ans_y:geo.ans_y + ANS_ROWS * CH].astype(np.float32)
    cells = a.reshape(B, ANS_ROWS, CH, geo.cols, CW).transpose(0, 1, 3, 2, 4)
    d = ((cells[:, :, :, None] - G[None, None, None]) ** 2).mean((-1, -2))
    best = d.argmin(-1)
    two = np.partition(d, 1, axis=-1)
    bad = (best != 0) & (two[..., 0] > 0.7 * two[..., 1])
    out = []
    for i in range(B):
        s = "".join(ILLEGIBLE if bad[i, r, c] else chr(32 + best[i, r, c]) for r in range(ANS_ROWS) for c in range(geo.cols))
        out.append(s.rstrip())
    return out


def synth_word(rng, lo=3, hi=10) -> str:
    """a pronounceable made-up word (the training names of Word Problems): no list to memorize"""
    n, w, vowel = rng.randint(lo, hi), "", rng.random() < 0.3
    while len(w) < n:
        w += rng.choice("aeiou" if vowel else "bcdfghjklmnpqrstvwxyz")
        vowel = not vowel if rng.random() < 0.85 else vowel
    return w.capitalize()
