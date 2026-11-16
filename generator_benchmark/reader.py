"""Local diagnostic reader for answers on the fixed character grid. NOT the evaluator of record: it gives
instant numbers while the judge (qwen3.8-max) works through its queue, and its agreement with the judge is measured.

Each grid cell of the answer box is compared with every candidate glyph rendered in the same font at that cell, with a
small position tolerance; the best normalised correlation wins. A cell without enough ink ends the string.
"""

from __future__ import annotations

import string

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import cards

CHARSET = string.ascii_letters + string.digits + "-"
INK_THRESHOLD = 150           # darker than this counts as ink (0-255)
MIN_INK_PIXELS = 40           # fewer dark pixels in a cell: treated as empty
SHIFT = 6                     # search +-SHIFT px around the grid position
MIN_SCORE = 0.45              # below this a cell is unreadable ("?")


def _advance() -> float:
    return ImageFont.truetype(str(cards.ASSETS / cards.MONO), cards.ANSWER_FONT_SIZE, layout_engine=ImageFont.Layout.BASIC).getlength("M")


def _glyphs() -> dict[str, np.ndarray]:
    font = ImageFont.truetype(str(cards.ASSETS / cards.MONO), cards.ANSWER_FONT_SIZE, layout_engine=ImageFont.Layout.BASIC)
    height = cards.ANSWER_FONT_SIZE + 24
    width = round(_advance())
    out = {}
    for ch in CHARSET:
        image = Image.new("L", (width, height), 255)
        ImageDraw.Draw(image).text((0, 4), ch, fill=0, font=font)
        out[ch] = (np.asarray(image, dtype=np.float32) < INK_THRESHOLD).astype(np.float32)
    return out


GLYPHS = _glyphs()


def _score(patch: np.ndarray, glyph: np.ndarray) -> float:
    a, b = patch - patch.mean(), glyph - glyph.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom else 0.0


def read_answer(image: Image.Image) -> str:
    """Transcribe the answer box of a generated write card."""
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    ink = (gray < INK_THRESHOLD).astype(np.float32)
    h = cards.ANSWER_FONT_SIZE + 24
    w = round(_advance())
    text = ""
    for i in range(cards.ANSWER_MAX_CHARS):
        x, y = cards.char_cell(i)
        y -= 4
        cell = ink[y:y + h, x:x + w]
        if cell.shape[1] < w or cell.sum() < MIN_INK_PIXELS:
            # one empty cell may be a space inside the answer; two in a row end it
            x2, _ = cards.char_cell(i + 1)
            next_cell = ink[y:y + h, x2:x2 + w]
            if next_cell.shape[1] < w or next_cell.sum() < MIN_INK_PIXELS:
                break
            text += " "
            continue
        best, best_score = "?", MIN_SCORE
        for ch, glyph in GLYPHS.items():
            for dy in range(-SHIFT, SHIFT + 1, 2):
                for dx in range(-SHIFT, SHIFT + 1, 2):
                    y0, x0 = y + dy, x + dx
                    if y0 < 0 or x0 < 0:
                        continue
                    patch = ink[y0:y0 + h, x0:x0 + w]
                    if patch.shape != glyph.shape:
                        continue
                    score = _score(patch, glyph)
                    if score > best_score:
                        best, best_score = ch, score
        text += best
    return text
