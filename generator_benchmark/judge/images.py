"""Pixel operations of the measurement instrument (crops, redaction, pixel hashes; half-open xyxy everywhere, no
resizing).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageChops, ImageDraw

WHITE = (255, 255, 255)
CHOICE_FILL = (248, 250, 252)


def pixel_hash(image: Image.Image) -> str:
    return hashlib.sha256(f"{image.width}x{image.height}:{image.mode}\n".encode("ascii") + image.tobytes()).hexdigest()


def file_pixel_hash(path: Path) -> str:
    with Image.open(path) as image:
        image.load()
        return pixel_hash(image)


def _box(roi: Sequence[int], size: tuple[int, int]) -> tuple[int, int, int, int]:
    if len(roi) != 4 or any(type(v) is not int for v in roi):
        raise ValueError(f"ROI must be four ints: {roi}")
    x0, y0, x1, y1 = roi
    if not (0 <= x0 < x1 <= size[0] and 0 <= y0 < y1 <= size[1]):
        raise ValueError(f"ROI {list(roi)} outside {size}")
    return x0, y0, x1, y1


def _save(image: Image.Image, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, format="PNG", optimize=False, compress_level=6)
    return pixel_hash(image)


def crop_native(source: Path, roi: Sequence[int], destination: Path) -> str:
    """Crop without mode conversion. Returns the crop's pixel hash."""
    with Image.open(source) as image:
        image.load()
        return _save(image.crop(_box(roi, image.size)), destination)


def crop_structure(source: Path, outer: Sequence[int], redactions: Sequence[Sequence[int]], destination: Path,
                   fill: tuple[int, int, int]) -> str:
    """RGB crop of `outer` with each redaction box (absolute, half-open) filled."""
    with Image.open(source) as image:
        rgb = image.convert("RGB")
    x0, y0, x1, y1 = _box(outer, rgb.size)
    crop = rgb.crop((x0, y0, x1, y1))
    draw = ImageDraw.Draw(crop)
    for rx0, ry0, rx1, ry1 in redactions:
        if not (x0 <= rx0 < rx1 <= x1 and y0 <= ry0 < ry1 <= y1):
            raise ValueError("redaction outside structure crop")
        draw.rectangle((rx0 - x0, ry0 - y0, rx1 - x0 - 1, ry1 - y0 - 1), fill=fill)
    return _save(crop, destination)


def masked_canvas(source: Path, masks: Sequence[tuple[Sequence[int], tuple[int, int, int]]], destination: Path) -> str:
    with Image.open(source) as image:
        canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for roi, fill in masks:
        x0, y0, x1, y1 = _box(roi, canvas.size)
        draw.rectangle((x0, y0, x1 - 1, y1 - 1), fill=fill)
    return _save(canvas, destination)


def difference_gain8(a: Path, b: Path, destination: Path) -> str:
    with Image.open(a) as ia, Image.open(b) as ib:
        if ia.size != ib.size:
            raise ValueError("difference inputs differ in size")
        diff = ImageChops.difference(ia.convert("RGB"), ib.convert("RGB"))
    return _save(diff.point([min(255, v * 8) for v in range(256)] * 3), destination)


def choice_content_roi(cell: Sequence[int]) -> list[int]:
    x0, y0, x1, y1 = cell
    margin = max(4, min(x1 - x0, y1 - y0) // 30)
    return [x0 + margin, y0 + margin, x1 - margin, y1 - margin]


def printed_label_rois(source: Path, cells: Sequence[Sequence[int]]) -> list[list[int]]:
    """Bounding box (+2 px) of dark printed label ink inside each choice cell, from the ORIGINAL card only."""
    with Image.open(source) as raw:
        image = raw.convert("RGB")
    out = []
    for x0, y0, x1, y1 in cells:
        inner = [x0 + 8, y0 + 8, x1 - 8, y1 - 8]
        crop = image.crop(tuple(inner))
        mask = Image.new("1", crop.size)
        mask.putdata([max(p) < 120 for p in crop.getdata()])
        bounds = mask.getbbox()
        if bounds is None:
            raise ValueError("choice cell has no printed label ink")
        left, top, right, bottom = bounds
        out.append([max(inner[0], inner[0] + left - 2), max(inner[1], inner[1] + top - 2),
                    min(inner[2], inner[0] + right + 2), min(inner[3], inner[1] + bottom + 2)])
    return out
