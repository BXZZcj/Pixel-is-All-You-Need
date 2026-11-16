"""The benchmark canvases ("cards", 1024x1024).

A card has a title, a task text and either one ANSWER box (write tasks) or three option cells (select tasks).
Geometry is fixed, so every ROI is a constant. Answers use a monospace font on a fixed character grid, so a character
stays at the same pixel position whatever follows it: appending one character changes only that character's cell.

Half-open xyxy coordinates everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SIZE = 1024
ASSETS = Path(__file__).resolve().parents[1] / "assets"
INK = (20, 24, 32)
FRAME = (40, 48, 64)
MARK = (220, 30, 30)
WHITE = (255, 255, 255)

# Layout (half-open xyxy)
TITLE_XY = (64, 44)
SOURCE_ROI = [56, 110, 968, 625]       # task text region (pre-existing content the model must keep)
ANSWER_ROI = [72, 737, 952, 942]       # outer answer box including its printed border
ANSWER_INNER = 12                      # interior inset of the answer box
ANSWER_INNER_ROI = [ANSWER_ROI[0] + ANSWER_INNER, ANSWER_ROI[1] + ANSWER_INNER,
                    ANSWER_ROI[2] - ANSWER_INNER, ANSWER_ROI[3] - ANSWER_INNER]
PROTECTED_ROI = [SOURCE_ROI[0], SOURCE_ROI[3], SOURCE_ROI[2], ANSWER_ROI[1]]
OPTION_ROIS = [[67, 655, 354, 911], [368, 655, 655, 911], [669, 655, 956, 911]]  # the three option cells
OPTION_MARK_INSET = 10                 # the correct-option mark is drawn this far inside the cell
OPTION_MARK_TOP = 40                   # below the printed OPTION label
SOURCE_PROTECTED_SELECT = [SOURCE_ROI[0], SOURCE_ROI[3], SOURCE_ROI[2], OPTION_ROIS[0][1]]

# Answer character grid
ANSWER_FONT_SIZE = 64
ANSWER_LEFT = ANSWER_INNER_ROI[0] + 16
ANSWER_MAX_CHARS = 21
TEXT_FONT_SIZE = 40
OPTION_FONT_SIZE = 36
TITLE_FONT_SIZE = 30
LABEL_FONT_SIZE = 24
LINE_GAP = 12


def _font(name: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(ASSETS / name), size, layout_engine=ImageFont.Layout.BASIC)


SANS = "LiberationSans-Regular.ttf"
BOLD = "LiberationSans-Bold.ttf"
MONO = "LiberationMono-Regular.ttf"


def wrap(text: str, font: ImageFont.FreeTypeFont, width: int) -> list[str]:
    """Greedy word wrap; explicit newlines are kept. Raises if one word does not fit."""
    lines: list[str] = []
    for para in text.split("\n"):
        current = ""
        for word in para.split(" "):
            trial = f"{current} {word}" if current else word
            if font.getlength(trial) <= width:
                current = trial
            else:
                if not current:
                    raise ValueError(f"word too wide for card: {word!r}")
                lines.append(current)
                current = word
                if font.getlength(current) > width:
                    raise ValueError(f"word too wide for card: {word!r}")
        lines.append(current)
    return lines


SCENE_BOX = [72, 330, 952, 615]        # picture area inside SOURCE_ROI for visual items (text goes above it)
SHAPE_COLORS = {"red": (220, 40, 40), "blue": (40, 90, 220), "green": (30, 160, 60), "yellow": (235, 200, 20),
                "purple": (140, 60, 190), "orange": (245, 130, 20)}


def draw_shape(draw: ImageDraw.ImageDraw, shape: str, color: str, cx: int, cy: int, r: int) -> None:
    fill = SHAPE_COLORS[color]
    if shape == "circle":
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=fill)
    elif shape == "square":
        draw.rectangle((cx - r, cy - r, cx + r, cy + r), fill=fill)
    elif shape == "triangle":
        draw.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=fill)
    elif shape == "star":
        import math
        pts = [(cx + (r if k % 2 == 0 else r * 0.45) * math.sin(k * math.pi / 5),
                cy - (r if k % 2 == 0 else r * 0.45) * math.cos(k * math.pi / 5)) for k in range(10)]
        draw.polygon(pts, fill=fill)
    else:
        raise ValueError(shape)


def _draw_task_text(draw: ImageDraw.ImageDraw, text: str, region: list[int] | None = None) -> None:
    font = _font(SANS, TEXT_FONT_SIZE)
    x0, y0, x1, y1 = region or SOURCE_ROI
    lines = wrap(text, font, x1 - x0 - 32)
    line_h = TEXT_FONT_SIZE + LINE_GAP
    total = line_h * len(lines) - LINE_GAP
    if total > y1 - y0 - 16:
        raise ValueError(f"task text too tall for card ({len(lines)} lines): {text!r}")
    y = y0 + (y1 - y0 - total) // 2
    for line in lines:
        draw.text((x0 + 16, y), line, fill=INK, font=font)
        y += line_h


def _base(title: str, text: str, scene: list[dict] | None = None) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (SIZE, SIZE), WHITE)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((28, 28, SIZE - 29, SIZE - 29), radius=24, outline=FRAME, width=4)
    draw.text(TITLE_XY, title, fill=FRAME, font=_font(BOLD, TITLE_FONT_SIZE))
    draw.line((56, 96, 968, 96), fill=FRAME, width=2)
    if scene:
        _draw_task_text(draw, text, [SOURCE_ROI[0], SOURCE_ROI[1], SOURCE_ROI[2], SCENE_BOX[1] - 10])
        for obj in scene:
            draw_shape(draw, obj["shape"], obj["color"], obj["x"], obj["y"], obj["r"])
    else:
        _draw_task_text(draw, text)
    return image, draw


def char_cell(index: int) -> tuple[int, int]:
    """Left x of character `index` on the answer grid and the common top y."""
    advance = _font(MONO, ANSWER_FONT_SIZE).getlength("M")
    return (ANSWER_LEFT + round(index * advance),
            ANSWER_INNER_ROI[1] + (ANSWER_INNER_ROI[3] - ANSWER_INNER_ROI[1] - ANSWER_FONT_SIZE) // 2 - 4)


def write_card(text: str, answer: str = "", title: str = "QUESTION CARD", scene: list[dict] | None = None) -> Image.Image:
    """Write-task card. `answer` is the text already in the box (empty for the blank input card)."""
    if len(answer) > ANSWER_MAX_CHARS:
        raise ValueError(f"answer longer than {ANSWER_MAX_CHARS} chars: {answer!r}")
    if answer != answer.strip():
        raise ValueError(f"answer has outer whitespace: {answer!r}")
    image, draw = _base(title, text, scene)
    x0, y0, x1, y1 = ANSWER_ROI
    draw.rounded_rectangle((x0, y0, x1 - 1, y1 - 1), radius=10, outline=FRAME, width=3)
    draw.text((x0 + 4, y0 - 32), "Write the answer here:", fill=FRAME, font=_font(SANS, LABEL_FONT_SIZE))
    font = _font(MONO, ANSWER_FONT_SIZE)
    for i, ch in enumerate(answer):
        if ch != " ":
            draw.text(char_cell(i), ch, fill=INK, font=font)
    return image


def option_fits(option: str) -> bool:
    x0, _, x1, _ = OPTION_ROIS[0]
    return _font(SANS, OPTION_FONT_SIZE).getlength(option) <= x1 - x0 - 48


def select_card(text: str, options: list[str], mark: int | None = None, title: str = "QUESTION CARD") -> Image.Image:
    """Select-task card with three options; `mark` draws the red outline in that option cell (the target image)."""
    if len(options) != 3:
        raise ValueError("select cards have exactly three options")
    image, draw = _base(title, text)
    font = _font(SANS, OPTION_FONT_SIZE)
    for i, (roi, option) in enumerate(zip(OPTION_ROIS, options)):
        x0, y0, x1, y1 = roi
        draw.rounded_rectangle((x0, y0, x1 - 1, y1 - 1), radius=10, outline=FRAME, width=3)
        draw.text((x0 + 12, y0 + 8), f"OPTION {i}", fill=FRAME, font=_font(SANS, LABEL_FONT_SIZE))
        width = font.getlength(option)
        if not option_fits(option):
            raise ValueError(f"option too wide: {option!r}")
        draw.text((x0 + (x1 - x0 - width) / 2, y0 + (y1 - y0 - OPTION_FONT_SIZE) // 2 + 10), option, fill=INK, font=font)
    if mark is not None:
        x0, y0, x1, y1 = OPTION_ROIS[mark]
        inset = OPTION_MARK_INSET
        draw.rectangle((x0 + inset, y0 + OPTION_MARK_TOP, x1 - inset - 1, y1 - inset - 1), outline=MARK, width=6)
    return image


@dataclass(frozen=True)
class Rois:
    source: list[int]
    protected: list[int]
    answer: list[int]
    text_answer_inner: list[int]
    choices: list[list[int]]


WRITE_ROIS = Rois(SOURCE_ROI, PROTECTED_ROI, ANSWER_ROI, ANSWER_INNER_ROI, [])
