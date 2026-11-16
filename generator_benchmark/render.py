"""Render benchmark items to (input, target) canvas images."""

from __future__ import annotations

from PIL import Image

from . import cards


def render_input(item: dict) -> Image.Image:
    if item["mode"] == "select":
        return cards.select_card(item["card_text"], item["options"])
    return cards.write_card(item["card_text"], item["prefix"], scene=item.get("scene"))


def render_target(item: dict) -> Image.Image:
    if item["mode"] == "select":
        return cards.select_card(item["card_text"], item["options"], mark=item["correct_index"])
    return cards.write_card(item["card_text"], item["target"], scene=item.get("scene"))


def render_blank(item: dict) -> Image.Image:
    """Same layout and same prefix, but without the question text."""
    return cards.write_card("", item["prefix"], scene=item.get("scene"))
