"""Reply schemas of the five judge roles.

The prompts (judge/prompts/*.txt) define these JSON objects; a reply that violates them is InvalidReply
(scored as Unknown). Validators return a new dict and never coerce a fact.
"""

from __future__ import annotations

import math
from typing import Any

from .judge import InvalidReply

TEXT_ONLY, CHOICE_ONLY, MIXED = "text_only", "choice_only", "mixed"
REQUIREMENTS = {TEXT_ONLY: {"text": True, "choice": False}, CHOICE_ONLY: {"text": False, "choice": True},
                MIXED: {"text": True, "choice": True}}


def _keys(value: dict[str, Any], expected: set[str], role: str) -> None:
    if set(value) != expected:
        raise InvalidReply(f"{role} keys must be exactly {sorted(expected)}, got {sorted(value)}")


def _tri(value: Any, key: str) -> None:
    if value is not None and type(value) is not bool:
        raise InvalidReply(f"{key} must be boolean or null")


def _confidence(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise InvalidReply("confidence must be a finite number")
    if not 0 <= float(value) <= 1:
        raise InvalidReply("confidence must be in [0,1]")
    return float(value)


def text_structure(value: dict[str, Any]) -> dict[str, Any]:
    _keys(value, {"text_present_in_roi", "text_extra_marks_present", "text_legible", "confidence"}, "text_structure")
    for key in ("text_present_in_roi", "text_extra_marks_present", "text_legible"):
        _tri(value[key], key)
    present, extra, legible = value["text_present_in_roi"], value["text_extra_marks_present"], value["text_legible"]
    if present is None and (extra is not None or legible is not None):
        raise InvalidReply("unknown text presence requires null dependent facts")
    if present is False and (extra is not False or legible is not None):
        raise InvalidReply("absent text requires extra=false and legible=null")
    if present is True and (type(extra) is not bool or type(legible) is not bool):
        raise InvalidReply("present text requires boolean extra/legibility facts")
    return {**value, "confidence": _confidence(value["confidence"])}


def text_transcription(value: dict[str, Any]) -> dict[str, Any]:
    _keys(value, {"observed_text", "legible", "empty", "confidence"}, "text_transcription")
    observed = value["observed_text"]
    if observed is not None and (not isinstance(observed, str) or not observed.strip()):
        raise InvalidReply("observed_text must be a non-empty string or null")
    if type(value["legible"]) is not bool or type(value["empty"]) is not bool:
        raise InvalidReply("legible and empty must be booleans")
    if value["empty"] and (observed is not None or value["legible"]):
        raise InvalidReply("empty transcription cannot contain legible text")
    if value["legible"] and (value["empty"] or observed is None):
        raise InvalidReply("legible transcription requires non-empty observed_text")
    if observed is not None and not value["legible"]:
        raise InvalidReply("a transcription requires legible=true")
    return {**value, "confidence": _confidence(value["confidence"])}


def choice_detection(value: dict[str, Any]) -> dict[str, Any]:
    _keys(value, {"choice_mark_count", "observed_choice_indices", "choice_mark_legible", "confidence"},
          "choice_detection")
    count, indices, legible = value["choice_mark_count"], value["observed_choice_indices"], value["choice_mark_legible"]
    if count is not None and (type(count) is not int or not 0 <= count <= 3):
        raise InvalidReply("choice_mark_count must be 0..3 or null")
    if not isinstance(indices, list) or any(type(i) is not int or i not in (0, 1, 2) for i in indices):
        raise InvalidReply("observed_choice_indices must contain only 0,1,2")
    if len(indices) != len(set(indices)):
        raise InvalidReply("observed_choice_indices contains duplicates")
    _tri(legible, "choice_mark_legible")
    if count is None and (indices or legible is not None):
        raise InvalidReply("unknown choice count requires empty indices and null legibility")
    if count == 0 and (indices or legible is not None):
        raise InvalidReply("zero choice marks require empty indices and null legibility")
    if count is not None and count > 0 and (len(indices) != count or type(legible) is not bool):
        raise InvalidReply("present choice marks require matching indices and boolean legibility")
    return {**value, "observed_choice_indices": sorted(indices), "confidence": _confidence(value["confidence"])}


COMPLIANCE_KEYS = ("source_text_changed", "source_graphic_changed", "wrong_region_text_present",
                   "wrong_region_mark_present", "answer_structure_changed", "choice_structure_changed",
                   "other_unrequested_content_present")


def atomic_compliance(output_mode: str):
    requirements = REQUIREMENTS[output_mode]

    def validate(value: dict[str, Any]) -> dict[str, Any]:
        _keys(value, {*COMPLIANCE_KEYS, "confidence"}, "atomic_compliance")
        for key in COMPLIANCE_KEYS:
            _tri(value[key], key)
        if not requirements["text"] and value["answer_structure_changed"] is not None:
            raise InvalidReply("answer_structure_changed must be null without a text mechanism")
        if not requirements["choice"] and value["choice_structure_changed"] is not None:
            raise InvalidReply("choice_structure_changed must be null without a choice mechanism")
        return {**value, "confidence": _confidence(value["confidence"])}

    return validate


def source_integrity(value: dict[str, Any]) -> dict[str, Any]:
    _keys(value, {"source_text_changed", "source_graphic_changed", "confidence"}, "source_integrity")
    for key in ("source_text_changed", "source_graphic_changed"):
        _tri(value[key], key)
    return {**value, "confidence": _confidence(value["confidence"])}
