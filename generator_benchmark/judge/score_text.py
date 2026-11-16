"""Metrics for answer-box cases (image outputs and video final frames). Exact is the accuracy reported in the paper.

  * Each metric depends only on the judge views it reads. A failed view makes those metrics Unknown and nothing else.
  * answer/choice structure change: pixel-identical redacted crops => unchanged; otherwise the judge's fact, and a
    null fact stays Unknown.
  * The answer comparison does not depend on the structure role.
  * Text normalisation: NFKC, all whitespace removed; case-folded only where the task is not a copy task.
Definitions (tri-state; False dominates, then Unknown):
  Presence  = text_structure.text_present_in_roi                           [mixed: ∧ choice_mark_count > 0]
  Legible   = text_transcription.legible                                    [mixed: ∧ count > 0 ∧ choice_mark_legible]
  Exact     = present ∧ legible ∧ text matches ∧ ¬extra marks in the box     [mixed: ∧ one legible mark on the key]
  Preserved = source ROI unchanged ∧ answer/choice structure unchanged ∧ masked canvas unchanged
              [∧ printed choice labels unchanged] [∧ items with a printed prefix: prefix kept]
  Strict    = Exact ∧ Preserved                                             [mixed: ∧ text/choice consistent]
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Iterable

METRICS = ("Presence", "Legible", "Exact", "Preserved", "Strict")


def tri_and(values: Iterable[bool | None]) -> bool | None:
    values = list(values)
    if any(v is False for v in values):
        return False
    if any(v is None for v in values):
        return None
    return True


def neg(value: bool | None) -> bool | None:
    return None if value is None else not value


def normalize(text: str | None, *, casefold: bool) -> str | None:
    if text is None:
        return None
    out = re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))
    return out.casefold() if casefold else out


def score(facts: dict[str, dict[str, Any] | None], truth: dict[str, Any], *, output_mode: str,
          structure_pixel_equal: dict[str, bool]) -> dict[str, Any]:
    """facts: view -> validated facts, or None when that view failed (Unknown). Views: text_structure,
    text_transcription, choice_rois, region_pairs, source_roi, masked_canvas, choice_label_<i>.
    truth: expected_text, casefold, [correct_choice_index, choice_texts], [original_prefix]."""
    mixed = output_mode == "mixed"
    S, T = facts.get("text_structure"), facts.get("text_transcription")
    C, AC = facts.get("choice_rois"), facts.get("region_pairs")
    fold = truth["casefold"]

    present = None if S is None else S["text_present_in_roi"]
    extra = None if S is None else S["text_extra_marks_present"]
    legible = None if T is None else T["legible"]
    observed = T["observed_text"] if T is not None and T["legible"] else None
    if T is None:
        match = None
    elif T["empty"]:
        match = False          # blank box: the answer is not there
    elif observed is None:
        match = None           # ink present but unreadable
    else:
        match = normalize(observed, casefold=fold) == normalize(truth["expected_text"], casefold=fold)
    text_exact = tri_and((present, legible, match, neg(extra)))

    if mixed:
        count = None if C is None else C["choice_mark_count"]
        has_mark = None if count is None else count > 0
        one_mark = None if count is None else count == 1
        c_legible = None if C is None else C["choice_mark_legible"]
        if C is not None and count == 1 and c_legible is True:
            index = C["observed_choice_indices"][0]
            index_match = index == truth["correct_choice_index"]
            consistent = None if observed is None else (
                normalize(observed, casefold=fold) == normalize(truth["choice_texts"][index], casefold=fold))
        else:
            index_match = consistent = None
        choice_exact = tri_and((one_mark, c_legible, index_match))
        presence = tri_and((present, has_mark))
        legible_metric = tri_and((legible, has_mark, c_legible))
        exact = tri_and((text_exact, choice_exact))
    else:
        presence, legible_metric, exact, consistent = present, legible, text_exact, True

    # Preservation terms
    def unchanged(view):
        f = facts.get(view)
        return None if f is None else tri_and((neg(f["source_text_changed"]), neg(f["source_graphic_changed"])))

    terms = {"source_roi": unchanged("source_roi"), "masked_canvas": unchanged("masked_canvas")}
    if structure_pixel_equal.get("answer"):
        terms["answer_structure"] = True
    elif "answer" in structure_pixel_equal:
        terms["answer_structure"] = None if AC is None else neg(AC["answer_structure_changed"])
    if mixed:
        if structure_pixel_equal.get("choice"):
            terms["choice_structure"] = True
        else:
            terms["choice_structure"] = None if AC is None else neg(AC["choice_structure_changed"])
        for view in sorted(v for v in facts if v.startswith("choice_label_")):
            f = facts[view]
            terms[view] = None if f is None else neg(f["source_text_changed"])
    if truth.get("original_prefix"):
        if T is None:
            prefix = None
        elif T["empty"]:
            prefix = False
        elif observed is None:
            prefix = None
        else:
            prefix = normalize(observed, casefold=fold).startswith(normalize(truth["original_prefix"], casefold=fold))
        terms["old_prefix"] = prefix
    preserved = tri_and(terms.values())

    strict = tri_and((exact, preserved, consistent))
    metrics = {"Presence": presence, "Legible": legible_metric, "Exact": exact, "Preserved": preserved, "Strict": strict}
    diagnostics = {
        "observed_text": observed, "text_match": match, "extra_marks": extra, "preserved_terms": terms,
        "role_disagreement": None if S is None or T is None else bool(
            (present is True and T["empty"]) or (present is False and not T["empty"])
            or (present is True and S["text_legible"] is not None and S["text_legible"] != T["legible"])),
        "wrong_region_text": None if AC is None else AC["wrong_region_text_present"],
        "wrong_region_mark": None if AC is None else AC["wrong_region_mark_present"],
        "other_unrequested": None if AC is None else AC["other_unrequested_content_present"],
    }
    if mixed:
        diagnostics.update(choice_mark_count=None if C is None else C["choice_mark_count"],
                           choice_index_match=index_match, text_choice_consistent=consistent)
    return {"metrics": metrics, "diagnostics": diagnostics}
