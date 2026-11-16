"""Answer-box instrument: which judge views are requested for a case, and on which crops.

Views: text_structure (is there text in the answer box, are the box and its surroundings intact), text_transcription
(what exactly is written), source_integrity and atomic_compliance (is the rest of the canvas preserved). One JSON suffix
on the last text part; structure crops use half-open redaction; region_pairs atomic_compliance is skipped when every
region pair is pixel-identical (all facts False).
Every request carries only public pixels; no answer, target or condition is shown to the judge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import images, schemas
from .judge import Part

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
SUFFIX = "Return exactly one JSON object using the schema in the system instructions."
MASK_SOURCE_TEXT = ("The complete aligned canvas is shown with the same public text-answer interior replaced by an identical "
                    "solid white rectangle in both images. Assess only pre-existing content outside this mask, including "
                    "printed labels, boxes, borders and source text. The mask itself is not damage; independent newly added "
                    "content belongs to the separate extra-content endpoint.")
MASK_SOURCE_MIXED = ("Public output mode: mixed. ANSWER mechanism applicable: true. CHOICE mechanism applicable: true. "
                     "Extra PROTECTED full-canvas context: the same fixed public text-answer interior is white and each "
                     "applicable choice interior is light gray in both images. Answer glyphs, choice labels, and selection "
                     "marks are hidden. All original printed borders, ANSWER label, source text and margins remain "
                     "continuous. Evaluate observable changes outside these masks. A pre-existing border that shifts, "
                     "thickens or deforms is structure/source damage, not independent added content. Independent new "
                     "protected-region text or marks of any color are extra. Masks themselves are neither damage nor extra. "
                     "Do not judge answer correctness. Assess pre-existing visible printed content only, including all "
                     "retained labels and borders.")


def prompt(name: str) -> str:
    return (PROMPT_DIR / f"{name}.txt").read_text(encoding="utf-8").strip()


@dataclass
class Request:
    role: str            # prompt / schema name
    view: str            # unique per case; also the fact key used by score_text
    system: str
    parts: list[Part]
    validator: Callable[[dict[str, Any]], dict[str, Any]]


@dataclass
class Plan:
    requests: list[Request] = field(default_factory=list)
    deterministic: dict[str, dict[str, Any]] = field(default_factory=dict)  # view -> facts decided from pixels
    audit: dict[str, Any] = field(default_factory=dict)


def _with_suffix(parts: list[Part]) -> list[Part]:
    """Append the JSON instruction to the last text part (as the service instrument did)."""
    last = max(i for i, p in enumerate(parts) if p.text is not None)
    return [Part.t(p.text + "\n" + SUFFIX) if i == last else p for i, p in enumerate(parts)]


def plan_case(case: dict[str, Any], work: Path) -> Plan:
    """case: {case_id, output_mode, source, candidate, rois:{source, protected, answer, text_answer_inner, choices},
    label_rois (mixed only)}. Crops are written under work/."""
    mode = case["output_mode"]
    req = schemas.REQUIREMENTS[mode]
    src, cand = Path(case["source"]), Path(case["candidate"])
    rois = case["rois"]
    work.mkdir(parents=True, exist_ok=True)
    plan = Plan()

    def add(role, view, parts, validator):
        plan.requests.append(Request(role, view, prompt(role), _with_suffix(parts), validator))

    # 1-2 answer text roles on the native inner answer crop
    if req["text"]:
        inner = work / "text_roi.png"
        images.crop_native(cand, rois["text_answer_inner"], inner)
        add("text_structure", "text_structure", [Part.t("Native candidate text-answer ROI only:"), Part.img(inner)],
            schemas.text_structure)
        add("text_transcription", "text_transcription", [Part.t("The same native candidate text-answer ROI only:"), Part.img(inner)],
            schemas.text_transcription)
    if case.get("only_views") and set(case["only_views"]) <= {"text_structure", "text_transcription"}:
        plan.audit["structure_pixel_equal"] = {}  # text roles only: skip the preservation views entirely
        return plan
    # 3 choice detection on the three full native choice cells
    if req["choice"]:
        parts = []
        for i, cell in enumerate(rois["choices"]):
            path = work / f"choice_{i}.png"
            images.crop_native(cand, cell, path)
            parts += [Part.t(f"Native candidate CHOICE ROI {i}:"), Part.img(path)]
        add("choice_detection", "choice_rois", parts, schemas.choice_detection)

    # 4 region-pair compliance (answer/choice structure, plus placement diagnostics)
    pairs, equal = [], True
    for label, roi in (("SOURCE REGION", rois["source"]), ("PROTECTED REGION", rois["protected"])):
        a, b = work / f"rp_{len(pairs)}_s.png", work / f"rp_{len(pairs)}_c.png"
        equal &= images.crop_native(src, roi, a) == images.crop_native(cand, roi, b)
        pairs.append((label, "", a, b))
    structure_equal = {}
    if req["text"]:
        a, b = work / "struct_answer_s.png", work / "struct_answer_c.png"
        eq = (images.crop_structure(src, rois["answer"], [rois["text_answer_inner"]], a, images.WHITE)
              == images.crop_structure(cand, rois["answer"], [rois["text_answer_inner"]], b, images.WHITE))
        structure_equal["answer"] = eq
        equal &= eq
        pairs.append(("ANSWER STRUCTURE", " (content redacted)", a, b))
    if req["choice"]:
        eqs = []
        for i, cell in enumerate(rois["choices"]):
            a, b = work / f"struct_choice{i}_s.png", work / f"struct_choice{i}_c.png"
            redact = [images.choice_content_roi(cell)]
            eqs.append(images.crop_structure(src, cell, redact, a, images.CHOICE_FILL)
                       == images.crop_structure(cand, cell, redact, b, images.CHOICE_FILL))
            pairs.append((f"CHOICE {i} STRUCTURE", " (content redacted)", a, b))
        structure_equal["choice"] = all(eqs)
        equal &= all(eqs)
    plan.audit["structure_pixel_equal"] = structure_equal
    if equal:
        plan.deterministic["region_pairs"] = {k: False for k in schemas.COMPLIANCE_KEYS}
        if not req["text"]:
            plan.deterministic["region_pairs"]["answer_structure_changed"] = None
        if not req["choice"]:
            plan.deterministic["region_pairs"]["choice_structure_changed"] = None
    else:
        parts = [Part.t(f"Public output mode: {mode}. ANSWER mechanism applicable: {str(req['text']).lower()}. "
                        f"CHOICE mechanism applicable: {str(req['choice']).lower()}.")]
        for label, note, a, b in pairs:
            parts += [Part.t(f"{label} — SOURCE{note}:"), Part.img(a), Part.t(f"{label} — CANDIDATE{note}:"), Part.img(b)]
        add("atomic_compliance", "region_pairs", parts, schemas.atomic_compliance(mode))

    # 5 source integrity on the source ROI
    a, b, d = work / "src_s.png", work / "src_c.png", work / "src_diff.png"
    if images.crop_native(src, rois["source"], a) == images.crop_native(cand, rois["source"], b):
        plan.deterministic["source_roi"] = {"source_text_changed": False, "source_graphic_changed": False}
    else:
        images.difference_gain8(a, b, d)
        add("source_integrity", "source_roi",
            [Part.t("Native SOURCE ROI — SOURCE:"), Part.img(a), Part.t("The same native SOURCE ROI — CANDIDATE:"),
             Part.img(b), Part.t("Enhanced absolute RGB difference (8x clipped; black means unchanged):"), Part.img(d)],
            schemas.source_integrity)

    # 6 source integrity on the full canvas with the answer (and choice) interiors masked
    masks = [(rois["text_answer_inner"], images.WHITE)] if req["text"] else []
    masks += [(images.choice_content_roi(cell), images.CHOICE_FILL) for cell in rois.get("choices", [])]
    a, b, d = work / "masked_s.png", work / "masked_c.png", work / "masked_diff.png"
    if images.masked_canvas(src, masks, a) == images.masked_canvas(cand, masks, b):
        plan.deterministic["masked_canvas"] = {"source_text_changed": False, "source_graphic_changed": False}
    else:
        images.difference_gain8(a, b, d)
        if mode == schemas.MIXED:
            parts = [Part.t(MASK_SOURCE_MIXED), Part.t("Masked ORIGINAL:"), Part.img(a), Part.t("Masked CANDIDATE:"),
                     Part.img(b), Part.t("Enhanced absolute RGB difference (8x clipped):"), Part.img(d)]
        else:
            parts = [Part.t(MASK_SOURCE_TEXT), Part.t("Native protected region — SOURCE:"), Part.img(a),
                     Part.t("The same native protected region — CANDIDATE:"), Part.img(b),
                     Part.t("Enhanced absolute RGB difference (8x clipped; black means unchanged):"), Part.img(d)]
        add("source_integrity", "masked_canvas", parts, schemas.source_integrity)

    # 7 printed choice labels (hidden by the canvas mask) are checked one by one
    for i, roi in enumerate(case.get("label_rois") or []):
        a, b, d = work / f"label{i}_s.png", work / f"label{i}_c.png", work / f"label{i}_diff.png"
        if images.crop_native(src, roi, a) == images.crop_native(cand, roi, b):
            plan.deterministic[f"choice_label_{i}"] = {"source_text_changed": False, "source_graphic_changed": False}
        else:
            images.difference_gain8(a, b, d)
            add("source_integrity", f"choice_label_{i}",
                [Part.t("Native protected region — SOURCE:"), Part.img(a),
                 Part.t("The same native protected region — CANDIDATE:"), Part.img(b),
                 Part.t("Enhanced absolute RGB difference (8x clipped; black means unchanged):"), Part.img(d)],
                schemas.source_integrity)
    return plan
