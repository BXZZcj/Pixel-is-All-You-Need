"""Turn a generation directory into judge case lists.

usage: python -m generator_benchmark.cases --gen GEN_DIR --out DIR [--text-only]
Writes DIR/text_cases.jsonl (write items; the answer-box instrument of judge/run_text.py) and DIR/select_cases.jsonl
(select items; not part of the reported benchmark). The judge's reference image is always the blank canvas. A planned
item without an output stays in the list as "missing" and counts as Unknown.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import cards, tasks


def item_index() -> dict[str, dict]:
    return {i["item_id"]: i for d in tasks.build_all().values() for i in d["test"] + d["train"]}


def text_case(item: dict, gen: Path) -> dict:
    d = gen / item["item_id"]
    blank = d / "judge_source.png"
    if not blank.exists() and d.exists():
        cards.write_card(item["card_text"], item["prefix"] if item.get("original_prefix") else "").save(blank)
    truth = {"expected_text": item["expected"], "casefold": item["casefold"]}
    if item["original_prefix"]:
        truth["original_prefix"] = item["original_prefix"]
    rois = cards.WRITE_ROIS
    case = {"case_id": item["item_id"], "output_mode": "text_only", "source": str(blank), "candidate": str(d / "output.png"),
            "rois": {"source": rois.source, "protected": rois.protected, "answer": rois.answer,
                     "text_answer_inner": rois.text_answer_inner, "choices": []},
            "truth": truth, "meta": {k: item[k] for k in ("item_id", "task", "level", "fact", "target", "card_text")}}
    if not (d / "output.png").exists():
        case["missing"] = "no output.png"
    return case


def select_case(item: dict, gen: Path) -> dict:
    d = gen / item["item_id"]
    blank = d / "judge_source.png"
    if not blank.exists() and d.exists():
        cards.select_card(item["card_text"], item["options"]).save(blank)
    case = {"case_id": item["item_id"], "condition": "S_select", "source": str(blank), "candidate": str(d / "output.png"),
            "answer_roi_xyxy": [cards.OPTION_ROIS[0][0], cards.OPTION_ROIS[0][1], cards.OPTION_ROIS[2][2], cards.OPTION_ROIS[0][3]],
            "choice_rois_xyxy": cards.OPTION_ROIS, "truth": {"answer": item["target"], "expected_index": item["correct_index"]},
            "meta": {k: item[k] for k in ("item_id", "task", "level", "fact", "target", "card_text")}}
    if not (d / "output.png").exists():
        case["missing"] = "no output.png"
    return case


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--text-only", action="store_true", help="only the two text roles (Exact; no preservation views)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    index = item_index()
    planned = [json.loads(l)["item_id"] for l in (a.gen / "manifest.jsonl").read_text().splitlines() if l.strip()]
    planned += [p.name for p in a.gen.iterdir() if p.is_dir() and (p / "source.png").exists()]  # incl. failed API items
    write, select = [], []
    for item_id in sorted(set(planned)):
        item = index[item_id]
        (select if item["mode"] == "select" else write).append(select_case(item, a.gen) if item["mode"] == "select" else text_case(item, a.gen))
    if a.text_only:
        for row in write:
            row["only_views"] = ["text_structure", "text_transcription"]
    for name, rows in (("text_cases", write), ("select_cases", select)):
        (a.out / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(json.dumps({"text": len(write), "select": len(select)}))


if __name__ == "__main__":
    main()
