"""Evaluation of the video models. A video output directory holds, per item:
  DIR/<item_id>/frames/f_XXXX.png   frames sampled from the generated video, resized to the 1024x1024 card layout
  DIR/<item_id>/output.png          the final frame (same as the last sampled frame)
Judge requests (two text roles on the answer box) go to the final frame and to every sampled frame.

usage: python -m generator_benchmark.video_eval cases --gen DIR [DIR ...] --out OUT   (OUT/text_cases.jsonl, one case
           per frame; several DIRs = shards of one run)
       python -m generator_benchmark.video_eval summary --scored OUT/scored.jsonl
Metrics per item: FinalFrame Exact (last frame; the number reported in Table 1), AnyFrame Exact (any sampled frame; Unknown only if no frame is True
and some frame is Unknown), and HoldAfterFirst (fraction of sampled frames after the first correct one that stay
correct).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .cases import item_index, text_case


def build(gens: list[Path], out: Path) -> None:
    index = item_index()
    rows = []
    for gen, d in sorted((g, p) for g in gens for p in g.iterdir() if p.is_dir() and (p / "source.png").exists()):
        item = dict(index[d.name])
        if item["mode"] != "write":
            continue
        if (d / "meta.json").exists():  # the same item id can sit in several task sets; keep the one generated
            item.update({k: json.loads((d / "meta.json").read_text())[k] for k in ("task", "level")})
        frames = sorted((d / "frames").glob("f_*.png")) if (d / "frames").exists() else []
        if not frames:
            case = text_case(item, gen)
            case["case_id"] = f"{d.name}::missing"
            case["meta"].update(frame=-1, n_frames=0)
            case["only_views"] = ["text_structure", "text_transcription"]
            case["missing"] = "no frames"
            rows.append(case)
            continue
        for k, f in enumerate(frames):
            case = text_case(item, gen)
            case["case_id"] = f"{d.name}::{f.stem}"
            case["candidate"] = str(f)
            case["only_views"] = ["text_structure", "text_transcription"]
            case["meta"].update(frame=k, n_frames=len(frames))
            rows.append(case)
    out.mkdir(parents=True, exist_ok=True)
    (out / "text_cases.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"{len(rows)} frame cases")


def summarize(scored: Path) -> dict:
    per_item: dict[str, list] = defaultdict(list)
    meta = {}
    for line in scored.read_text().splitlines():
        r = json.loads(line)
        item_id = r["case_id"].split("::")[0]
        per_item[item_id].append((r["meta"].get("frame", -1), r["metrics"]["Exact"]))
        meta[item_id] = r["meta"]
    table = defaultdict(lambda: {"final": [0, 0, 0], "any": [0, 0, 0], "hold": []})
    for item_id, frames in per_item.items():
        frames.sort()
        key = f"{meta[item_id].get('task')}/{meta[item_id].get('level')}"
        values = [v for _, v in frames]
        final = values[-1]
        anyv = True if any(v is True for v in values) else (None if any(v is None for v in values) else False)
        for name, v in (("final", final), ("any", anyv)):
            table[key][name][0 if v is True else 1 if v is False else 2] += 1
        if True in values:
            first = values.index(True)
            tail = values[first:]
            table[key]["hold"].append(sum(v is True for v in tail) / len(tail))
    out = {}
    for key, t in sorted(table.items()):
        n = sum(t["final"])
        out[key] = {"n": n, "FinalFrame": f"{t['final'][0]}/{n} (U{t['final'][2]})",
                    "AnyFrame": f"{t['any'][0]}/{n} (U{t['any'][2]})",
                    "HoldAfterFirst": round(sum(t["hold"]) / len(t["hold"]), 3) if t["hold"] else None}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("cases", "summary"))
    ap.add_argument("--gen", type=Path, nargs="+")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--scored", type=Path)
    a = ap.parse_args()
    if a.command == "cases":
        build(a.gen, a.out)
    else:
        result = summarize(a.scored)
        print(json.dumps(result, indent=1))
        (a.scored.parent / "video_summary.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
