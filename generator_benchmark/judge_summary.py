"""Exact / Presence / Legible per task and level from judged rows (fixed denominators). Exact is the reported accuracy.

usage: python -m generator_benchmark.judge_summary --scored DIR/scored.jsonl [--out summary.json]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def contains(row: dict) -> bool | None:
    """Lenient: the expected answer occurs as a whole word/number in the judge's transcription (closed models often
    write a short sentence, e.g. 'In the cellar.' for 'cellar'). Unknown when the transcription is unknown."""
    import re
    import unicodedata
    m, d = row["metrics"], row.get("diagnostics") or {}
    if m.get("Presence") is False or m.get("Legible") is False:
        return False
    observed, expected = d.get("observed_text"), row["meta"].get("target")
    if observed is None or expected is None:
        return None
    norm = lambda t: " ".join(re.findall(r"[0-9a-z]+", unicodedata.normalize("NFKC", t).casefold()))
    return f" {norm(expected)} " in f" {norm(observed)} "


def summarize(path: Path) -> dict:
    table = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))  # key -> metric -> [true, false, unknown]
    for line in path.read_text().splitlines():
        r = json.loads(line)
        key = f"{r['meta'].get('task')}/{r['meta'].get('level')}"
        metrics = dict(r["metrics"])
        metrics["Contains"] = contains(r)
        for metric, value in metrics.items():
            table[key][metric][0 if value is True else 1 if value is False else 2] += 1
    return {k: {m: {"true": v[0], "false": v[1], "unknown": v[2], "rate": v[0] / sum(v)} for m, v in ms.items()}
            for k, ms in sorted(table.items())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    result = summarize(a.scored)
    print(json.dumps({k: f"Exact {v['Exact']['true']} Contains {v['Contains']['true']} / "
                      f"{v['Exact']['true'] + v['Exact']['false'] + v['Exact']['unknown']}" for k, v in result.items()}))
    if a.out:
        a.out.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
