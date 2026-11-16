"""Run the answer-box instrument over a case list and score it with the judge (Qwen3.8-Max).

usage: python -m generator_benchmark.judge.run_text --cases cases.jsonl --work DIR [--workers 24] [--judge-model qwen3.8-max]
Endpoint: JUDGE_BASE_URL (or --base-url), key: JUDGE_API_KEY.

cases.jsonl rows: {case_id, output_mode, source, candidate, rois, [label_rois], truth, [only_views], meta}
  truth = {expected_text, casefold, [correct_choice_index, choice_texts], [original_prefix]}
Outputs under DIR: crops/<case_id>/, cache/ (judge records), scored.jsonl (one row per case, input order), run.json.
A case whose planning fails (unreadable image, bad ROI) is kept with status "case_error" and all metrics Unknown.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from .instrument_text import plan_case
from .judge import Judge, JudgeConfig
from .score_text import METRICS, score


def load_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [r["case_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate case_id in case list")
    return rows


def execute(jobs, judge: Judge, workers: int, log=print) -> dict:
    """jobs: list of (key, Request). Returns key -> JudgeResult. Identical requests are sent once (judge cache)."""
    results, lock, started = {}, threading.Lock(), time.time()

    def one(job):
        key, req = job
        res = judge.ask(req.system, req.parts, req.validator, role=req.role)
        with lock:
            results[key] = res
            if len(results) % 50 == 0 or len(results) == len(jobs):
                stat = {}
                for r in results.values():
                    stat[r.status] = stat.get(r.status, 0) + 1
                log(f"{len(results)}/{len(jobs)} {stat} {time.time() - started:.0f}s", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, jobs))
    return results


def run(rows: list[dict], work: Path, judge: Judge, workers: int, log=print) -> list[dict]:
    plans, errors = {}, {}
    for row in rows:
        if row.get("missing"):
            errors[row["case_id"]] = "missing: " + row["missing"]
            continue
        try:
            plan = plan_case(row, work / "crops" / row["case_id"])
            if row.get("only_views"):
                plan.requests = [r for r in plan.requests if r.view in set(row["only_views"])]
            plans[row["case_id"]] = plan
        except Exception as exc:  # noqa: BLE001 - recorded per case
            errors[row["case_id"]] = f"{type(exc).__name__}: {exc}"
    jobs = [((cid, req.view), req) for cid, plan in plans.items() for req in plan.requests]
    log(f"cases={len(rows)} planned={len(plans)} case_errors={len(errors)} requests={len(jobs)}", flush=True)
    results = execute(jobs, judge, workers, log)

    out = []
    for row in rows:
        cid = row["case_id"]
        base = {"case_id": cid, "meta": row.get("meta", {}), "output_mode": row["output_mode"]}
        if cid in errors:
            status = "missing" if errors[cid].startswith("missing: ") else "case_error"
            out.append({**base, "status": status, "error": errors[cid], "metrics": {m: None for m in METRICS}})
            continue
        plan = plans[cid]
        facts = dict(plan.deterministic)
        views = {v: {"status": "deterministic"} for v in plan.deterministic}
        for req in plan.requests:
            res = results[(cid, req.view)]
            facts[req.view] = res.value if res.status == "ok" else None
            views[req.view] = {"status": res.status, "request_sha256": res.request_sha256,
                               "resolved_model": res.resolved_model, "attempts": res.attempts, "error": res.error}
        scored = score(facts, row["truth"], output_mode=row["output_mode"],
                       structure_pixel_equal=plan.audit["structure_pixel_equal"])
        out.append({**base, "status": "scored", **scored, "facts": facts, "views": views, "audit": plan.audit})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--cache", type=Path, help="judge record cache (default WORK/cache)")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--judge-model", default=JudgeConfig.model)
    ap.add_argument("--base-url", default=JudgeConfig.base_url)
    ap.add_argument("--api-key-env", default=JudgeConfig.api_key_env)
    ap.add_argument("--reasoning-effort", default=JudgeConfig.reasoning_effort)
    ap.add_argument("--no-stream", action="store_true")
    a = ap.parse_args()
    config = JudgeConfig(model=a.judge_model, base_url=a.base_url, api_key_env=a.api_key_env,
                         reasoning_effort=a.reasoning_effort, stream=not a.no_stream)
    judge = Judge(config, a.cache or a.work / "cache")
    rows = load_rows(a.cases)
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    scored = run(rows, a.work, judge, a.workers)
    with (a.work / "scored.jsonl").open("w", encoding="utf-8") as f:
        for r in scored:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    status = {}
    for r in scored:
        for v in r.get("views", {}).values():
            status[v["status"]] = status.get(v["status"], 0) + 1
    (a.work / "run.json").write_text(json.dumps({
        "started": started, "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cases": str(a.cases.resolve()), "n_cases": len(rows), "judge": asdict(config), "view_status": status,
        "case_errors": sum(r["status"] == "case_error" for r in scored)}, indent=1))
    print(json.dumps(status), flush=True)


if __name__ == "__main__":
    main()
