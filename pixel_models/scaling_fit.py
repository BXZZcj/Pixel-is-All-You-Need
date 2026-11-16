"""Power-law fits of the per-step error (Sec. 5.2, Tab. 7).

For each size: the median per-step error over the seeds; a least-squares line of log error against log non-embedding
parameters N (error = A * N^-alpha) on 7.6M-130M (XS, S, M, B); the extrapolation to the held-out 458M model (L), with
relative error = measured / predicted - 1. 90% intervals from a seed bootstrap (4,000 draws): resampling the seeds of
every fitted size gives the interval of the prediction, resampling the 458M seeds the interval of its measured median,
and resampling all sizes together the interval of the relative error. The last line fits all five sizes.

Per-step error (Appendix G, J), from RUN/valloss*.json written by `train_text valloss` / `train_joint valloss`:
  text    1 - step_exact, the mean of expr4/k4, mul4x4/hard4x4 and Factual Questions (the mean of its four sets of new
          entities), 4,096 transitions per test set
  trace   1 - draw_step_ok (changed-pixel IoU >= 0.8) of Curve Tracing in the joint cross-modal model, averaged over
          its sets with 2, 4, 6 and 8 curves, 4,096 transitions per test set

usage: python -m pixel_models.scaling_fit --ladder text  --runs RUNS... [--out fit.json]
       python -m pixel_models.scaling_fit --ladder trace --runs RUNS... [--out fit.json]
       (each run directory holds config.json and the valloss file; size and seed are read from config.json)
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np

SIZES = ("XS", "S", "M", "B", "L")
FIT_SIZES = SIZES[:4]
HOP_SETS = ("new_country", "new_mcity", "new_mjob", "new_mcountry")


def text_err(v: dict) -> float:
    x = v["step_exact"]
    hop = sum(1 - x["hop3/" + k] for k in HOP_SETS) / len(HOP_SETS)
    return ((1 - x["expr4/k4"]) + (1 - x["mul4x4/hard4x4"]) + hop) / 3


def trace_err(v: dict) -> float:
    ks = [k for k in v["draw_step_ok"] if k.startswith("trace6")]
    return sum(1 - v["draw_step_ok"][k] for k in ks) / len(ks)


LADDERS = {"text": (text_err, "valloss.json"), "trace": (trace_err, "valloss_4k.json")}


def fit(n, err):
    """-> A, alpha, log RMSE of err = A * n^-alpha"""
    x, y = np.log(np.asarray(n, float)), np.log(np.asarray(err, float))
    s, c = np.polyfit(x, y, 1)
    return math.exp(c), -s, float(np.sqrt(np.mean((c + s * x - y) ** 2)))


def analyse(seeds: dict, N: dict, draws: int = 4000) -> dict:
    """seeds: {size: [per-step error of each seed]}, N: {size: non-embedding parameters}"""
    med = {s: float(np.median(seeds[s])) for s in SIZES if seeds.get(s)}
    A, a, rmse = fit([N[s] for s in FIT_SIZES], [med[s] for s in FIT_SIZES])
    pred, obs = A * N["L"] ** -a, med["L"]
    rng = random.Random(0)
    boot = lambda s: float(np.median([rng.choice(seeds[s]) for _ in seeds[s]]))
    bp, bo, br = [], [], []
    for _ in range(draws):
        A_, a_, _ = fit([N[s] for s in FIT_SIZES], [boot(s) for s in FIT_SIZES])
        bp.append(A_ * N["L"] ** -a_)
    for _ in range(draws):
        bo.append(boot("L"))
    for _ in range(draws):
        A_, a_, _ = fit([N[s] for s in FIT_SIZES], [boot(s) for s in FIT_SIZES])
        br.append(boot("L") / (A_ * N["L"] ** -a_) - 1)
    A2, a2, rmse2 = fit([N[s] for s in med], [med[s] for s in med])
    ci = lambda v: [float(x) for x in np.percentile(v, [5, 95])]
    return {"median": med, "N": N, "fit_XS_B": {"A": A, "alpha": a, "log_rmse": rmse},
            "L_predicted": pred, "L_predicted_ci90": ci(bp), "L_measured": obs, "L_measured_ci90": ci(bo),
            "relative_error": obs / pred - 1, "relative_error_ci90": ci(br),
            "fit_all": {"A": A2, "alpha": a2, "log_rmse": rmse2}}


def collect(runs: list[Path], ladder: str, seeds_kept=range(5)) -> tuple[dict, dict]:
    f, name = LADDERS[ladder]
    seeds, N = {s: {} for s in SIZES}, {}
    for run in runs:
        cfg = json.loads((run / "config.json").read_text())
        v = json.loads((run / name).read_text())
        size, seed = cfg["size"], int(cfg.get("seed", 0))
        if seed in seeds_kept:
            seeds[size][seed] = f(v)
            N[size] = int(v["nonembed_params"])
    return {s: [d[k] for k in sorted(d)] for s, d in seeds.items()}, N


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ladder", choices=sorted(LADDERS), required=True)
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    seeds, N = collect(a.runs, a.ladder)
    r = analyse(seeds, N)
    for s in SIZES:
        print(f"{s:2s} N={N[s] / 1e6:6.1f}M  seeds " + " ".join(f"{x:.4f}" for x in seeds[s]) + f"  median {r['median'][s]:.4f}")
    f, lo, hi = r["fit_XS_B"], *r["L_predicted_ci90"]
    print(f"fit 7.6M-130M: {f['A']:.3g} * N^-{f['alpha']:.3f} (log RMSE {f['log_rmse']:.3f})")
    print(f"458M predicted {r['L_predicted']:.4f} ({lo:.4f}-{hi:.4f}), measured {r['L_measured']:.4f} "
          f"({r['L_measured_ci90'][0]:.4f}-{r['L_measured_ci90'][1]:.4f}), relative error {100 * r['relative_error']:+.0f}% "
          f"({100 * r['relative_error_ci90'][0]:+.0f}% to {100 * r['relative_error_ci90'][1]:+.0f}%)")
    g = r["fit_all"]
    print(f"fit on all five sizes: {g['A']:.3g} * N^-{g['alpha']:.3f} (log RMSE {g['log_rmse']:.3f})")
    if a.out:
        a.out.write_text(json.dumps({"ladder": a.ladder, "seeds": seeds, **r}, indent=1))


if __name__ == "__main__":
    main()
