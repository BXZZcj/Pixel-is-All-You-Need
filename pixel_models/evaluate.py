"""The evaluation of the pixel models (Appendix G): accuracy on 1,024 problems per test set.

Items: each task's own deterministic test sequence (the first 128 items are those its training run evaluated on), so
every regime and every run of a task sees the same problems:
  text tasks       Task.test_sets(n): Random(f"think-{spec}-{set}"), drawn in order (hop:K: the knowledge base is the
                   run's --seed)
  drawing tasks    item i of set k: Random(f"draw-{spec}-{k}-{i}")
  mazes            item i of set k: Random(f"world-{spec}-{k}-{i}")
  --items SPEC draws the items of another spec of the same problems: a direct countpaint run evaluated with
  --items countpaint:tallysel sees the problems of the frame-by-frame runs.
Protocols (chosen from the run's config.json; the trainers' own evaluation loops, run on more items):
  fixed      text tasks: deterministic sampler (the run's --sample-steps, 30); frame by frame rolled out for the true
             number of frames, direct one frame; chunks of --eval-batch with noise seed 7 + i
  free       text tasks trained with --per-frame > 1 and rl_text runs: free-running until the model writes an answer
             or --max-frames ("free", the reported number), and after the true number of frames ("oracle")
  draw       draw_tasks runs: rolled out for each scene's true number of frames, noise seed 7 per chunk, --sample-steps
             (10); Count Then Paint: solved = letter right, counts right and painting IoU >= 0.8; trace: trace_iou
  draw_free  free-length runs (rl_draw, :detour / :clean starts, single-task maze runs of train_joint): until the answer
             is written; minimal = solved in the fewest frames; mazes: valid = a shortest path with its own total
  joint      train_joint runs with several tasks: its drawing and maze families
Weights: the EMA of RUN/ckpt.pt, or --snap STEP: RUN/snaps/ema_STEP.pt.
Output: printed, and written to --out (a .json path, or a directory: DIR/final.json) and --collect DIR/RUN[...].json.

usage: python -m pixel_models.evaluate --run RUN [--n 1024] [--items SPEC] [--protocol P] [--snap STEP] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import socket
import time
from pathlib import Path

import numpy as np

from . import canvas as Q
from . import model as P

FIRST = 128   # the training runs' --n-test: reported separately ("first128") as a reproduction check


def kind_of(cfg: dict) -> str:
    """which trainer wrote RUN/config.json"""
    if "trainer" in cfg:
        return cfg["trainer"]
    if "tasks" in cfg:   # configs written before the "trainer" key: told apart by their argument names
        return "train_joint"
    if "algo" in cfg:
        return "rl_draw" if "w_letter" in cfg else "rl_text"
    if "mode" in cfg:
        return "text_tasks"
    if "nonembed_params" in cfg:
        return "train_text"
    if "cf_paint" in cfg or "workers" in cfg:
        return "draw_tasks"
    raise SystemExit("evaluate: cannot tell which trainer wrote this config.json")


def load_ema(run: Path, snap: int | None, size: str, dev: str):
    import torch
    net = P.build(size).to(dev)
    if snap is not None:
        path = run / "snaps" / f"ema_{snap:06d}.pt"
        state, step = torch.load(path, map_location="cpu"), snap
    else:
        path = run / "ckpt.pt"
        s = torch.load(path, map_location="cpu")
        state = s["ema"] if "ema" in s else s
        step = (s.get("step", s.get("iter")) if "ema" in s else None)
    net.load_state_dict({k: v.float() for k, v in state.items()})
    net.eval()
    net.requires_grad_(False)
    return net, step, str(path)


def subset(tests: dict, sets: list[str]) -> dict:
    if not sets:
        return tests
    missing = [s for s in sets if s not in tests]
    if missing:
        raise SystemExit(f"evaluate: no test set {missing} (have {sorted(tests)})")
    return {k: v for k, v in tests.items() if k in sets}


# ------------------------------------------------------------------------------------------------ the protocols
def run_fixed(ema, task, geo, arm, dev, steps, batch, n, sets) -> dict:
    """train_text.accuracy (the text_tasks protocol) on the first n items of every test set"""
    import torch
    from . import train_text as S
    tests = subset(task.test_sets(n), sets)
    images = S.make_images(torch.from_numpy(Q.glyphs()).to(dev), geo, dev)
    acc = S.accuracy(ema, tests, images, geo, arm, dev, steps, batch)
    out = {k: {"n": len(tests[k]), "exact": acc[k]} for k in tests}
    if n > FIRST and FIRST % batch == 0:   # the training evaluation's items, chunks and noise: a reproduction check
        head = S.accuracy(ema, {k: v[:FIRST] for k, v in tests.items()}, images, geo, arm, dev, steps, batch)
        for k in out:
            out[k]["first128"] = {"exact": head[k]}
    return out


def run_free(ema, task, geo, dev, steps, batch, n, sets, max_frames, own_frames) -> dict:
    """rl_text's evaluate(): the true number of frames ("oracle") and free-running ("free")"""
    import torch
    from . import train_text as S
    from . import text_tasks as T
    tests = subset(task.test_sets(n), sets)
    images = S.make_images(torch.from_numpy(Q.glyphs()).to(dev), geo, dev)

    def first_canvases(exs):
        return images(np.stack([T.frame_codes(geo, e.rows[0], None) for e in exs]))

    def read(x):
        return [g.strip() for g in Q.read(geo, ((x.float().clamp(-1, 1) + 1) * 127.5)[:, 0].cpu().numpy())]

    out = {}
    tt = torch.linspace(0, 1, steps + 1, device=dev)
    with torch.no_grad():
        for name, ex in tests.items():
            oo, ff, fr_sum = [], [], 0
            for i in range(0, len(ex), batch):
                chunk = ex[i:i + batch]
                cur = first_canvases(chunk)
                Ts = [len(e.rows) for e in chunk]
                g_ = torch.Generator(device=dev).manual_seed(7 + i)
                o_ans, f_ans, f_n = [None] * len(chunk), [None] * len(chunk), [max_frames] * len(chunk)
                for fr in range(max(max_frames, max(Ts))):
                    x = torch.randn(cur.shape, device=dev, generator=g_)
                    with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                        for k in range(steps):
                            xh = ema(cur, x, tt[k].repeat(len(chunk))).float()
                            x = x + (tt[k + 1] - tt[k]) * (xh - x) / (1 - tt[k]).clamp(min=0.05)
                    cur = x.float().clamp(-1, 1)
                    for j, gj in enumerate(read(cur)):
                        if fr + 1 == Ts[j]:
                            o_ans[j] = gj
                        if f_ans[j] is None and gj and fr < max_frames:
                            f_ans[j], f_n[j] = gj, fr + 1
                    if all(v is not None for v in o_ans) and (all(v is not None for v in f_ans) or fr + 1 >= max_frames):
                        break
                oo += [o == e.answer for o, e in zip(o_ans, chunk)]
                ff += [f == e.answer for f, e in zip(f_ans, chunk)]
                fr_sum += sum(f_n)
            m = len(ex)
            r = {"n": m, "oracle": round(sum(oo) / m, 4), "free": round(sum(ff) / m, 4), "frames": round(fr_sum / m, 2)}
            r["exact"] = r["free"] if own_frames else r["oracle"]
            if m > FIRST and FIRST % batch == 0:
                r["first128"] = {"oracle": round(sum(oo[:FIRST]) / FIRST, 4), "free": round(sum(ff[:FIRST]) / FIRST, 4)}
            out[name] = r
    return out


def run_draw(ema, task, geo, arm, dev, steps, batch, n, sets) -> dict:
    """draw_tasks train_main's rollout() and evaluate()"""
    import torch
    from . import draw_tasks as D
    tests = subset(task.test_sets(n), sets)
    to_f = lambda t: (t.to(dev, non_blocking=True).float() / 127.5 - 1)[:, None]  # noqa: E731

    @torch.no_grad()
    def rollout(scs):
        cur = to_f(torch.from_numpy(np.stack([D.render(sc, geo, 0) for sc in scs])))
        n_fr = [D.n_frames(sc) if arm == "video" else 1 for sc in scs]
        g = torch.Generator(device=dev).manual_seed(7)
        ts = torch.linspace(0, 1, steps + 1, device=dev)
        final = cur.clone()
        for f in range(max(n_fr)):
            x = torch.randn(cur.shape, device=dev, generator=g)
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                for k in range(steps):
                    xh = ema(cur, x, ts[k].repeat(len(scs))).float()
                    x = x + (ts[k + 1] - ts[k]) * (xh - x) / (1 - ts[k]).clamp(min=0.05)
            cur = x.float().clamp(-1, 1)
            for j, nf_ in enumerate(n_fr):
                if nf_ == f + 1:
                    final[j] = cur[j]
        return ((final + 1) * 127.5)[:, 0].round().clamp(0, 255).byte().cpu().numpy()

    out = {}
    y0, y1 = Q.CH, geo.q_rows * Q.CH
    for name, ex in tests.items():
        oks, iou, cp = [], [], {"iou": [], "iou80": [], "pick": [], "counts": [], "solved": []}
        for i in range(0, len(ex), batch):
            chunk = ex[i:i + batch]
            fin = rollout(chunk)
            got = [g.strip() for g in Q.read(geo, fin)]
            oks += [g == sc.answer for g, sc in zip(got, chunk)]
            if task.name == "countpaint":
                notes = D.read_row(geo, fin, geo.q_rows - 1)
                for f, sc, g, nt in zip(fin, chunk, got, notes):
                    ious = D.paint_ious(sc, geo, D.render(sc, geo, 0), f)
                    cp["iou"].append(ious[sc.chosen]); cp["iou80"].append(ious[sc.chosen] >= 0.8)
                    cp["pick"].append(int(np.argmax(ious)) == sc.chosen and max(ious) >= 0.5)
                    want = [st[2] for st in sc.steps if st[0] in ("note", "tally")][-1]
                    cnt = nt == want if arm != "direct" else True
                    cp["counts"].append(nt == want)
                    cp["solved"].append(g == sc.answer and ious[sc.chosen] >= 0.8 and cnt)
            elif arm != "direct":
                for f, sc in zip(fin, chunk):
                    want = D.render(sc, geo, len(sc.steps) + 1)[y0:y1] < 60
                    have = f[y0:y1] < 60
                    iou.append((want & have).sum() / max(1, (want | have).sum()))
        r = {"n": len(ex), "exact": sum(oks) / len(ex)}
        if iou:
            r["trace_iou"] = float(np.mean(iou))
        if cp["iou"]:
            r.update({k: float(np.mean(v)) for k, v in cp.items() if not (k == "counts" and arm == "direct")})
        out[name] = {k: (v if k == "n" else round(float(v), 4)) for k, v in r.items()}
        if len(ex) > FIRST and FIRST % batch == 0:   # sequential chunks, seed 7 each: the training evaluation's first 128
            head = {"exact": round(float(np.mean(oks[:FIRST])), 4)}
            if cp["solved"]:
                head["solved"] = round(float(np.mean(cp["solved"][:FIRST])), 4)
            out[name]["first128"] = head
    return out


def run_draw_free(ema, task, geo, arm, dev, steps, batch, n, sets) -> dict:
    """free-length rollouts (rl_draw's evaluate()), scored by rl_draw.score"""
    import torch
    from . import rl_draw as DR
    from . import maze_tasks as W
    tests = subset(task.test_sets(n), sets)
    maze = isinstance(task, DR.WorldTask) and task.name == "maze"
    free_len = getattr(task, "free", False)   # :detour / :clean specs: free length, success in the fewest frames ("minimal")
    free = (maze and arm == "video") or free_len   # a direct maze run: one frame

    def frames_of(sc):
        if free:
            return W.MAX_FRAMES if maze else 32
        return DR.n_frames(task, sc) if arm == "video" else 1

    to_t = lambda arrs: torch.from_numpy(np.stack(arrs)).to(dev).float()[:, None] / 127.5 - 1  # noqa: E731
    to_u8 = lambda x: ((x.float().clamp(-1, 1) + 1) * 127.5)[:, 0].round().clamp(0, 255).byte().cpu().numpy()  # noqa: E731
    tt = torch.linspace(0, 1, steps + 1, device=dev)
    out = {}
    with torch.no_grad():
        for name, ex in tests.items():
            ms, by_nf = [], {}
            for sc in ex:
                by_nf.setdefault(frames_of(sc), []).append(sc)
            chunks = [g[i:i + batch] for g in by_nf.values() for i in range(0, len(g), batch)]
            for chunk in chunks:
                starts = [DR.render(task, sc, geo, 0) for sc in chunk]
                cur = to_t(starts)
                g_ = torch.Generator(device=dev).manual_seed(7)
                final, done = cur.clone(), torch.zeros(len(chunk), dtype=torch.bool, device=dev)
                length = torch.full((len(chunk),), frames_of(chunk[0]), dtype=torch.long, device=dev)
                for fr_ in range(frames_of(chunk[0])):
                    x = torch.randn(cur.shape, device=dev, generator=g_)
                    with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                        for k in range(steps):
                            xh = ema(cur, x, tt[k].repeat(len(chunk))).float()
                            x = x + (tt[k + 1] - tt[k]) * (xh - x) / (1 - tt[k]).clamp(min=0.05)
                    cur = x.float().clamp(-1, 1)
                    if free:
                        wrote = torch.tensor(W.answer_written(geo, to_u8(cur)), device=dev)
                        new = wrote & ~done
                        final[new] = cur[new]; length[new] = fr_ + 1; done |= wrote
                        if bool(done.all()):
                            break
                if free:
                    final[~done] = cur[~done]
                    cur = final
                mc = DR.score(task, geo, chunk, starts, to_u8(cur), arm)
                if free_len:
                    for m, sc, L in zip(mc, chunk, length.tolist()):
                        m["minimal"], m["length"] = float(bool(m["solved"]) and int(L) == sc.n_min), float(int(L) - sc.n_min)
                ms += mc
            r = {"exact": np.mean([m["letter"] for m in ms])}
            for k in ("iou", "pick", "counts", "count_frac", "solved", "minimal", "length", "valid", "legal", "extra", "path_iou"):
                if k in ms[0] and not (k == "counts" and arm == "direct"):
                    r[k] = float(np.nanmean([m[k] for m in ms]))
            out[name] = {"n": len(ex), **{k: round(float(v), 4) for k, v in r.items()}}
            if free and len(ex) > FIRST and FIRST % batch == 0:   # one frame count: chunks in item order, seed 7 each
                out[name]["first128"] = {k: round(float(np.nanmean([m[k] for m in ms[:FIRST]])), 4)
                                         for k in ("letter", "solved", "valid", "legal") if k in ms[0]}
    return out


def run_joint(ema, cfg, geo, dev, steps, batch, n, sets) -> dict:
    """train_joint.eval_draw on its drawing and maze families (the text tasks are not evaluated here)"""
    from . import train_joint as PA
    fams, _ = PA.build(cfg["tasks"], int(cfg["px"]), int(cfg["seed"]), int(cfg.get("kb_people", 300)), float(cfg.get("cf_paint", 0)))
    dt = {f.tag: f.t.test_sets(n) for f in fams if f.fam != "text"}
    res = PA.eval_draw(ema, fams, dt, geo, dev, steps, batch)
    res = subset(res, sets)
    return {k: {"n": n, **v} for k, v in res.items()}


# ------------------------------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="a run directory (config.json, ckpt.pt)")
    ap.add_argument("--snap", type=int, help="evaluate RUN/snaps/ema_STEP.pt instead of the final EMA")
    ap.add_argument("--n", type=int, default=1024, help="items per test set")
    ap.add_argument("--sets", default="", help="comma list of test sets (default: all)")
    ap.add_argument("--items", help="draw the items of this task spec (the same problems as another regime's runs)")
    ap.add_argument("--protocol", choices=("fixed", "free", "draw", "draw_free", "joint"))
    ap.add_argument("--batch", type=int, help="rollout chunk (default: the run's own; another value changes the noise)")
    ap.add_argument("--max-frames", type=int, help="free protocol: frame budget (default: the run's own, else 10)")
    ap.add_argument("--collect", type=Path, help="also write a copy to this directory, named after the run")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    import torch
    t0 = time.time()
    cfg = json.loads((a.run / "config.json").read_text())
    kind = kind_of(cfg)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    px, size, arm = int(cfg["px"]), cfg["size"], cfg.get("arm", "video")
    if kind == "train_joint":   # its maze regime (direct answering on Sum Mazes)
        arm = cfg.get("world_arm", "video")
    P.S = px
    geo = Q.Geo(px)
    spec = cfg.get("tasks") if kind == "train_joint" else cfg["task"]
    item_spec = a.items or spec
    sets = [s for s in a.sets.split(",") if s]
    per_frame = int(cfg.get("per_frame", 1))
    if a.protocol:
        proto = a.protocol
    elif kind == "train_text":
        proto = "fixed"
    elif kind == "text_tasks":
        proto = "free" if arm == "video" and per_frame > 1 else "fixed"
    elif kind == "rl_text":
        proto = "free"
    elif kind == "draw_tasks":   # the RL start models (:detour / :clean) as their RL runs see them: free length
        proto = "draw_free" if spec.endswith((":detour", ":clean")) else "draw"
    elif kind == "rl_draw":
        proto = "draw_free"
    else:   # train_joint: a single maze task is evaluated as rl_draw evaluates it
        proto = "draw_free" if "+" not in spec else "joint"
    if kind == "rl_text" or proto == "free":   # rl_text evaluates with 30 steps in chunks of 64
        steps, batch = 30, 64
    elif kind in ("train_joint", "draw_tasks") and proto == "draw_free":
        steps, batch = 10, 64   # rl_draw's evaluation (--sample-steps 10, --eval-batch 64)
    elif kind == "train_joint":
        steps, batch = int(cfg.get("draw_sample_steps", 10)), int(cfg["eval_batch"])
    else:
        steps, batch = int(cfg["sample_steps"]), int(cfg["eval_batch"])
    if kind == "text_tasks" and proto == "free":
        steps = int(cfg["sample_steps"])
    batch = a.batch or batch
    ema, step, wpath = load_ema(a.run, a.snap, size, dev)
    if proto in ("fixed", "free"):
        from . import text_tasks as T
        task = T.make_task(item_spec, px, int(cfg.get("seed", 0)), int(cfg.get("kb_people", 300)))
        if proto == "fixed":
            res = run_fixed(ema, task, geo, arm, dev, steps, batch, a.n, sets)
        else:
            mf = a.max_frames or int(cfg.get("max_frames", 10))
            res = run_free(ema, task, geo, dev, steps, batch, a.n, sets, mf, per_frame > 1 or kind == "rl_text")
    elif proto == "draw":
        from . import draw_tasks as D
        res = run_draw(ema, D.Task(item_spec, px), geo, arm, dev, steps, batch, a.n, sets)
    elif proto == "draw_free":
        from . import rl_draw as DR
        res = run_draw_free(ema, DR.make_task(item_spec, px), geo, arm, dev, steps, batch, a.n, sets)
    else:
        res = run_joint(ema, cfg, geo, dev, steps, batch, a.n, sets)
    out = {"run": str(a.run), "weights": wpath, "step": step, "trainer": kind, "protocol": proto, "task": spec,
           "items": item_spec, "arm": arm, "size": size, "n": a.n, "sample_steps": steps, "batch": batch,
           **({"max_frames": a.max_frames or int(cfg.get("max_frames", 10))} if proto == "free" else {}),
           "sets": res, "host": socket.gethostname(), "seconds": round(time.time() - t0, 1)}
    text = json.dumps(out, indent=1)
    print(text, flush=True)
    tag = a.run.name + (f"_s{a.snap}" if a.snap is not None else "") + (f"_items-{a.items.replace(':', '')}" if a.items else "")
    if a.protocol:
        tag += f"_{a.protocol}"
    if a.n != 1024:
        tag += f"_n{a.n}"
    if a.out is not None:
        dst = a.out if a.out.suffix == ".json" else a.out / "final.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(text)
    if a.collect:
        a.collect.mkdir(parents=True, exist_ok=True)
        (a.collect / f"{tag}.json").write_text(text)


if __name__ == "__main__":
    main()
