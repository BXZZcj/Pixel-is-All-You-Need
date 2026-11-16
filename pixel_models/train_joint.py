"""One model, several tasks: the cross-modal scaling ladder of Sec. 5 and the Sum Maze runs of Sec. 6.

Text tasks (text_tasks), drawing tasks (draw_tasks) and mazes (maze_tasks) share one 192 px canvas, one flow ViT and one
training loop (lr 1e-4 after a 1000-step warm-up, EMA 0.999, canvas t -> canvas t + 1). Every step every task gets
exactly --per-task examples (batch = per-task x tasks). Drawing tasks and mazes weigh the pixels a step changes
(--chg-weight); text tasks do not.

  train     RUN/evals.jsonl: the text sets' losses and accuracy (as train_text), and draw: {TASK/SET: exact, iou, ...}
            for the drawing and maze sets (mazes: free length, a rollout stops when an answer is written)
  valloss   the per-step metrics on fixed held-out transitions -> RUN/valloss{--tag}.json: for drawing sets
            draw_step_ok, the share of transitions whose generated next frame changes pixels with an IoU >= 0.8 against
            the true change (the per-step error of Sec. 5 on Curve Tracing is 1 - draw_step_ok)

usage: torchrun --standalone --nproc_per_node 4 -m pixel_models.train_joint train --tasks maze:5+trace:6+countpaint:tallysel:hard \
           --per-task 88 --steps 132000 --decay-from 128000 --cf-paint 0 --size L --out DIR
       python -m pixel_models.train_joint valloss --run DIR --val-per-set 4096 --tag _4k
Checkpoints (model, EMA, optimizer, step) every --save-every steps on rank 0; a restarted run resumes on any GPU count.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
import zlib
from pathlib import Path

import numpy as np

from . import draw_tasks as D
from . import model as P
from . import canvas as Q
from . import train_text as S
from . import text_tasks as T
from . import maze_tasks as W

WORLD, DRAW = ("maze",), ("trace", "tracesum", "countpaint")


def canvas(geo: Q.Geo, m: np.ndarray) -> np.ndarray:
    x = Q.glyphs()[m].transpose(0, 2, 1, 3).reshape(geo.px, geo.px).copy()
    x[geo.line_y, 4:geo.px - 4] = 0
    return x


class TextFam:
    fam = "text"

    def __init__(self, task: T.Task, px: int):
        self.t, self.spec, self.geo, self.px, self.tag = task, task.spec, Q.Geo(px), px, task.spec.replace(":", "")

    def pair(self, rng):
        e = self.t.sample(rng)
        p_ = T.pair(e, "video", rng, 1)
        return canvas(self.geo, T.frame_codes(self.geo, p_[0], p_[1])), canvas(self.geo, T.frame_codes(self.geo, p_[2], p_[3]))


class DrawFam:
    fam = "draw"

    def __init__(self, spec: str, px: int, cf: float):
        self.t, self.spec, self.geo, self.cf, self.tag = D.Task(spec, px), spec, Q.Geo(px), cf, spec.replace(":", "")
        self.reuse, self._sc, self._left = getattr(self.t, "reuse", 1), None, 0   # slow scenes: several transitions each

    def pair(self, rng):
        if self.reuse > 1:
            if self._left <= 0:
                self._sc, self._left = self.t.train(rng), self.reuse
            self._left -= 1
            return D.pair(self._sc, self.geo, "video", rng, self.cf)
        return D.pair(self.t.train(rng), self.geo, "video", rng, self.cf)

    def pair_of(self, sc, rng):
        return D.pair(sc, self.geo, "video", rng, 0.0)

    def render(self, sc, f):
        return D.render(sc, self.geo, f)

    def n_frames(self, sc):
        return D.n_frames(sc)

    def score(self, sc, fin, got, note):
        geo = self.geo
        if self.t.name == "countpaint":
            ious = D.paint_ious(sc, geo, D.render(sc, geo, 0), fin)
            want, iou = [st[2] for st in sc.steps if st[0] in ("note", "tally")][-1], ious[sc.chosen]
            return {"exact": float(got == sc.answer), "iou": float(iou), "iou80": float(iou >= 0.8),
                    "pick": float(int(np.argmax(ious)) == sc.chosen and max(ious) >= 0.5), "counts": float(note == want),
                    "solved": float(got == sc.answer and iou >= 0.8 and note == want)}
        y0, y1 = Q.CH, geo.q_rows * Q.CH
        want, have = D.render(sc, geo, len(sc.steps) + 1)[y0:y1] < 60, fin[y0:y1] < 60
        return {"exact": float(got == sc.answer), "trace_iou": float((want & have).sum() / max(1, (want | have).sum()))}


class WorldFam:
    """frame by frame: one move per frame (free length at test time); direct (--world-arm direct): the problem canvas ->
    its last frame (the whole path drawn and the moves written) in one generation"""
    fam = "world"

    def __init__(self, spec: str, px: int, arm: str = "video"):
        self.t, self.spec, self.geo, self.tag = W.Task(spec, px), spec, Q.Geo(px), spec.replace(":", "")
        self.arm = arm

    def pair(self, rng):
        return self.pair_of(self.t.train(rng), rng)

    def pair_of(self, sc, rng):
        if self.arm == "direct":
            return W.render(sc, self.geo, 0), W.render(sc, self.geo, W.n_frames(sc))
        return W.pair(sc, self.geo, rng)

    def render(self, sc, f):
        return W.render(sc, self.geo, f)

    def n_frames(self, sc):
        return 1 if self.arm == "direct" else W.n_frames(sc)

    def score(self, sc, fin, got, note):
        return W.score_maze(sc, self.geo, fin, got)


def build(spec: str, px: int, seed: int, kb_people: int, cf: float, world_arm: str = "video"):
    """-> the task families in the order of spec, and the text tasks as one T.MultiTask (None if there are none)"""
    parts = spec.split("+")
    is_text = lambda s: s.split(":")[0] not in WORLD + DRAW
    texts = [s for s in parts if is_text(s)]
    multi = T.MultiTask("multi:" + "+".join(texts), px, seed, kb_people) if texts else None
    by = dict(zip(texts, multi.tasks)) if texts else {}
    fams = [TextFam(by[s], px) if is_text(s) else WorldFam(s, px, world_arm) if s.split(":")[0] in WORLD else DrawFam(s, px, cf) for s in parts]
    return fams, multi


class ValText:
    """the text test sets under a fixed name: the same held-out pairs whichever other tasks an arm trains"""
    spec = "all"

    def __init__(self, multi):
        self.sets, self.fits = multi.sets, multi.fits


def draw_val_set(fams, per_set: int) -> dict:
    """fixed held-out transitions (input, target) of every drawing test set"""
    out = {}
    for f in fams:
        if f.fam == "text":
            continue
        for name, g in f.t.sets.items():
            rng = random.Random(f"all-val-{f.spec}-{name}")
            pr = [f.pair_of(g(rng), rng) for _ in range(per_set)]
            out[f"{f.tag}/{name}"] = {"x": np.stack([p[0] for p in pr]), "y": np.stack([p[1] for p in pr]),
                                      "seed": zlib.crc32(f"{f.spec}|{name}".encode())}
    return out


def draw_step_metrics(net, dvs: dict, dev, steps=10, batch=64) -> dict:
    """one held-out transition: generate the next frame; IoU of the pixels it changes against the pixels the true step changes"""
    import torch
    net.eval()
    out = {"draw_step_iou": {}, "draw_step_ok": {}}
    ts = torch.linspace(0, 1, steps + 1, device=dev)
    with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
        for name, v in dvs.items():
            gen = torch.Generator(device=dev).manual_seed(v["seed"])
            ious = []
            for i in range(0, len(v["x"]), batch):
                xn, yn = v["x"][i:i + batch], v["y"][i:i + batch]
                c = (torch.from_numpy(xn).to(dev).float() / 127.5 - 1)[:, None]
                x = torch.randn(c.shape, device=dev, generator=gen)
                for k in range(steps):
                    xh = net(c, x, ts[k].repeat(len(c))).float()
                    x = x + (ts[k + 1] - ts[k]) * (xh - x) / (1 - ts[k]).clamp(min=0.05)
                fin = ((x.float().clamp(-1, 1) + 1) * 127.5)[:, 0].round().clamp(0, 255).byte().cpu().numpy()
                want, have = np.abs(yn.astype(int) - xn.astype(int)) > 40, np.abs(fin.astype(int) - xn.astype(int)) > 40
                inter, union = (want & have).sum((1, 2)), (want | have).sum((1, 2))
                ious += list(np.where(union > 0, inter / np.maximum(union, 1), 1.0))
            out["draw_step_iou"][name] = round(float(np.mean(ious)), 5)
            out["draw_step_ok"][name] = round(float(np.mean(np.array(ious) >= 0.8)), 5)
    return out


def eval_draw(ema, fams, dtests: dict, geo: Q.Geo, dev, steps=10, batch=64) -> dict:
    """roll every drawing test scene out for its true number of frames and score the last frame"""
    import torch
    ema.eval()
    out = {}
    to_f = lambda a: (torch.from_numpy(a).to(dev).float() / 127.5 - 1)[:, None]
    ts = torch.linspace(0, 1, steps + 1, device=dev)
    with torch.no_grad():
        for f in fams:
            if f.fam == "text":
                continue
            for name, ex in dtests[f.tag].items():
                rows = []
                for i in range(0, len(ex), batch):
                    chunk = ex[i:i + batch]
                    cur = to_f(np.stack([f.render(sc, 0) for sc in chunk]))
                    # maze: free length - the model walks until it writes an answer (up to W.MAX_FRAMES frames): its path
                    # may be shorter or longer than the demonstrated one; the other tasks: each scene's own frame count
                    free = f.fam == "world" and f.t.name == "maze" and f.arm == "video"
                    n_fr = [W.MAX_FRAMES if free else f.n_frames(sc) for sc in chunk]
                    final, done = cur.clone(), [False] * len(chunk)
                    g = torch.Generator(device=dev).manual_seed(7 + i)
                    for fr in range(max(n_fr)):
                        x = torch.randn(cur.shape, device=dev, generator=g)
                        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                            for k in range(steps):
                                xh = ema(cur, x, ts[k].repeat(len(chunk))).float()
                                x = x + (ts[k + 1] - ts[k]) * (xh - x) / (1 - ts[k]).clamp(min=0.05)
                        cur = x.float().clamp(-1, 1)
                        if free:
                            wrote = W.answer_written(geo, ((cur + 1) * 127.5)[:, 0].round().clamp(0, 255).byte().cpu().numpy())
                            for j, w in enumerate(wrote):
                                if w and not done[j]:
                                    final[j], done[j] = cur[j], True
                            if all(done):
                                break
                        else:
                            for j, n_ in enumerate(n_fr):
                                if n_ == fr + 1:
                                    final[j] = cur[j]
                    if free:
                        for j in range(len(chunk)):
                            if not done[j]:
                                final[j] = cur[j]
                    fin = ((final + 1) * 127.5)[:, 0].round().clamp(0, 255).byte().cpu().numpy()
                    got = [g_.strip() for g_ in Q.read(geo, fin)]
                    notes = D.read_row(geo, fin, geo.q_rows - 1)
                    rows += [f.score(sc, fi, g_, nt) for sc, fi, g_, nt in zip(chunk, fin, got, notes)]
                out[f"{f.tag}/{name}"] = {k: round(float(np.nanmean([r[k] for r in rows])), 4) for k in rows[0]}
    return out


def main_train(a) -> None:
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader, IterableDataset
    world = int(os.environ.get("WORLD_SIZE", "1"))
    ddp = world > 1
    cuda = torch.cuda.is_available()
    if ddp:
        import datetime
        dist.init_process_group(backend="nccl" if cuda else "gloo", timeout=datetime.timedelta(hours=3))
    rank = dist.get_rank() if ddp else 0
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if cuda:
        torch.cuda.set_device(local)
    dev = "cuda" if cuda else "cpu"
    assert a.per_task % world == 0, "--per-task must be divisible by the number of processes"
    per = a.per_task // world
    torch.manual_seed(a.seed)
    P.S = a.px
    P.CKPT_ACT = a.ckpt_act
    geo = Q.Geo(a.px)
    Q.glyphs()   # cached before the data workers fork
    fams, multi = build(a.tasks, a.px, a.seed, a.kb_people, a.cf_paint, a.world_arm)
    total = a.per_task * len(fams)
    model = P.build(a.size).to(dev)
    ema = P.build(a.size).to(dev) if rank == 0 else None
    if ema is not None:
        ema.load_state_dict(model.state_dict()); ema.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=a.wd)
    step, ck = 0, a.out / "ckpt.pt"
    a.out.mkdir(parents=True, exist_ok=True)
    if ck.exists():
        s = torch.load(ck, map_location="cpu")
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); step = s["step"]
        if ema is not None:
            ema.load_state_dict(s["ema"])
        if rank == 0:
            print(f"resumed at step {step}", flush=True)
    n_params = sum(p.numel() for p in model.parameters())
    n_nonembed = S.nonembed_params(model)
    fpe = S.flops_per_example(n_nonembed, a.px)
    net = DDP(model, device_ids=[local] if cuda else None) if ddp else model
    G = torch.from_numpy(Q.glyphs()).to(dev)
    images = S.make_images(G, geo, dev)
    images.px = a.px
    tests, vs, dtests = {}, {}, {}
    if rank == 0:
        if multi:
            tests = multi.test_sets(a.n_test)
            vs = S.val_set(ValText(multi), geo, "video", a.px, a.val_per_set)
        dtests = {f.tag: f.t.test_sets(a.n_test) for f in fams if f.fam != "text"}
        (a.out / "config.json").write_text(json.dumps({"trainer": "train_joint", **{k: str(v) for k, v in vars(a).items()}, "world": world, "batch": total,
                                                       "size": a.size, "params": n_params, "nonembed_params": n_nonembed,
                                                       "flops_per_example": fpe, "tags": [f.tag for f in fams]}, indent=1))

    class Batches(IterableDataset):
        def __iter__(self):
            wi = torch.utils.data.get_worker_info()
            rng = random.Random(f"{a.seed}-{rank}-{step}-{wi.id if wi else 0}")
            while True:
                xs, ys, ids = [], [], []
                for i, f in enumerate(fams):
                    for _ in range(per):
                        x, y = f.pair(rng)
                        xs.append(x); ys.append(y); ids.append(i)
                yield torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys)), torch.tensor(ids)

    loader = iter(DataLoader(Batches(), batch_size=None, num_workers=a.workers, persistent_workers=a.workers > 0,
                             prefetch_factor=4 if a.workers else None))
    to_f = lambda t: (t.to(dev, non_blocking=True).float() / 127.5 - 1)[:, None]
    cw = torch.tensor([a.chg_weight if f.fam != "text" else 0.0 for f in fams], device=dev)

    def evaluate(loss_now):
        res = {"step": step, "examples": step * total, "flops": step * total * fpe, "train_loss": round(float(loss_now.detach()), 6),
               "wall_hours": round((time.time() - t_start) / 3600, 3)}
        if vs:
            res.update(S.val_loss(ema, vs, images, dev))
        if step % a.acc_every == 0 or step == a.steps:
            if tests:
                res["exact"] = S.accuracy(ema, tests, images, geo, "video", dev, a.sample_steps, a.eval_batch)
            if dtests:
                res["draw"] = eval_draw(ema, fams, dtests, geo, dev, a.draw_sample_steps, a.eval_batch)
        ema.train()
        with (a.out / "evals.jsonl").open("a") as f_:
            f_.write(json.dumps(res) + "\n")
        print(json.dumps(res), flush=True)

    net.train(); t0 = t_start = time.time(); s0 = step; loss = torch.tensor(0.0)
    while step < a.steps:
        x1, y, tid = next(loader)
        x1, y, tid = to_f(x1), to_f(y), tid.to(dev)
        lr = a.lr * min(1.0, (step + 1) / a.warmup)
        if a.decay_from and step >= a.decay_from:
            lr *= max(0.0, (a.steps - step) / max(1, a.steps - a.decay_from))
        for g_ in opt.param_groups:
            g_["lr"] = lr
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=cuda):
            w = 1 + cw[tid][:, None, None, None] * F.max_pool2d(((y - x1).abs() > 0.1).float(), 5, stride=1, padding=2)
            if a.ctx_noise:
                x1 = x1 + a.ctx_noise * torch.randn_like(x1)
            t = torch.sigmoid(torch.randn(len(x1), device=dev))
            e = torch.randn_like(y)
            xt = t[:, None, None, None] * y + (1 - t[:, None, None, None]) * e
            xh = net(x1, xt, t).float()
            lmap = ((xh - xt) / (1 - t).clamp(min=0.05)[:, None, None, None] - (y - e)) ** 2
            per_ex = (lmap * w).sum((1, 2, 3)) / w.sum((1, 2, 3))   # every example the same total weight
            loss = per_ex.mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        ok = torch.isfinite(gnorm)
        if ddp:
            okt = torch.tensor([1.0 if ok else 0.0], device=dev); dist.all_reduce(okt, op=dist.ReduceOp.MIN); ok = bool(okt.item() > 0)
        if ok:
            opt.step()
            if ema is not None:
                with torch.no_grad():
                    for pe, pm in zip(ema.parameters(), model.parameters()):
                        pe.mul_(a.ema).add_(pm.detach(), alpha=1 - a.ema)
        step += 1
        if rank == 0 and step % 200 == 0:
            by = {f.tag: round(float(per_ex.detach()[tid == i].float().mean()), 6) for i, f in enumerate(fams)}
            print(json.dumps({"step": step, "loss": round(float(loss), 6), "lr": lr, "sec_per_step": round((time.time() - t0) / (step - s0), 4), "by_task": by}), flush=True)
        if rank == 0 and a.snap_every and step % a.snap_every == 0:
            (a.out / "snaps").mkdir(exist_ok=True)
            torch.save({k: v.to(torch.bfloat16) for k, v in ema.state_dict().items()}, a.out / "snaps" / f"ema_{step:06d}.pt")
        if rank == 0 and (step % a.save_every == 0 or step == a.steps):
            if ck.exists() and step % a.keep_every == 0:
                os.replace(ck, a.out / "ckpt_prev.pt")
            torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "step": step}, str(ck) + ".tmp")
            os.replace(str(ck) + ".tmp", ck)
        if step % a.eval_every == 0 or step == a.steps:
            if rank == 0:
                evaluate(loss)
            if ddp:
                dist.barrier()
    if rank == 0:
        (a.out / "TRAIN_DONE").write_text(str(step))
    if ddp:
        dist.destroy_process_group()


def main_valloss(a) -> None:
    """the fixed-set step metrics of a finished run -> RUN/valloss.json; with --snaps, of every snapshot -> RUN/valloss_curve.jsonl"""
    import torch
    cfg = json.loads((a.run / "config.json").read_text())
    px, size = int(cfg["px"]), cfg["size"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    P.S = px
    geo = Q.Geo(px)
    fams, multi = build(cfg["tasks"], px, int(cfg.get("seed", 0)), int(cfg.get("kb_people", 300)), 0.0, cfg.get("world_arm", "video"))
    ema = P.build(size).to(dev)
    G = torch.from_numpy(Q.glyphs()).to(dev)
    images = S.make_images(G, geo, dev); images.px = px
    vs = S.val_set(ValText(multi), geo, "video", px, a.val_per_set) if multi else {}
    dvs = draw_val_set(fams, a.val_per_set)
    n_nonembed, total = S.nonembed_params(ema), int(cfg["batch"])

    def one(state, step):
        ema.load_state_dict({k: v.float() for k, v in state.items()})
        r = {"run": str(a.run), "step": step, "size": size, "params": sum(p.numel() for p in ema.parameters()), "nonembed_params": n_nonembed,
             "batch": total, "flops": (step or 0) * total * S.flops_per_example(n_nonembed, px), "val_per_set": a.val_per_set}
        if vs:
            r.update(S.val_loss(ema, vs, images, dev, a.batch)); r.update(S.step_metrics(ema, vs, images, geo, G, dev, a.sample_steps, a.batch))
        if dvs:
            r.update(draw_step_metrics(ema, dvs, dev, a.sample_steps, a.batch))
        return r
    if not a.snaps_only:
        s = torch.load(a.run / "ckpt.pt", map_location="cpu")
        res = one(s["ema"], s.get("step"))
        (a.run / f"valloss{a.tag}.json").write_text(json.dumps(res, indent=1))
        print(json.dumps(res), flush=True)
    if (a.snaps or a.snaps_only) and (a.run / "snaps").is_dir():
        with (a.run / "valloss_curve.jsonl").open("w") as f:
            for snap in sorted((a.run / "snaps").glob("ema_*.pt")):
                r = one(torch.load(snap, map_location="cpu"), int(snap.stem.split("_")[1]))
                f.write(json.dumps(r) + "\n"); f.flush()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--tasks", required=True, help="A+B+...: text, drawing and maze tasks")
    t.add_argument("--size", choices=sorted(P.SIZES), default="S")
    t.add_argument("--px", type=int, default=192)
    t.add_argument("--per-task", type=int, default=32, help="examples of every task in every step (batch = this x tasks)")
    t.add_argument("--steps", type=int, default=44000)
    t.add_argument("--decay-from", type=int, default=40000)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--wd", type=float, default=0.0)
    t.add_argument("--warmup", type=int, default=1000)
    t.add_argument("--ema", type=float, default=0.999)
    t.add_argument("--ctx-noise", type=float, default=0.03)
    t.add_argument("--chg-weight", type=float, default=30.0, help="drawing tasks: weight of the pixels a step changes")
    t.add_argument("--cf-paint", type=float, default=0.5, help="countpaint: share of counterfactual letters after the letter frame")
    t.add_argument("--world-arm", choices=("video", "direct"), default="video", help="mazes: frame by frame, or direct answering")
    t.add_argument("--kb-people", type=int, default=300)
    t.add_argument("--sample-steps", type=int, default=30)
    t.add_argument("--draw-sample-steps", type=int, default=10)
    t.add_argument("--n-test", type=int, default=128)
    t.add_argument("--val-per-set", type=int, default=256)
    t.add_argument("--eval-batch", type=int, default=64)
    t.add_argument("--eval-every", type=int, default=2000)
    t.add_argument("--acc-every", type=int, default=4000)
    t.add_argument("--save-every", type=int, default=1000)
    t.add_argument("--keep-every", type=int, default=5000)
    t.add_argument("--snap-every", type=int, default=4000)
    t.add_argument("--ckpt-act", action="store_true")
    t.add_argument("--workers", type=int, default=16)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--out", type=Path, required=True)
    v = sub.add_parser("valloss")
    v.add_argument("--run", type=Path, required=True)
    v.add_argument("--val-per-set", type=int, default=1024, help="held-out transitions per test set (Sec. 5: 4096)")
    v.add_argument("--batch", type=int, default=64)
    v.add_argument("--sample-steps", type=int, default=10)
    v.add_argument("--snaps", action="store_true")
    v.add_argument("--snaps-only", action="store_true")
    v.add_argument("--tag", default="", help="write RUN/valloss{TAG}.json (scaling_fit reads valloss_4k.json)")
    a = ap.parse_args()
    {"train": main_train, "valloss": main_valloss}[a.cmd](a)


if __name__ == "__main__":
    main()
