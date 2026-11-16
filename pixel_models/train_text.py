"""Trainer of the text tasks on 1-8 GPUs: the text scaling ladder of Sec. 5 and the 160k-step runs of Sec. 6.

The model and tasks are those of text_tasks (multi:A+B+C trains several tasks in one model, each example from a task
drawn uniformly), with the same recipe (batch 256, AdamW, lr 1e-4 after a 1000-step warm-up, EMA 0.999), trained data
parallel with torch.distributed so that one run can use 1-8 GPUs at the same global batch.

  train     RUN/evals.jsonl every --eval-every steps: the losses on a fixed held-out set of transitions, and every
            --acc-every steps the answer accuracy (deterministic sampler, true number of frames)
  valloss   the per-step metrics of a finished run on fixed held-out transitions -> RUN/valloss.json: step_exact is the
            share of transitions whose generated next frame reads back right in every cell (the per-step error of
            Sec. 5 is 1 - step_exact; scaling_fit.py fits it)

usage (one node, 8 GPUs):
  torchrun --standalone --nproc_per_node 8 -m pixel_models.train_text train --task multi:expr:4+mul:4x4+hop:3 --arm video --size L --out DIR
  python -m pixel_models.train_text valloss --run DIR --val-per-set 4096
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

from . import model as P
from . import canvas as Q
from . import text_tasks as T

VAL_PER_SET = 1024


def tokens_per_example(px: int) -> int:
    return (px // 8) ** 2


def flops_per_example(n_params: int, px: int) -> float:
    return 6.0 * n_params * tokens_per_example(px)


def nonembed_params(model) -> int:
    return sum(p.numel() for n, p in model.named_parameters() if n != "pos")


def val_set(task, geo, arm: str, px: int, per_set: int):
    """fixed held-out (input, target) frame pairs of every test set, with fixed noise seeds and times: numpy arrays"""
    out = {}
    for name, f in task.sets.items():
        rng = random.Random(f"scale-val-{task.spec}-{arm}-{name}")
        cond, tgt = [], []
        while len(cond) < per_set:
            e = f(rng)
            if not task.fits(e):
                continue
            p_ = T.pair(e, arm, rng)
            cond.append(T.frame_codes(geo, p_[0], p_[1])); tgt.append(T.frame_codes(geo, p_[2], p_[3]))
        g = np.random.default_rng(zlib.crc32(f"{task.spec}|{arm}|{name}".encode()))   # not hash(): it is salted per process
        out[name] = {"cond": np.stack(cond), "tgt": np.stack(tgt),
                     "t": 1 / (1 + np.exp(-g.standard_normal(per_set))).astype(np.float32),
                     "seed": int(g.integers(1 << 30))}
    return out


def make_images(G, geo, dev):
    import torch

    def images(m):
        x = G[torch.from_numpy(m).long().to(dev)].permute(0, 1, 3, 2, 4).reshape(len(m), 1, geo.px, geo.px)
        x[:, :, geo.line_y, 4:geo.px - 4] = 0
        return x.float() / 127.5 - 1
    return images


def val_loss(net, vs, images, dev, batch=64):
    """three losses on the fixed set (per test set and their mean "all"):
      val_loss  the training loss (v-loss at the fixed random times) over the whole canvas. Weighted by 1 / (1 - t)^2, it is
                dominated near t = 1 by tiny errors on pixels copied from the input: it hardly moves with size or training
      val_chg   the same loss on the pixels the step changes (|target - input| > 0.1, dilated by 2 px): the new writing
      val_x0    the one-step prediction from pure noise (t = 0, fixed noise) against the target, squared error on the
                changed pixels: the model's guess of the new content"""
    import torch
    import torch.nn.functional as F
    net.eval(); res = {"val_loss": {}, "val_chg": {}, "val_x0": {}}
    with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
        for name, v in vs.items():
            n, tot, chg, x0 = len(v["cond"]), 0.0, 0.0, 0.0
            gen = torch.Generator(device=dev).manual_seed(v["seed"])
            noise = torch.randn((n, 1, images.px, images.px), device=dev, generator=gen)
            for i in range(0, n, batch):
                c = images(v["cond"][i:i + batch])
                y = images(v["tgt"][i:i + batch])
                t = torch.from_numpy(v["t"][i:i + batch]).to(dev)
                e = noise[i:i + batch]
                xt = t[:, None, None, None] * y + (1 - t[:, None, None, None]) * e
                xh = net(c, xt, t).float()
                lmap = ((xh - xt) / (1 - t).clamp(min=0.05)[:, None, None, None] - (y - e)) ** 2
                m = F.max_pool2d(((y - c).abs() > 0.1).float(), 5, stride=1, padding=2)
                msum = m.sum((1, 2, 3)).clamp(min=1)
                tot += float(lmap.mean((1, 2, 3)).sum())
                chg += float(((lmap * m).sum((1, 2, 3)) / msum).sum())
                x0h = net(c, e, torch.zeros(len(c), device=dev)).float()
                x0 += float(((((x0h - y) ** 2) * m).sum((1, 2, 3)) / msum).sum())
            res["val_loss"][name] = round(tot / n, 6)
            res["val_chg"][name] = round(chg / n, 6)
            res["val_x0"][name] = round(x0 / n, 6)
    for k in res:
        res[k]["all"] = round(float(np.mean([res[k][s_] for s_ in vs])), 6)
    return res


def step_metrics(net, vs, images, geo, G, dev, steps=10, batch=64):
    """per held-out frame transition (the fixed set): generate the next frame (deterministic sampler, fixed noise), read
    every character cell back (nearest glyph) and compare with the target frame:
      step_exact  the whole frame right (every cell; the separator row excluded): the analogue of per-token accuracy
      cell_err    wrong cells among those the step should change;  copy_err  cells it should keep but changed"""
    import torch
    GF = G.float()
    net.eval()
    out = {"step_exact": {}, "cell_err": {}, "copy_err": {}}
    keep = np.ones(geo.rows, bool); keep[geo.q_rows] = False
    ts = torch.linspace(0, 1, steps + 1, device=dev)
    with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
        for name, v in vs.items():
            n = len(v["cond"]); ok = wrong = changed = spurious = kept = 0
            gen = torch.Generator(device=dev).manual_seed(v["seed"] + 1)
            for i in range(0, n, batch):
                c = images(v["cond"][i:i + batch])
                x = torch.randn(c.shape, device=dev, generator=gen)
                for k in range(steps):
                    xh = net(c, x, ts[k].repeat(len(c))).float()
                    x = x + (ts[k + 1] - ts[k]) * (xh - x) / (1 - ts[k]).clamp(min=0.05)
                img = ((x.float().clamp(-1, 1) + 1) * 127.5)[:, 0]
                cells = img.reshape(len(img), geo.rows, Q.CH, geo.cols, Q.CW).permute(0, 1, 3, 2, 4)
                code = torch.cat([((ch[:, :, :, None] - GF[None, None, None]) ** 2).mean((-1, -2)).argmin(-1)
                                  for ch in cells.split(8)]).cpu().numpy()[:, keep]
                tgt, cnd = v["tgt"][i:i + batch][:, keep], v["cond"][i:i + batch][:, keep]
                ok += int((code == tgt).all((1, 2)).sum())
                chg = tgt != cnd
                wrong += int(((code != tgt) & chg).sum()); changed += int(chg.sum())
                spurious += int(((code != tgt) & ~chg).sum()); kept += int((~chg).sum())
            out["step_exact"][name] = round(ok / n, 5)
            out["cell_err"][name] = round(wrong / max(1, changed), 6)
            out["copy_err"][name] = round(spurious / max(1, kept), 7)
    for k in out:
        if out[k]:
            out[k]["all"] = round(float(np.mean(list(out[k].values()))), 6)
    return out


def accuracy(net, tests, images, geo, arm, dev, steps=30, batch=64):
    """the text_tasks evaluation: deterministic sampler, frame by frame rolled out for the true number of frames"""
    import torch
    net.eval(); res = {}
    ts = torch.linspace(0, 1, steps + 1, device=dev)
    with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
        for name, ex in tests.items():
            ok = 0
            for i in range(0, len(ex), batch):
                chunk = ex[i:i + batch]
                cur = images(np.stack([T.frame_codes(geo, e.rows[0], None) for e in chunk]))
                Ts = [1 if arm != "video" else len(e.rows) for e in chunk]
                final = cur.clone()
                g = torch.Generator(device=dev).manual_seed(7 + i)
                for fr in range(max(Ts)):
                    x = torch.randn(cur.shape, device=dev, generator=g)
                    for k in range(steps):
                        xh = net(cur, x, ts[k].repeat(len(chunk))).float()
                        x = x + (ts[k + 1] - ts[k]) * (xh - x) / (1 - ts[k]).clamp(min=0.05)
                    cur = x.float().clamp(-1, 1)
                    for j, T_ in enumerate(Ts):
                        if T_ == fr + 1:
                            final[j] = cur[j]
                got = Q.read(geo, ((final + 1) * 127.5)[:, 0].cpu().numpy())
                ok += sum(g_.strip() == e.answer for g_, e in zip(got, chunk))
            res[name] = round(ok / len(ex), 4)
    return res


def main_train(a) -> None:
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    world = int(os.environ.get("WORLD_SIZE", "1"))
    ddp = world > 1
    cuda = torch.cuda.is_available()
    if ddp:   # rank 0 evaluates while the others wait at a barrier: a long timeout (the default 10 min can be too short)
        import datetime
        dist.init_process_group(backend="nccl" if cuda else "gloo", timeout=datetime.timedelta(hours=3))
    rank = dist.get_rank() if ddp else 0
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if cuda:
        torch.cuda.set_device(local)
    dev = "cuda" if cuda else "cpu"
    assert a.batch % world == 0, "--batch must be divisible by the number of processes"
    bpr = a.batch // world
    torch.manual_seed(a.seed)
    P.S = a.px
    P.CKPT_ACT = a.ckpt_act
    geo = Q.Geo(a.px)
    task = T.make_task(a.task, a.px, a.seed, a.kb_people)
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
    n_nonembed = nonembed_params(model)
    fpe = flops_per_example(n_nonembed, a.px)
    net = DDP(model, device_ids=[local] if cuda else None) if ddp else model
    G = torch.from_numpy(Q.glyphs()).to(dev)
    images = make_images(G, geo, dev)
    images.px = a.px
    if rank == 0:
        tests = task.test_sets(a.n_test)
        vs = val_set(task, geo, a.arm, a.px, a.val_per_set)
        (a.out / "config.json").write_text(json.dumps({"trainer": "train_text", **{k: str(v) for k, v in vars(a).items()}, "world": world,
                                                       "params": n_params, "nonembed_params": n_nonembed,
                                                       "flops_per_example": fpe, "tests": {k: len(v) for k, v in tests.items()},
                                                       "val_per_set": a.val_per_set}, indent=1))
    rng = random.Random(f"{a.seed}-{rank}-{step}")

    def batch():
        ex = [task.sample(rng) for _ in range(bpr)]
        prs = [T.pair(e, a.arm, rng, a.per_frame) for e in ex]
        return (images(np.stack([T.frame_codes(geo, p_[0], p_[1]) for p_ in prs])),
                images(np.stack([T.frame_codes(geo, p_[2], p_[3]) for p_ in prs])))

    def evaluate(loss_now):
        res = {"step": step, "examples": step * a.batch, "flops": step * a.batch * fpe, "train_loss": round(float(loss_now.detach()), 6),
               **val_loss(ema, vs, images, dev), "wall_hours": round((time.time() - t_start) / 3600, 3)}
        if step % a.acc_every == 0 or step == a.steps:
            res["exact"] = accuracy(ema, tests, images, geo, a.arm, dev, a.sample_steps, a.eval_batch)
        ema.train()
        with (a.out / "evals.jsonl").open("a") as f:
            f.write(json.dumps(res) + "\n")
        print(json.dumps(res), flush=True)

    net.train(); t0 = t_start = time.time(); s0 = step; loss = torch.tensor(0.0)
    while step < a.steps:
        x1, y = batch()
        lr = a.lr * min(1.0, (step + 1) / a.warmup)
        if a.decay_from and step >= a.decay_from:
            lr *= max(0.0, (a.steps - step) / max(1, a.steps - a.decay_from))
        for g_ in opt.param_groups:
            g_["lr"] = lr
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=cuda):
            if a.arm == "video" and a.ctx_noise:
                x1 = x1 + a.ctx_noise * torch.randn_like(x1)
            t = torch.sigmoid(torch.randn(len(x1), device=dev))
            e = torch.randn_like(y)
            xt = t[:, None, None, None] * y + (1 - t[:, None, None, None]) * e
            xh = net(x1, xt, t).float()
            loss = (((xh - xt) / (1 - t).clamp(min=0.05)[:, None, None, None] - (y - e)) ** 2).mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        ok = torch.isfinite(gnorm)
        if ddp:   # every rank must take the same decision (the gradients are already averaged, but be explicit)
            okt = torch.tensor([1.0 if ok else 0.0], device=dev); dist.all_reduce(okt, op=dist.ReduceOp.MIN); ok = bool(okt.item() > 0)
        if ok:
            opt.step()
            if ema is not None:
                with torch.no_grad():
                    for pe, pm in zip(ema.parameters(), model.parameters()):
                        pe.mul_(a.ema).add_(pm.detach(), alpha=1 - a.ema)
        step += 1
        if rank == 0 and step % 200 == 0:
            print(json.dumps({"step": step, "loss": round(float(loss), 6), "lr": lr, "sec_per_step": round((time.time() - t0) / (step - s0), 4)}), flush=True)
        if rank == 0 and a.snap_every and step % a.snap_every == 0:   # EMA weights (bf16) for metrics computed later
            (a.out / "snaps").mkdir(exist_ok=True)
            torch.save({k: v.to(torch.bfloat16) for k, v in ema.state_dict().items()}, a.out / "snaps" / f"ema_{step:06d}.pt")
        if rank == 0 and (step % a.save_every == 0 or step == a.steps):
            if ck.exists() and step % a.keep_every == 0:   # a second, older checkpoint to roll back to after a loss spike
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
    """the fixed-set losses and step metrics of a finished run's EMA weights -> RUN/valloss.json; with --snaps, of every
    snapshot in RUN/snaps too -> RUN/valloss_curve.jsonl"""
    import torch
    cfg = json.loads((a.run / "config.json").read_text())
    px, size, spec, arm = int(cfg["px"]), cfg["size"], cfg["task"], cfg["arm"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    P.S = px
    geo = Q.Geo(px)
    task = T.make_task(spec, px, int(cfg.get("seed", 0)), int(cfg.get("kb_people", 300)))
    ema = P.build(size).to(dev)
    G = torch.from_numpy(Q.glyphs()).to(dev)
    images = make_images(G, geo, dev); images.px = px
    vs = val_set(task, geo, arm, px, a.val_per_set)
    n_nonembed = nonembed_params(ema)
    batch = int(cfg.get("batch", 256))

    def one(state, step):
        ema.load_state_dict({k: v.float() for k, v in state.items()})
        return {"run": str(a.run), "step": step, "size": size, "params": sum(p.numel() for p in ema.parameters()),
                "nonembed_params": n_nonembed, "batch": batch, "flops": (step or 0) * batch * flops_per_example(n_nonembed, px),
                "val_per_set": a.val_per_set,
                **val_loss(ema, vs, images, dev, a.batch), **step_metrics(ema, vs, images, geo, G, dev, a.sample_steps, a.batch)}
    if not a.snaps_only:
        s = torch.load(a.run / "ckpt.pt", map_location="cpu")
        res = one(s["ema"], s.get("step"))
        (a.run / "valloss.json").write_text(json.dumps(res, indent=1))
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
    t.add_argument("--task", required=True, help="a text_tasks spec, or multi:A+B+C")
    t.add_argument("--arm", choices=("direct", "video"), required=True, help="direct answering, or frame by frame")
    t.add_argument("--size", choices=sorted(P.SIZES), default="S")
    t.add_argument("--px", type=int, default=192)
    t.add_argument("--steps", type=int, default=40000)
    t.add_argument("--batch", type=int, default=256, help="global batch (split over the processes)")
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--wd", type=float, default=0.0)
    t.add_argument("--warmup", type=int, default=1000)
    t.add_argument("--decay-from", type=int, default=0, help="linear lr cool-down from this step to 0 at --steps")
    t.add_argument("--ema", type=float, default=0.999)
    t.add_argument("--ctx-noise", type=float, default=0.03, help="frame by frame: Gaussian noise on the input canvas")
    t.add_argument("--per-frame", type=int, default=1)
    t.add_argument("--kb-people", type=int, default=300)
    t.add_argument("--sample-steps", type=int, default=30)
    t.add_argument("--n-test", type=int, default=128)
    t.add_argument("--val-per-set", type=int, default=VAL_PER_SET)
    t.add_argument("--eval-batch", type=int, default=64)
    t.add_argument("--eval-every", type=int, default=2000, help="validation loss")
    t.add_argument("--acc-every", type=int, default=4000, help="answer accuracy (a multiple of --eval-every)")
    t.add_argument("--save-every", type=int, default=1000)
    t.add_argument("--keep-every", type=int, default=5000, help="every N steps the previous checkpoint is kept as ckpt_prev.pt")
    t.add_argument("--snap-every", type=int, default=4000, help="keep the EMA weights (bf16) every N steps in RUN/snaps (0: off)")
    t.add_argument("--ckpt-act", action="store_true", help="activation checkpointing (less memory, ~30%% slower)")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--out", type=Path, required=True)
    v = sub.add_parser("valloss")
    v.add_argument("--run", type=Path, required=True)
    v.add_argument("--val-per-set", type=int, default=VAL_PER_SET, help="held-out transitions per test set (Sec. 5: 4096)")
    v.add_argument("--batch", type=int, default=64)
    v.add_argument("--sample-steps", type=int, default=10, help="sampler steps of the step metrics")
    v.add_argument("--snaps", action="store_true", help="also every snapshot in RUN/snaps -> RUN/valloss_curve.jsonl")
    v.add_argument("--snaps-only", action="store_true", help="only the snapshots (RUN/valloss.json is left as it is)")
    a = ap.parse_args()
    {"train": main_train, "valloss": main_valloss}[a.cmd](a)


if __name__ == "__main__":
    main()
