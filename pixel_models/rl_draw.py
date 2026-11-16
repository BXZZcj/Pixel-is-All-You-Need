"""RFT and its supervised controls on the cross-modal tasks (Sec. 7, Appendix H).

Every run continues from the same start model, trained on non-optimal demonstrations (draw_tasks / train_joint with a
:detour spec), for the same number of iterations:

  --algo rft --task SPEC:detour        rejection-sampling fine-tuning: each iteration samples --group rollouts of free
                                       length for each of --prompts problems, keeps the correct rollouts with the fewest
                                       frames (Sum Mazes: a shortest path with that path's own total; Trace Then Sum:
                                       both written values right too), redraws them as clean frames from what the model
                                       wrote, and trains on them with the supervised loss
  --algo sft --task SPEC:detour        non-optimal SFT: the supervised loss on the non-optimal demonstrations
  --algo sft --task SPEC:clean         optimal SFT (Sum Mazes: maze:5:sum:canon:as-detour): on the optimal ones

Rollouts sample each frame with the SDE of rl_text (--rl-steps steps, g = 0.7 (1 - t)) and end when the model writes an
answer. Evaluation (deterministic sampler, --sample-steps): "minimal" is the share of test problems solved in the fewest
frames, the accuracy of Sec. 7 (Sum Mazes: "valid", a shortest path with the right total).

usage: python -m pixel_models.rl_draw --init START/ckpt.pt --task countpaint:tallysel:detour --algo rft --size S --out DIR
Resumable: policy, EMA, optimizer and iteration are saved every --save-every iterations.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import time
from pathlib import Path

import numpy as np

from . import canvas as Q
from . import draw_tasks as D
from . import maze_tasks as W
from . import model as P


class WorldTask:
    """a maze_tasks task behind draw_tasks' Task interface"""

    def __init__(self, spec: str, px: int):
        self.t = W.Task(spec, px)
        self.geo, self.name, self.spec = self.t.geo, self.t.name, spec

    def train(self, rng):
        return self.t.train(rng)

    def test_sets(self, n):
        return self.t.test_sets(n)


def make_task(spec: str, px: int):
    return WorldTask(spec, px) if spec.split(":")[0] == "maze" else D.Task(spec, px)


def render(task, sc, geo, f):
    return W.render(sc, geo, f) if isinstance(task, WorldTask) else D.render(sc, geo, f)


def n_frames(task, sc):
    return W.n_frames(sc) if isinstance(task, WorldTask) else D.n_frames(sc)


def pair(task, sc, geo, arm, rng):
    return W.pair(sc, geo, rng) if isinstance(task, WorldTask) else D.pair(sc, geo, arm, rng, 0.0, task.per_frame)


def score(task, geo, scs, starts, finals, arm):
    """per episode: dict of the metrics read from the final frame (uint8 arrays)"""
    got = [g.strip() for g in Q.read(geo, finals)]
    out = []
    if task.name == "countpaint":
        notes = D.read_row(geo, finals, geo.q_rows - 1)
        for sc, s0, f, g, nt in zip(scs, starts, finals, got, notes):
            ious = D.paint_ious(sc, geo, s0, f)
            want = [st[2] for st in sc.steps if st[0] in ("note", "tally")][-1]
            letter, counts, iou = g == sc.answer, nt == want, ious[sc.chosen]
            got = {t[0]: t[1:] for t in nt.split() if len(t) >= 2 and t[0] in "ABC"}
            frac = sum(got.get(chr(65 + j)) == str(c) for j, c in enumerate(sc.counts)) / len(sc.counts)
            out.append({"letter": letter, "counts": counts, "count_frac": frac, "iou": iou,
                        "pick": int(np.argmax(ious)) == sc.chosen and max(ious) >= 0.5,
                        "solved": letter and iou >= 0.8 and (counts or arm == "direct")})
    elif task.name == "maze":   # solved: the written moves are a shortest path (any of them); letter: the demonstrated one
        for sc, f, g in zip(scs, finals, got):
            m = W.score_maze(sc, geo, f, g)
            out.append({"letter": bool(m["exact"]), "char": m["path_iou"], "solved": bool(m["valid"]),
                        "valid": float(m["valid"]), "legal": float(m["legal"]), "extra": m["extra"], "path_iou": m["path_iou"]})
    else:
        # Trace Then Sum: also whether the two values written on the note row ("A=2 F=3") are right - RFT keeps only
        # episodes with them right (a sum right by luck must not be redrawn with the true lookups)
        notes = D.read_row(geo, finals, geo.q_rows - 1) if task.name == "tracesum" and arm == "video" else None
        for j, (sc, g) in enumerate(zip(scs, got)):
            char = sum(p == q for p, q in zip(g.rjust(len(sc.answer)), sc.answer)) / max(1, len(sc.answer))
            m = {"letter": g == sc.answer, "char": char, "solved": g == sc.answer}
            if notes is not None:
                want = [st[2] for st in (getattr(sc, "base_steps", None) or sc.steps) if st[0] == "note"][-1]
                m["notes"] = notes[j] == want
                m["solved_notes"] = m["solved"] and m["notes"]
            out.append(m)
    return out


def main_train(a) -> None:
    import torch
    torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    a.out.mkdir(parents=True, exist_ok=True)
    arm = "video"
    task = make_task(a.task, a.px)
    MAZE = isinstance(task, WorldTask) and task.name == "maze"
    assert MAZE or getattr(task, "free", False), "a free-length task (a :detour / :clean spec, or a maze)"
    geo = task.geo
    P.S = a.px
    tests = task.test_sets(a.n_test)
    # three networks are built (the second is unused) so that the torch RNG stream matches the runs of the paper
    policy, _, ema = (P.build(a.size).to(dev) for _ in range(3))
    init = torch.load(a.init, map_location="cpu")
    init = {k: v.float() for k, v in (init["ema"] if "ema" in init else init).items()}   # ckpt.pt or an EMA snapshot
    for m in (policy, ema):
        m.load_state_dict(init)
    ema.requires_grad_(False)
    opt = torch.optim.AdamW(policy.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    it, ck = 0, a.out / "ckpt.pt"
    if ck.exists():
        s = torch.load(ck, map_location="cpu")
        if "iter" in s:
            policy.load_state_dict(s["model"]); ema.load_state_dict(s["ema"]); opt.load_state_dict(s["opt"]); it = s["iter"]
            print(f"resumed at iteration {it}", flush=True)
    (a.out / "config.json").write_text(json.dumps({"trainer": "rl_draw", **{k: str(v) for k, v in vars(a).items()}}, indent=1))
    rng = random.Random(a.seed * 7919 + it)
    ac = lambda: torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda")
    K1 = a.rl_steps - 1
    ts = torch.linspace(0, 1, a.rl_steps + 1, device=dev)
    to_t = lambda arrs: torch.from_numpy(np.stack(arrs)).to(dev).float()[:, None] / 127.5 - 1
    to_u8 = lambda x: ((x.float().clamp(-1, 1) + 1) * 127.5)[:, 0].round().clamp(0, 255).byte().cpu().numpy()
    max_frames = W.MAX_FRAMES if MAZE else a.free_frames

    def drift(net, cond, x, kk):
        t, t1 = ts[kk], ts[kk + 1]
        y = net(cond, x, t).float()
        one = (1 - t).clamp(min=0.05)[:, None, None, None]
        dt = (t1 - t)[:, None, None, None]
        g = a.noise * one
        v = (y - x) / one
        e = (x - t[:, None, None, None] * y) / one
        return x + dt * (v - 0.5 * g * g * e / one), g * dt.sqrt()

    @torch.no_grad()
    def rollout(cur):
        """SDE rollouts of free length: each ends at the first frame with an answer -> (final frames, lengths)"""
        n = len(cur)
        done = torch.zeros(n, dtype=torch.bool, device=dev)
        final = cur.clone()
        length = torch.full((n,), max_frames, dtype=torch.long, device=dev)
        for f in range(max_frames):
            x = torch.randn(cur.shape, device=dev)
            for k in range(K1):
                kk = torch.full((n,), k, device=dev, dtype=torch.long)
                with ac():
                    mu, sd = drift(policy, cur, x, kk)
                x = mu + sd * torch.randn_like(x)
            with ac():
                y = policy(cur, x, ts[K1].repeat(n)).float().clamp(-1, 1)
            cur = y
            wrote = torch.tensor(W.answer_written(geo, to_u8(y)), device=dev)
            new = wrote & ~done
            final[new] = y[new]; length[new] = f + 1; done |= wrote
            if bool(done.all()):
                break
        final[~done] = cur[~done]
        return final, length

    @torch.no_grad()
    def evaluate():
        ema.eval(); res = {"iter": it, "algo": a.algo}
        tt = torch.linspace(0, 1, a.sample_steps + 1, device=dev)
        for name, ex in tests.items():
            ms = []
            for i in range(0, len(ex), a.eval_batch):
                chunk = ex[i:i + a.eval_batch]
                starts = [render(task, sc, geo, 0) for sc in chunk]
                cur = to_t(starts)
                g_ = torch.Generator(device=dev).manual_seed(7)
                final, done = cur.clone(), torch.zeros(len(chunk), dtype=torch.bool, device=dev)
                length = torch.full((len(chunk),), max_frames, dtype=torch.long, device=dev)
                for fr_ in range(max_frames):
                    x = torch.randn(cur.shape, device=dev, generator=g_)
                    with ac():
                        for k in range(a.sample_steps):
                            xh = ema(cur, x, tt[k].repeat(len(chunk))).float()
                            x = x + (tt[k + 1] - tt[k]) * (xh - x) / (1 - tt[k]).clamp(min=0.05)
                    cur = x.float().clamp(-1, 1)
                    wrote = torch.tensor(W.answer_written(geo, to_u8(cur)), device=dev)
                    new = wrote & ~done
                    final[new] = cur[new]; length[new] = fr_ + 1; done |= wrote
                    if bool(done.all()):
                        break
                final[~done] = cur[~done]
                mc = score(task, geo, chunk, starts, to_u8(final), arm)
                if not MAZE:
                    for m, sc, L in zip(mc, chunk, length.tolist()):
                        m["minimal"], m["length"] = float(bool(m["solved"]) and int(L) == sc.n_min), float(int(L) - sc.n_min)
                ms += mc
            r = {"exact": np.mean([m["letter"] for m in ms])}
            for k in ("iou", "pick", "counts", "count_frac", "solved", "minimal", "length", "valid", "legal", "extra", "path_iou"):
                if k in ms[0]:
                    r[k] = float(np.nanmean([m[k] for m in ms]))
            res[name] = {k: round(float(v), 4) for k, v in r.items()}
        with (a.out / "evals.jsonl").open("a") as f:
            f.write(json.dumps(res) + "\n")
        print(json.dumps(res), flush=True)

    def flow_loss(c, y, c_clean):
        """the supervised loss of draw_tasks, with --chg-weight on the pixels the step changes"""
        with ac():
            t = torch.sigmoid(torch.randn(len(c), device=dev))
            e = torch.randn_like(y)
            xt = t[:, None, None, None] * y + (1 - t[:, None, None, None]) * e
            xh = policy(c, xt, t).float()
            lmap = ((xh - xt) / (1 - t).clamp(min=0.05)[:, None, None, None] - (y - e)) ** 2
            if not a.chg_weight:
                return lmap.mean()
            w = 1 + a.chg_weight * torch.nn.functional.max_pool2d(((y - c_clean).abs() > 0.1).float(), 5, stride=1, padding=2)
            return (lmap * w).sum() / w.sum()

    def fit(prs, mb):
        """the supervised loss on (input, target) frame pairs, minibatches of mb, context noise on the input"""
        loss = None
        for j in range(0, len(prs), mb):
            chunk = prs[j:j + mb]
            c0, y = to_t([p_[0] for p_ in chunk]), to_t([p_[1] for p_ in chunk])
            c = c0 + a.ctx_noise * torch.randn_like(c0)
            loss = flow_loss(c, y, c0)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0); opt.step()
        return loss

    if it == 0:
        evaluate()
    t0 = time.time()
    while it < a.iters:
        scs = [task.train(rng) for _ in range(a.prompts)]
        policy.train()
        if a.algo == "sft":   # ground-truth frame pairs of the demonstrations
            prs = [pair(task, sc, geo, arm, rng) for sc in scs for _ in range(a.group)]
            c0, y = to_t([p_[0] for p_ in prs]), to_t([p_[1] for p_ in prs])
            c = c0 + a.ctx_noise * torch.randn_like(c0)
            for j in range(0, len(c), a.minibatch * 4):
                sl = slice(j, j + a.minibatch * 4)
                loss = flow_loss(c[sl], y[sl], c0[sl])
                opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0); opt.step()
            stats = {"loss": round(float(loss), 6)}
        else:   # rft
            scg = [sc for sc in scs for _ in range(a.group)]
            starts = [render(task, sc, geo, 0) for sc in scs]
            policy.eval()
            final, length = rollout(to_t(starts).repeat_interleave(a.group, 0))
            starts_g = [s0 for s0 in starts for _ in range(a.group)]
            fin_u8 = to_u8(final)
            ms = score(task, geo, scg, starts_g, fin_u8, arm)
            if not MAZE:   # minimal: solved in n_min frames, the fewest
                for m, sc, L in zip(ms, scg, length.tolist()):
                    m["minimal"] = bool(m["solved"]) and int(L) == sc.n_min
            stats = {"train_solved": round(float(np.mean([m["solved"] for m in ms])), 4),
                     "train_letter": round(float(np.mean([m["letter"] for m in ms])), 4),
                     "train_len": round(float(length.float().mean()), 2)}
            if not MAZE:
                stats["train_minimal"] = round(float(np.mean([m["minimal"] for m in ms])), 4)
            policy.train()
            if MAZE:
                # Sum Mazes: the model's own solved paths (a shortest path and its own total) redrawn as clean frames
                good = [i for i, m in enumerate(ms) if m["solved"]]
                got = [g_.strip() for g_ in Q.read(geo, fin_u8)]
                prs = []
                for i in good:
                    sc2 = W.own_scene(scg[i], geo, fin_u8[i], got[i])
                    prs += [W.pair(sc2, geo, rng) for _ in range(W.n_frames(sc2))]
            else:
                # the problems solved in the fewest frames (Trace Then Sum: both written values right too), redrawn as
                # clean frames: a minimal solved episode is the detour-free demonstration of its problem
                good = [i for i, m in enumerate(ms) if m["minimal"] and m.get("notes", True)]
                prs = []
                for i in good:
                    sc2 = copy.copy(scg[i]); sc2.steps = list(sc2.base_steps)
                    prs += [D.pair(sc2, geo, "video", rng) for _ in range(D.n_frames(sc2))]
            random.Random(it).shuffle(prs)
            prs = prs[: a.max_transitions]
            stats["kept"] = len(good)
            fit(prs, a.minibatch * 4)
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), policy.parameters()):
                pe.mul_(a.ema).add_(pm.detach(), alpha=1 - a.ema)
        it += 1
        if it % 10 == 0:
            print(json.dumps({"iter": it, **stats, "sec_per_iter": round((time.time() - t0) / 10, 2)}), flush=True)
            t0 = time.time()
            with (a.out / "train_log.jsonl").open("a") as f:
                f.write(json.dumps({"iter": it, **stats}) + "\n")
        if it % a.save_every == 0 or it == a.iters:
            torch.save({"model": policy.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "iter": it}, str(ck) + ".tmp")
            os.replace(str(ck) + ".tmp", ck)
        if it % a.eval_every == 0 or it == a.iters:
            evaluate()
    (a.out / "TRAIN_DONE").write_text(str(it))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", type=Path, required=True, help="ckpt.pt of the start model (its EMA weights)")
    ap.add_argument("--task", required=True, help="a :detour / :clean spec of draw_tasks, or maze:5:sum:detour / maze:5:sum:canon:as-detour")
    ap.add_argument("--algo", choices=("rft", "sft"), required=True)
    ap.add_argument("--size", choices=sorted(P.SIZES), default="S")
    ap.add_argument("--px", type=int, default=192)
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--prompts", type=int, default=16, help="problems per iteration")
    ap.add_argument("--group", type=int, default=8, help="rollouts per problem (sft: frame pairs per problem)")
    ap.add_argument("--rl-steps", type=int, default=10, help="SDE sampling steps per frame of the rollouts")
    ap.add_argument("--noise", type=float, default=0.7, help="a in g = a (1 - t)")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--ema", type=float, default=0.99)
    ap.add_argument("--ctx-noise", type=float, default=0.03)
    ap.add_argument("--chg-weight", type=float, default=30.0, help="weight of the pixels a step changes, as draw_tasks")
    ap.add_argument("--minibatch", type=int, default=32)
    ap.add_argument("--max-transitions", type=int, default=512, help="rft: frame pairs per iteration used in the update")
    ap.add_argument("--free-frames", type=int, default=32, help="the frame budget of a rollout (mazes: 22)")
    ap.add_argument("--n-test", type=int, default=128)
    ap.add_argument("--eval-batch", type=int, default=64)
    ap.add_argument("--sample-steps", type=int, default=10, help="evaluation sampler steps")
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    main_train(ap.parse_args())


if __name__ == "__main__":
    main()
