"""GRPO post-training of the frame-by-frame models on the text tasks (Sec. 7, Appendix H).

The policy starts from a supervised text_tasks run trained with --per-frame 3 (each frame of the demonstrations carries
out 1-3 steps, so the number of frames is the model's choice). An episode is the rollout from the problem canvas, frame
by frame, until the model writes an answer (the answer rows are not blank) or --max-frames. Each frame is sampled with
an SDE that has the same marginals as the deterministic sampler, so every sampling step is a Gaussian transition with a
tractable log-probability (Flow-GRPO):

    t = 0 noise, t = 1 data, x_t = t y + (1 - t) e, the model predicts y;  v = (y^ - x) / (1 - t),
    e^ = (x - t y^) / (1 - t),  dx = [v - g^2 / 2 e^ / (1 - t)] dt + g dW,  g = 0.7 (1 - t)

with --rl-steps 10 steps per frame, the last one returning y^ without noise. Reward: 1 for an exact answer plus 0.2 x
the fraction of right characters. Advantages are normalized within each group of --group samples of a problem and
clipped at +-5, and every transition of a rollout shares its advantage. All transitions of an iteration (at most
--max-transitions) are accumulated into one gradient step (the ratio stays 1: plain policy gradient with the group
baseline), with a KL penalty of weight --beta to the supervised model. Training problems are fresh problems of the
hardest test level (--prompt-set k4 for Expressions; --hard-only for Word Problems), test items excluded.

Evaluation (the supervised runs' own test sets, deterministic 30-step sampler): KEY = after the true number of frames,
KEY_free = free-running until an answer is written (the number reported in Sec. 7), KEY_frames = mean frames used.

usage: python -m pixel_models.rl_text --init RUN/ckpt.pt --task expr:4 --prompt-set k4 --size S --out DIR
Resumable: policy, EMA, optimizer and iteration are saved every --save-every iterations.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np

from . import canvas as Q
from . import model as P
from . import text_tasks as T


def prompt_sampler(a, task, tests, rng):
    """the problems RL trains on: fresh problems of the kind of one test set (--prompt-set), or only training problems
    with the most steps (--hard-only); test items are excluded"""
    test_prompts = {e.prompt for ex in tests.values() for e in ex}
    hard_rows = 0
    if a.hard_only:
        prng = random.Random(f"probe-{a.task}")
        hard_rows = max(len(task.sample(prng).rows) for _ in range(400))

    def draw():
        while True:
            if a.prompt_set:
                e = task.sets[a.prompt_set](rng)
                if not task.fits(e):
                    continue
            else:
                e = task.sample(rng)
            if e.prompt in test_prompts or (a.hard_only and len(e.rows) < hard_rows):
                continue
            return e
    return draw


def main_train(a) -> None:
    import torch
    torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    a.out.mkdir(parents=True, exist_ok=True)
    geo = Q.Geo(a.px)
    P.S = a.px
    task = T.Task(a.task, a.px, a.seed)
    tests = task.test_sets(a.n_test)
    policy, ref, ema = (P.build(a.size).to(dev) for _ in range(3))
    init = torch.load(a.init, map_location=dev)["ema"]
    for m in (policy, ref, ema):
        m.load_state_dict(init)
    ref.requires_grad_(False); ema.requires_grad_(False); ref.eval()
    opt = torch.optim.AdamW(policy.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    it, ck = 0, a.out / "ckpt.pt"
    if ck.exists():
        s = torch.load(ck, map_location=dev)
        if "iter" in s:
            policy.load_state_dict(s["model"]); ema.load_state_dict(s["ema"]); opt.load_state_dict(s["opt"]); it = s["iter"]
            print(f"resumed at iteration {it}", flush=True)
    (a.out / "config.json").write_text(json.dumps({"trainer": "rl_text", **{k: str(v) for k, v in vars(a).items()}}, indent=1))
    rng = random.Random(a.seed * 7919 + it)
    G = torch.from_numpy(Q.glyphs()).to(dev)
    ac = lambda: torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda")
    K1 = a.rl_steps - 1
    ts = torch.linspace(0, 1, a.rl_steps + 1, device=dev)

    def images(m):
        x = G[torch.from_numpy(m).long().to(dev)].permute(0, 1, 3, 2, 4).reshape(len(m), 1, geo.px, geo.px)
        x[:, :, geo.line_y, 4:geo.px - 4] = 0
        return x.float() / 127.5 - 1

    def first_canvases(exs):
        return images(np.stack([T.frame_codes(geo, e.rows[0], None) for e in exs]))

    def read(x):
        return [g.strip() for g in Q.read(geo, ((x.float().clamp(-1, 1) + 1) * 127.5)[:, 0].cpu().numpy())]

    def drift(net, cond, x, kk):
        """kk: (n,) step indices; -> the mean and std of the next state, per sample"""
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
        n = len(cur)
        F = {"cond": [], "xs": [], "nxt": [], "old": [], "y": [], "alive": []}
        live = torch.ones(n, dtype=torch.bool, device=dev)
        answers, used = [""] * n, [a.max_frames] * n
        for f in range(a.max_frames):
            x = torch.randn(cur.shape, device=dev)
            xs, nx, old = [], [], []
            for k in range(K1):
                kk = torch.full((n,), k, device=dev, dtype=torch.long)
                with ac():
                    mu, sd = drift(policy, cur, x, kk)
                z = torch.randn_like(x)
                x1 = mu + sd * z
                xs.append(x); nx.append(x1); old.append(-(z ** 2).mean((1, 2, 3)) / 2); x = x1
            with ac():
                y = policy(cur, x, ts[K1].repeat(n)).float().clamp(-1, 1)
            F["cond"].append(cur); F["xs"].append(torch.stack(xs, 1)); F["nxt"].append(torch.stack(nx, 1))
            F["old"].append(torch.stack(old, 1)); F["y"].append(y); F["alive"].append(live.clone())
            for i, g_ in enumerate(read(y)):
                if live[i] and g_:
                    live[i] = False; answers[i] = g_; used[i] = f + 1
            if not live.any():
                break
            cur = y
        return F, answers, used

    def reward(answers, exs):
        r = []
        for g_, e in zip(answers, exs):
            char = sum(p == q for p, q in zip(g_.rjust(len(e.answer)), e.answer)) / max(1, len(e.answer)) if g_ else 0.0
            r.append(float(g_ == e.answer) + a.partial * char)
        return torch.tensor(r, device=dev), sum(g_ == e.answer for g_, e in zip(answers, exs)) / len(exs)

    @torch.no_grad()
    def evaluate():
        ema.eval(); res = {"iter": it}
        tt = torch.linspace(0, 1, 31, device=dev)
        for name, ex in tests.items():
            ok_o = ok_f = fr_sum = 0
            for i in range(0, len(ex), 64):
                chunk = ex[i:i + 64]
                cur = first_canvases(chunk)
                Ts = [len(e.rows) for e in chunk]
                g_ = torch.Generator(device=dev).manual_seed(7 + i)
                o_ans, f_ans, f_n = [None] * len(chunk), [None] * len(chunk), [a.max_frames] * len(chunk)
                for fr in range(max(a.max_frames, max(Ts))):
                    x = torch.randn(cur.shape, device=dev, generator=g_)
                    with ac():
                        for k in range(30):
                            xh = ema(cur, x, tt[k].repeat(len(chunk))).float()
                            x = x + (tt[k + 1] - tt[k]) * (xh - x) / (1 - tt[k]).clamp(min=0.05)
                    cur = x.float().clamp(-1, 1)
                    for j, gj in enumerate(read(cur)):
                        if fr + 1 == Ts[j]:
                            o_ans[j] = gj
                        if f_ans[j] is None and gj and fr < a.max_frames:
                            f_ans[j], f_n[j] = gj, fr + 1
                    if all(v is not None for v in o_ans) and (all(v is not None for v in f_ans) or fr + 1 >= a.max_frames):
                        break
                ok_o += sum(o == e.answer for o, e in zip(o_ans, chunk))
                ok_f += sum(f == e.answer for f, e in zip(f_ans, chunk))
                fr_sum += sum(f_n)
            res[name] = round(ok_o / len(ex), 4)
            res[name + "_free"] = round(ok_f / len(ex), 4)
            res[name + "_frames"] = round(fr_sum / len(ex), 2)
        with (a.out / "evals.jsonl").open("a") as f:
            f.write(json.dumps(res) + "\n")
        print(json.dumps(res), flush=True)

    if it == 0:
        evaluate()
    t0 = time.time()
    draw = prompt_sampler(a, task, tests, rng)
    while it < a.iters:
        exs = [draw() for _ in range(a.prompts)]
        exg = [e for e in exs for _ in range(a.group)]
        policy.eval()
        Fr, answers, used = rollout(first_canvases(exs).repeat_interleave(a.group, 0))
        r, acc = reward(answers, exg)
        rr = r.view(a.prompts, a.group)
        adv = ((rr - rr.mean(1, keepdim=True)) / (rr.std(1, keepdim=True) + 1e-4)).view(-1)
        stats = {"reward": round(float(r.mean()), 4), "train_exact": round(acc, 4), "groups_with_signal": int((rr.std(1) > 1e-6).sum()),
                 "frames": round(sum(used) / len(used), 2), "answered": round(sum(bool(x) for x in answers) / len(answers), 3)}
        policy.train()
        nF = len(Fr["cond"])
        if stats["groups_with_signal"]:
            pairs = [(i, f, k) for f in range(nF) for i in Fr["alive"][f].nonzero().squeeze(1).tolist()
                     if abs(float(adv[i])) > 1e-6 for k in range(K1)]
            random.Random(it).shuffle(pairs)
            pairs = pairs[: a.max_transitions]
            kls = devs = 0.0
            opt.zero_grad(set_to_none=True)
            for j in range(0, len(pairs), a.minibatch):
                P_ = pairs[j:j + a.minibatch]
                ii = torch.tensor([p_[0] for p_ in P_], device=dev)
                kk = torch.tensor([p_[2] for p_ in P_], device=dev)
                c = torch.stack([Fr["cond"][f][i] for i, f, k in P_])
                x = torch.stack([Fr["xs"][f][i, k] for i, f, k in P_])
                x1 = torch.stack([Fr["nxt"][f][i, k] for i, f, k in P_])
                old = torch.stack([Fr["old"][f][i, k] for i, f, k in P_])
                with ac():
                    mu, sd = drift(policy, c, x, kk)
                    with torch.no_grad():
                        mu_ref, _ = drift(ref, c, x, kk)
                logp = -(((x1 - mu.float()) / sd) ** 2).mean((1, 2, 3)) / 2
                ratio = torch.exp(logp - old)   # 1 up to bf16 noise: one gradient step per iteration
                A = adv[ii].clamp(-a.adv_clip, a.adv_clip)
                pg = -ratio * A
                kl = (((mu.float() - mu_ref.float()) / sd) ** 2).mean((1, 2, 3)) / 2
                loss = (pg + a.beta * kl).mean()
                kls += float(kl.detach().sum()); devs += float((ratio.detach() - 1).abs().sum())
                (loss * len(P_) / len(pairs)).backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0); opt.step()
            stats.update(kl=round(kls / max(1, len(pairs)), 5), ratio_dev=float(f"{devs / max(1, len(pairs)):.3g}"), transitions=len(pairs))
        del Fr
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
    ap.add_argument("--init", type=Path, required=True, help="ckpt.pt of a text_tasks run trained with --per-frame 3 (its EMA weights)")
    ap.add_argument("--task", required=True, help="expr:4 | wp:3")
    ap.add_argument("--max-frames", type=int, default=10, help="an episode ends when an answer is written or here")
    ap.add_argument("--prompt-set", help="train on fresh problems of the kind of this test set (its own items excluded)")
    ap.add_argument("--hard-only", action="store_true", help="train only on training problems with the most steps")
    ap.add_argument("--size", choices=sorted(P.SIZES), default="S")
    ap.add_argument("--px", type=int, default=192)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--prompts", type=int, default=16, help="problems per iteration")
    ap.add_argument("--group", type=int, default=8, help="samples per problem")
    ap.add_argument("--rl-steps", type=int, default=10, help="SDE sampling steps per frame of the rollouts")
    ap.add_argument("--noise", type=float, default=0.7, help="a in g = a (1 - t)")
    ap.add_argument("--partial", type=float, default=0.2, help="weight of the fraction of right characters in the reward")
    ap.add_argument("--adv-clip", type=float, default=5.0)
    ap.add_argument("--beta", type=float, default=0.004, help="KL weight to the supervised model")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--ema", type=float, default=0.99)
    ap.add_argument("--minibatch", type=int, default=32)
    ap.add_argument("--max-transitions", type=int, default=2048, help="transitions per iteration used in the update")
    ap.add_argument("--n-test", type=int, default=128)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    main_train(ap.parse_args())


if __name__ == "__main__":
    main()
