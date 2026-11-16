"""The four text tasks of Sec. 4.1 and their trainer for Secs. 6 and 7.

  expr:K     Expressions: operands 1-9 (80%) or 10-20, 1..K operations in training; each frame rewrites one operation
             (the leftmost multiplication first, then left to right). Test sets k1, k2, k3, k4, k6, k8.
  mul:AxB    Multiplication: each frame writes one partial product, right aligned and shifted; the last frame the sum.
  wp:K       Word Problems: a person starts with 2-60 objects, each sentence adds, removes, multiplies or halves them;
             each frame writes one equation. Training names are made up, test names are real names training never
             showed. Drawn on a 256 x 256 canvas (the problems need more rows).
  hop:K      Factual Questions: a knowledge base of 300 people (city, job, mother; cities belong to countries). All
             single facts are trained; two-hop questions (country, mother's city, mother's job) and three-hop ones
             (mother's country) are trained for half of the people and tested on the other half (new_*).

Two regimes with the same model, data and budget (Sec. 4.2):
  --arm direct   the problem canvas -> the final canvas with the answer, in one generation
  --arm video    frame by frame: canvas t -> canvas t + 1; at test time each generated frame, raw pixels and all, is
                 the input of the next, so the canvas is the only memory

--per-frame N (Sec. 7): a frame of the demonstrations carries out 1..N steps chosen at random, and rollouts stop at the
first frame with an answer. --detour P: the non-optimal demonstrations of Sec. 7 (a step written again on a new row).

usage: python -m pixel_models.text_tasks check [--out DIR]          (CPU: renders every task and reads the answers back)
       python -m pixel_models.text_tasks train --task expr:4 --arm video --size S --out DIR
       python -m pixel_models.text_tasks freeeval --run DIR [--max-frames 10]
Checkpoints (model, EMA, optimizer, step) every --save-every steps; a restarted run resumes.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import canvas as Q
from . import model as P
from . import names as N


@dataclass
class Ex:
    rows: list[list[str]]       # rows[t]: the rows above the rule in frame t
    answer: str                 # the final answer (only in the last frame)
    prompt: str                 # the text form of the problem
    steps: list[str]            # the text form of the written steps


def right(s: str, cols: int) -> str:
    return s.rjust(cols)


def wrap(s: str, cols: int) -> list[str]:
    return textwrap.wrap(s, cols)


def chained(first: list[str], new: list[str]) -> list[list[str]]:
    """frame 0: the problem; each later frame adds one written row; the answer comes in one more frame"""
    return [first + new[:i] for i in range(len(new) + 1)]


def detour_ex(ex: Ex, rng, p: float) -> Ex:
    """a non-optimal demonstration of Sec. 7 (Appendix H): after each written step, with probability p, one more frame
    writes that same step again on a new row - a legal, visible, useless step, so the solution takes more rows and
    frames. Only for the chained layout (each frame adds one row below the previous ones); other layouts are returned
    as they are. The answer frame is unchanged."""
    rows, first = ex.rows, ex.rows[0]
    n = len(rows) - 1
    if n < 1 or any(len(rows[t]) != len(first) + t or rows[t][:len(rows[t - 1])] != rows[t - 1] for t in range(1, n + 1)):
        return ex
    new, out = rows[n][len(first):], []
    for r in new:
        out.append(r)
        if rng.random() < p:
            out.append(r)
    if len(out) == len(new):
        return ex
    return Ex(chained(first, out), ex.answer, ex.prompt, ex.steps)


def num(rng, l):
    return rng.randint(10 ** (l - 1) if l > 1 else 0, 10 ** l - 1)


def mul_ex(rng, A, B, cols, hard=False) -> Ex:
    a, b = num(rng, A if hard else rng.randint(1, A)), num(rng, B if hard else rng.randint(1, B))
    a, b = max(a, 1), max(b, 1)
    pp = [right(str(a * int(d)) + " " * i, cols) for i, d in enumerate(reversed(str(b)))]
    first = [right(str(a), cols), "x" + right(str(b), cols - 1)]
    return Ex(chained(first, pp), str(a * b), f"{a}x{b}", [str(a * int(d)) + "0" * i for i, d in enumerate(reversed(str(b)))])


def eval_step(toks):
    i = next((j for j, t in enumerate(toks) if t == "*"), None)
    if i is None:
        i = next(j for j, t in enumerate(toks) if t in ("+", "-"))
    a, op, b = toks[i - 1], toks[i], toks[i + 1]
    v = a * b if op == "*" else a + b if op == "+" else a - b
    return toks[:i - 1] + [v] + toks[i + 2:], v


def expr_ex(rng, k, cols, maxv=99999) -> Ex:
    """an expression with k operations, rewritten one operation at a time (leftmost * first, then left to right)"""
    while True:
        toks = [rng.randint(1, 9) if rng.random() < 0.8 else rng.randint(10, 20)]
        for _ in range(k):
            toks += [rng.choice("+-*"), rng.randint(1, 9) if rng.random() < 0.8 else rng.randint(10, 20)]
        show = lambda ts: "".join(str(t) for t in ts)
        rows, cur, ok = [], list(toks), True
        for _ in range(k):
            cur, v = eval_step(cur)
            ok = ok and 0 <= v <= maxv
            rows.append(show(cur))
        if ok and len(show(toks)) <= cols:
            return Ex(chained([show(toks)], rows), rows[-1], show(toks), rows)


# ---------------------------------------------------------------------------------------------------- word problems
NOUNS = "apples coins books pens cards shells stamps marbles candles buttons ribbons cookies plums keys rocks".split()
UP = ["{p} buys {x} more {o}.", "{p} finds {x} more {o}.", "A friend gives {p} {x} {o}."]
DOWN = ["{p} gives away {x} {o}.", "{p} loses {x} {o}.", "{p} sells {x} {o}."]
TIMES = {2: "{p} doubles the number of {o}.", 3: "{p} triples the number of {o}."}
HALF = "{p} gives half of the {o} to a friend."


def wp_ex(rng, k, cols, names) -> Ex:
    p, o, v = rng.choice(names), rng.choice(NOUNS), rng.randint(2, 60)
    sents, eqs = [f"{p} has {v} {o}."], []
    for _ in range(k):
        kinds = ["up", "up"] + (["down"] if v > 1 else []) + (["times"] if v < 200 else []) + (["half"] if v % 2 == 0 and v > 0 else [])
        kind = rng.choice(kinds)
        if kind == "up":
            x = rng.randint(1, 40); sents.append(rng.choice(UP).format(p=p, x=x, o=o)); eqs.append(f"{v}+{x}={v + x}"); v += x
        elif kind == "down":
            x = rng.randint(1, min(v, 40)); sents.append(rng.choice(DOWN).format(p=p, x=x, o=o)); eqs.append(f"{v}-{x}={v - x}"); v -= x
        elif kind == "times":
            m = rng.choice((2, 3)); sents.append(TIMES[m].format(p=p, o=o)); eqs.append(f"{v}*{m}={v * m}"); v *= m
        else:
            sents.append(HALF.format(p=p, o=o)); eqs.append(f"{v}/2={v // 2}"); v //= 2
    text = " ".join(sents) + f" How many {o} does {p} have now?"
    return Ex(chained(wrap(text, cols), eqs), str(v), text, eqs)


# ----------------------------------------------------------------------------------- closed knowledge base, multi-hop
FORMS = {"city": ["Where does {x} live?", "What city does {x} live in?", "{x} lives in which city?", "Name the home city of {x}."],
         "job": ["What does {x} do for a living?", "What is the job of {x}?", "{x} works as what?", "Name the job of {x}."],
         "mother": ["Who is the mother of {x}?", "Whose child is {x}?", "{x} is the child of whom?", "Name the mother of {x}."],
         "country": ["Which country is {x} in?", "What country is {x} located in?", "{x} is in which country?", "Name the country of {x}."]}


class KB:
    """a synthetic knowledge base: names, homes, jobs and mothers drawn at random, cities assigned to countries at random"""

    def __init__(self, n: int, seed: int):
        rng = random.Random(f"kb-{n}-{seed}")
        names = sorted({f"{a} {b}" for a in N.PERSONS for b in N.LAST if len(a) + len(b) < 18})
        self.people = rng.sample(names, n)
        self.cities = rng.sample(N.PLACES, 48)
        self.country = {c: rng.choice(N.COUNTRIES) for c in self.cities}
        self.city = {p: rng.choice(self.cities) for p in self.people}
        self.job = {p: rng.choice(N.JOBS) for p in self.people}
        self.mother = {p: rng.choice([q for q in rng.sample(self.people, 2) if q != p]) for p in self.people}
        # every fact is trained under two of its four question forms; the other two are test-only for that fact
        self.forms = {(k, x): rng.sample(range(4), 4) for k, xs in (("city", self.people), ("job", self.people),
                      ("mother", self.people), ("country", self.cities)) for x in xs}
        shuffled = rng.sample(self.people, n)
        self.hop_train, self.hop_held = shuffled[: n // 2], shuffled[n // 2:]

    def value(self, kind, x) -> str:
        return {"city": self.city, "job": self.job, "mother": self.mother, "country": self.country}[kind][x]


class HopKB(KB):
    """the written steps name each fact as it is recalled; the last row then gives the answer"""

    def one(self, rng, cols, trained=True, kinds=("city", "job", "mother")) -> Ex:
        kind = rng.choice(kinds)
        x = rng.choice(self.cities if kind == "country" else self.people)
        f = self.forms[(kind, x)]
        q = FORMS[kind][rng.choice(f[:2] if trained else f[2:])].format(x=x)
        v = self.value(kind, x)
        step = f"{kind}: {v}"
        return Ex(chained(wrap(q, cols), [step]), v, q, [step])

    def hops(self, rng, people, cols, kind) -> Ex:
        x = rng.choice(people)
        m, form = self.mother[x], {"country": "In which country does {x} live?", "mcity": "Where does the mother of {x} live?",
                                   "mjob": "What is the job of the mother of {x}?",
                                   "mcountry": "In which country does the mother of {x} live?"}[kind]
        q = form.format(x=x)
        if kind == "country":
            steps, v = [f"city: {self.city[x]}", f"country: {self.country[self.city[x]]}"], self.country[self.city[x]]
        elif kind == "mcity":
            steps, v = [f"mother: {m}", f"city: {self.city[m]}"], self.city[m]
        elif kind == "mjob":
            steps, v = [f"mother: {m}", f"job: {self.job[m]}"], self.job[m]
        else:
            steps, v = [f"mother: {m}", f"city: {self.city[m]}", f"country: {self.country[self.city[m]]}"], self.country[self.city[m]]
        return Ex(chained(wrap(q, cols), steps), v, q, steps)

    def train_ex(self, rng, cols, K) -> Ex:
        r = rng.random()
        if r < 0.65:
            return self.one(rng, cols)
        if r < 0.75:
            return self.one(rng, cols, kinds=("country",))
        return self.hops(rng, self.hop_train, cols, rng.choice(["country", "mcity", "mjob"] + (["mcountry"] if K >= 3 else [])))


# ------------------------------------------------------------------------------------------------------- the task list
class Task:
    """training examples and fixed test sets of one task; every test set has one rollout length"""

    def __init__(self, spec: str, px: int, seed: int, kb_people: int = 300):
        self.spec, self.geo = spec, Q.Geo(px)
        c = self.geo.cols
        name, _, arg = spec.partition(":")
        self.name = name
        if name == "mul":
            A, B = map(int, arg.split("x"))
            self.train = lambda r: mul_ex(r, A, B, c)
            self.sets = {f"hard{a}x{b}": (lambda aa, bb: lambda r: mul_ex(r, aa, bb, c, True))(a, b) for a, b in {(A, B), (A, 1), (A // 2 or 1, B // 2 or 1)}}
        elif name == "expr":
            K = int(arg)
            self.train = lambda r: expr_ex(r, r.randint(1, K), c)
            self.sets = {f"k{k}": (lambda kk: lambda r: expr_ex(r, kk, c))(k) for k in (1, 2, 3, 4, 6, 8) if k <= self.geo.q_rows - 1}
        elif name == "wp":
            K = int(arg)
            self.train = lambda r: wp_ex(r, r.randint(1, K), c, [Q.synth_word(r) for _ in range(1)])
            self.sets = {f"new_k{k}": (lambda kk: lambda r: wp_ex(r, kk, c, N.PERSONS_TEST))(k) for k in range(1, K + 1)}
        elif name == "hop":
            K = int(arg)
            kb = HopKB(kb_people, seed)
            self.train = lambda r: kb.train_ex(r, c, K)
            self.sets = {"fact_seen": lambda r: kb.one(r, c), "fact_newform": lambda r: kb.one(r, c, False),
                         "hop2_seen": lambda r: kb.hops(r, kb.hop_train, c, "mcity")}
            for k_ in ("country", "mcity", "mjob") + (("mcountry",) if K >= 3 else ()):
                self.sets[f"new_{k_}"] = (lambda kk: lambda r: kb.hops(r, kb.hop_held, c, kk))(k_)
        else:
            raise ValueError(spec)

    def fits(self, ex: Ex) -> bool:
        g = self.geo
        return len(ex.rows[-1]) <= g.q_rows and all(len(r) <= g.cols for r in ex.rows[-1]) and len(ex.answer) <= Q.ANS_ROWS * g.cols

    detour = 0.0   # --detour P: training demonstrations with redundant rewritten steps (detour_ex); tests never have them

    def sample(self, rng) -> Ex:
        while True:
            ex = self.train(rng)
            if self.fits(ex):
                if self.detour:
                    d = detour_ex(ex, rng, self.detour)
                    if self.fits(d):   # a detour that would leave the canvas is dropped (the demonstration stays optimal)
                        return d
                return ex

    def test_sets(self, n: int) -> dict:
        out = {}
        for k, f in self.sets.items():
            rng = random.Random(f"think-{self.spec}-{k}")
            ex = []
            for _ in range(n * 30):
                e = f(rng)
                if self.fits(e):
                    ex.append(e)
                if len(ex) == n:
                    break
            if ex:
                out[k] = ex
        return out


class MultiTask:
    """several tasks in one model (spec multi:A+B+C, e.g. multi:expr:4+mul:4x4+hop:3; all on the same canvas size): each
    training example comes from a task drawn uniformly, the test sets are every task's, named TASK/SET"""

    def __init__(self, spec: str, px: int, seed: int, kb_people: int = 300):
        self.spec, self.geo, self.name = spec, Q.Geo(px), "multi"
        self.tasks = [Task(p, px, seed, kb_people) for p in spec.split(":", 1)[1].split("+")]
        tag = lambda t: t.spec.replace(":", "")
        self.sets = {f"{tag(t)}/{k}": f for t in self.tasks for k, f in t.sets.items()}
        self._tag = tag

    def fits(self, ex) -> bool:
        return self.tasks[0].fits(ex)

    def train(self, rng):
        return rng.choice(self.tasks).sample(rng)

    def sample(self, rng):
        return self.train(rng)

    def test_sets(self, n: int) -> dict:
        return {f"{self._tag(t)}/{k}": v for t in self.tasks for k, v in t.test_sets(n).items()}


def make_task(spec: str, px: int, seed: int, kb_people: int = 300):
    return MultiTask(spec, px, seed, kb_people) if spec.startswith("multi:") else Task(spec, px, seed, kb_people)

def frame_codes(geo: Q.Geo, rows: list[str], ans: str | None) -> np.ndarray:
    m = np.zeros((geo.rows, geo.cols), np.uint8)
    for r, s in enumerate(rows):
        if s:
            a = np.frombuffer(s.encode("ascii"), np.uint8) - 32
            m[r, :len(a)] = a
    if ans:
        m[geo.q_rows + 1:] = Q.wrap_rows(ans.rjust(geo.cols), Q.ANS_ROWS, geo.cols, False)
    return m


def pair(ex: Ex, arm: str, rng, per_frame: int = 1) -> tuple:
    """-> (rows in, answer in, rows out, answer out) of one training example. Direct: the problem canvas -> the final
    canvas with the answer and no written steps. Frame by frame (video): from state t (t = n: the last written state)
    to state t + j, state n + 1 being the last rows with the answer; j = 1, or uniform in 1..per_frame (frames that
    write up to per_frame steps: the number of frames of an episode is then the model's choice)"""
    n = len(ex.rows) - 1
    if arm == "direct":
        return ex.rows[0], None, ex.rows[0], ex.answer
    t = rng.randint(0, n)
    e = t + (rng.randint(1, per_frame) if per_frame > 1 else 1)
    return (ex.rows[t], None, ex.rows[e], None) if e <= n else (ex.rows[n], None, ex.rows[n], ex.answer)


def train_main(a) -> None:
    import torch
    geo = Q.Geo(a.px)
    P.S = a.px
    torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    a.out.mkdir(parents=True, exist_ok=True)
    task = Task(a.task, a.px, a.seed, a.kb_people)
    task.detour = a.detour
    model, ema = P.build(a.size).to(dev), P.build(a.size).to(dev)
    ema.load_state_dict(model.state_dict()); ema.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=a.wd)
    step, ck = 0, a.out / "ckpt.pt"
    src = ck if ck.exists() else a.init_from      # --init-from: continue another run's training in a new directory
    if src is not None and Path(src).exists():
        s = torch.load(src, map_location=dev)
        model.load_state_dict(s["model"]); ema.load_state_dict(s["ema"]); opt.load_state_dict(s["opt"]); step = s["step"]
        print(f"resumed at step {step} from {src}", flush=True)
    tests = task.test_sets(a.n_test)
    (a.out / "config.json").write_text(json.dumps({"trainer": "text_tasks", **{k: str(v) for k, v in vars(a).items()},
                                                   "tests": {k: len(v) for k, v in tests.items()},
                                                   "params": sum(p.numel() for p in model.parameters())}, indent=1))
    rng = random.Random(a.seed * 1000003 + step)
    G = torch.from_numpy(Q.glyphs()).to(dev)
    GF = G.float()

    def images(m: np.ndarray):
        x = G[torch.from_numpy(m).long().to(dev)].permute(0, 1, 3, 2, 4).reshape(len(m), 1, geo.px, geo.px)
        x[:, :, geo.line_y, 4:geo.px - 4] = 0
        return x.float() / 127.5 - 1

    def batch():
        ex = [task.sample(rng) for _ in range(a.batch)]
        prs = [pair(e, a.arm, rng, a.per_frame) for e in ex]
        return (images(np.stack([frame_codes(geo, p[0], p[1]) for p in prs])),
                images(np.stack([frame_codes(geo, p[2], p[3]) for p in prs])))

    def read_work(arr):
        """(B, q_rows, cols) nearest glyph over the work rows, as strings"""
        x = arr[:, :geo.q_rows * Q.CH]
        cells = x.reshape(len(x), geo.q_rows, Q.CH, geo.cols, Q.CW).permute(0, 1, 3, 2, 4)
        out = []
        for chunk in cells.split(25):
            d = ((chunk[:, :, :, None] - GF[None, None, None]) ** 2).mean((-1, -2))
            out.append(d.argmin(-1).cpu())
        idx = torch.cat(out).numpy()
        return [["".join(chr(32 + c) for c in row).rstrip() for row in card] for card in idx]

    frames_used: list[int] = []

    def rollout(ex_list, seed):
        """the deterministic sampler from the problem canvas; each generated frame, as it is, is the next input. With
        --per-frame 1 the rollout runs for the true number of frames; otherwise it stops at the first frame with an
        answer (the model chooses how many frames it draws)"""
        cur = images(np.stack([frame_codes(geo, e.rows[0], None) for e in ex_list]))
        Ts = [1 if a.arm != "video" else len(e.rows) for e in ex_list]
        free = a.arm == "video" and a.per_frame > 1
        nmax = max(Ts) + 2 if free else max(Ts)
        final = cur.clone()
        done = [False] * len(ex_list)
        g = torch.Generator(device=dev).manual_seed(seed)
        ts = torch.linspace(0, 1, a.sample_steps + 1, device=dev)
        for fr in range(nmax):
            x = torch.randn(cur.shape, device=dev, generator=g)
            for k in range(a.sample_steps):
                xh = ema(cur, x, ts[k].repeat(len(ex_list))).float()
                x = x + (ts[k + 1] - ts[k]) * (xh - x) / (1 - ts[k]).clamp(min=0.05)
            cur = x.float().clamp(-1, 1)
            if free:
                got = Q.read(geo, ((cur + 1) * 127.5)[:, 0].cpu().numpy())
                for i, gi in enumerate(got):
                    if not done[i] and (gi.strip() or fr + 1 == nmax):
                        final[i], done[i] = cur[i], True
                        frames_used.append(fr + 1)
                if all(done):
                    break
            else:
                for i, T in enumerate(Ts):
                    if T == fr + 1:
                        final[i] = cur[i]
        return ((final + 1) * 127.5)[:, 0]

    def evaluate():
        """exact answers on every test set; trace: the written steps right as well"""
        ema.eval(); res = {"step": step, "weights": "ema"}; show = []
        with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            for name, ex in tests.items():
                got, tr = [], 0
                frames_used.clear()
                for i in range(0, len(ex), a.eval_batch):
                    chunk = ex[i:i + a.eval_batch]
                    arr = rollout(chunk, 7 + i)
                    got += Q.read(geo, arr.cpu().numpy())
                    if a.arm != "direct":
                        for e, r in zip(chunk, read_work(arr)):
                            n0 = len(e.rows[0]) if len(e.rows[-1]) > len(e.rows[0]) else 0
                            tr += [x.strip() for x in r[n0:len(e.rows[-1])]] == [x.strip() for x in e.rows[-1][n0:]]
                    if i == 0:
                        show.append(arr[:6].cpu().numpy())
                r = {"exact": sum(g.strip() == e.answer for g, e in zip(got, ex)) / len(ex)}
                if a.arm != "direct":
                    r["trace"] = tr / len(ex)
                if frames_used:
                    r["frames"] = sum(frames_used) / len(frames_used)
                res[name] = {k: round(v, 4) for k, v in r.items()}
        ema.train()
        with (a.out / "evals.jsonl").open("a") as f:
            f.write(json.dumps(res) + "\n")
        print(json.dumps(res), flush=True)
        if show:
            from PIL import Image
            (a.out / "samples").mkdir(exist_ok=True)
            rows = [np.concatenate(list(s), 1) for s in show if len(s) == 6][:4]
            if rows:
                Image.fromarray(np.concatenate(rows, 0).astype(np.uint8)).save(a.out / "samples" / f"s{step:07d}.png")

    model.train(); t0 = time.time(); s0 = step
    while step < a.steps:
        x1, y = batch()
        lr = a.lr * min(1.0, (step + 1) / a.warmup)
        if a.decay_from and step >= a.decay_from:   # linear cool-down to 0 at --steps
            lr *= max(0.0, (a.steps - step) / max(1, a.steps - a.decay_from))
        for g in opt.param_groups:
            g["lr"] = lr
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            if a.arm == "video" and a.ctx_noise:
                x1 = x1 + a.ctx_noise * torch.randn_like(x1)     # the rollout feeds back its own imperfect frames
            t = torch.sigmoid(torch.randn(len(x1), device=dev))
            e = torch.randn_like(y)
            xt = t[:, None, None, None] * y + (1 - t[:, None, None, None]) * e
            xh = model(x1, xt, t).float()
            loss = (((xh - xt) / (1 - t).clamp(min=0.05)[:, None, None, None] - (y - e)) ** 2).mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if torch.isfinite(gnorm):   # a non-finite loss or gradient skips the update
            opt.step()
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.mul_(a.ema).add_(pm.detach(), alpha=1 - a.ema)
        step += 1
        if step % 200 == 0:
            print(json.dumps({"step": step, "loss": round(float(loss), 5), "sec_per_step": round((time.time() - t0) / (step - s0), 4)}), flush=True)
        if step % a.save_every == 0 or step == a.steps:
            torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "step": step}, str(ck) + ".tmp")
            os.replace(str(ck) + ".tmp", ck)
        if step % a.eval_every == 0 or step == a.steps:
            evaluate()
    (a.out / "TRAIN_DONE").write_text(str(step))


CHECK_SPECS = [("expr:4", 192), ("mul:4x4", 192), ("hop:3", 192), ("wp:3", 256)]


def check_main(a) -> None:
    """renders every task, reads every answer back from the true final canvas and checks that the answer box is empty
    before the last frame (CPU, no model)"""
    from PIL import Image
    bad = 0
    tiles = []
    for spec, px in CHECK_SPECS:
        task = Task(spec, px, 0)
        geo = task.geo
        rng = random.Random(1)
        ex = [task.sample(rng) for _ in range(120)]
        sets = task.test_sets(40)
        ex_all = ex + [e for v in sets.values() for e in v]
        card = lambda rows, ans: Q.render(geo, frame_codes(geo, rows, ans))
        got = Q.read(geo, np.stack([card(e.rows[-1], e.answer) for e in ex_all]))
        ok = sum(g.strip() == e.answer for g, e in zip(got, ex_all))
        blank = np.stack([card(e.rows[t], None) for e in ex_all for t in (0, len(e.rows) - 1)])
        okb = sum(g == "" for g in Q.read(geo, blank))
        same = sum(e.rows[t] == e.rows[t + 1] for e in ex_all for t in range(len(e.rows) - 1))
        nrows = max(len(e.rows[-1]) for e in ex_all)
        print(f"{spec:8s} {px}px | answers read back {ok}/{len(ex_all)} | empty answer rows {okb}/{len(blank)} | frames that do not change {same} | rows {nrows}/{geo.q_rows} | sets {list(sets)}", flush=True)
        bad += (ok != len(ex_all)) + (okb != len(blank)) + (same > 0) + (not sets)
        e = sets[next(iter(reversed(list(sets))))][0]
        print("   e.g.", repr(e.prompt[:90]), e.steps[:4], "->", e.answer)
        if px == 192:
            tiles.append(card(e.rows[-1], e.answer))
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.concatenate(tiles, 1)).save(a.out / "text_tasks.png")
    print("ALL OK" if not bad else f"{bad} FAILURES")
    raise SystemExit(1 if bad else 0)


def free_main(a) -> None:
    """Frame-by-frame evaluation without the true frame count. Training evaluates the rollout after exactly len(rows)
    frames; here the rollout runs until the model writes an answer (the answer rows are not blank) or --max-frames,
    and the answer is read from that frame. Both protocols on the run's own test sets and seeds -> RUN/freeeval.json."""
    import torch
    run = a.run
    cfg = json.loads((run / "config.json").read_text())
    px, size, task_spec = int(cfg["px"]), cfg["size"], cfg["task"]
    assert cfg["arm"] == "video", "a frame-by-frame run"
    geo = Q.Geo(px)
    P.S = px
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    task = Task(task_spec, px, int(cfg["seed"]), int(cfg.get("kb_people", 300)))
    tests = task.test_sets(int(cfg["n_test"]))
    ema = P.build(size).to(dev)
    ema.load_state_dict(torch.load(run / "ckpt.pt", map_location=dev)["ema"]); ema.eval()
    G = torch.from_numpy(Q.glyphs()).to(dev)
    steps = a.sample_steps or int(cfg["sample_steps"])

    def images(m):
        x = G[torch.from_numpy(m).long().to(dev)].permute(0, 1, 3, 2, 4).reshape(len(m), 1, geo.px, geo.px)
        x[:, :, geo.line_y, 4:geo.px - 4] = 0
        return x.float() / 127.5 - 1

    res = {"run": str(run), "max_frames": a.max_frames, "sample_steps": steps}
    with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
        for name, ex in tests.items():
            oracle, free, used, true_n = [], [], [], []
            eb = int(cfg["eval_batch"])
            for i in range(0, len(ex), eb):
                chunk = ex[i:i + eb]
                cur = images(np.stack([frame_codes(geo, e.rows[0], None) for e in chunk]))
                Ts = [len(e.rows) for e in chunk]
                g = torch.Generator(device=dev).manual_seed(7 + i)       # the training run's evaluation seeds
                ts = torch.linspace(0, 1, steps + 1, device=dev)
                o_ans = [None] * len(chunk); f_ans = [None] * len(chunk); f_n = [a.max_frames + 1] * len(chunk)
                for fr in range(max(a.max_frames, max(Ts))):
                    x = torch.randn(cur.shape, device=dev, generator=g)
                    for k in range(steps):
                        xh = ema(cur, x, ts[k].repeat(len(chunk))).float()
                        x = x + (ts[k + 1] - ts[k]) * (xh - x) / (1 - ts[k]).clamp(min=0.05)
                    cur = x.float().clamp(-1, 1)
                    got = Q.read(geo, ((cur + 1) * 127.5)[:, 0].cpu().numpy())
                    for j, gj in enumerate(got):
                        if fr + 1 == Ts[j]:
                            o_ans[j] = gj.strip()
                        if f_ans[j] is None and gj.strip() and fr < a.max_frames:
                            f_ans[j], f_n[j] = gj.strip(), fr + 1
                    if all(v is not None for v in o_ans) and (all(v is not None for v in f_ans) or fr + 1 >= a.max_frames):
                        break
                oracle += [o == e.answer for o, e in zip(o_ans, chunk)]
                free += [f == e.answer for f, e in zip(f_ans, chunk)]
                used += f_n; true_n += Ts
            n = len(ex)
            res[name] = {"oracle": round(sum(oracle) / n, 4), "free": round(sum(free) / n, 4),
                         "stop_at_true": round(sum(u == t for u, t in zip(used, true_n)) / n, 4),
                         "early": round(sum(u < t for u, t in zip(used, true_n)) / n, 4),
                         "late": round(sum(t < u <= a.max_frames for u, t in zip(used, true_n)) / n, 4),
                         "never": round(sum(u > a.max_frames for u in used) / n, 4),
                         "mean_frames": round(sum(min(u, a.max_frames) for u in used) / n, 2), "true_frames": round(sum(true_n) / n, 2)}
            print(name, json.dumps(res[name]), flush=True)
    (run / ("freeeval.json" if not a.sample_steps else f"freeeval_steps{steps}.json")).write_text(json.dumps(res, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check"); c.add_argument("--out", type=Path)
    t = sub.add_parser("train")
    t.add_argument("--task", required=True, help="expr:4 | mul:4x4 | wp:3 | hop:3")
    t.add_argument("--arm", choices=("direct", "video"), required=True, help="direct answering, or frame by frame")
    t.add_argument("--size", choices=sorted(P.SIZES), default="S")
    t.add_argument("--px", type=int, default=192)
    t.add_argument("--steps", type=int, default=40000)
    t.add_argument("--batch", type=int, default=256)
    t.add_argument("--eval-batch", type=int, default=64)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--wd", type=float, default=0.0)
    t.add_argument("--warmup", type=int, default=1000)
    t.add_argument("--ema", type=float, default=0.999)
    t.add_argument("--ctx-noise", type=float, default=0.03, help="frame by frame: Gaussian noise on the input canvas")
    t.add_argument("--kb-people", type=int, default=300)
    t.add_argument("--sample-steps", type=int, default=30)
    t.add_argument("--n-test", type=int, default=128)
    t.add_argument("--eval-every", type=int, default=2000)
    t.add_argument("--save-every", type=int, default=1000)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--per-frame", type=int, default=1, help="frame by frame: a frame writes 1..N steps (uniform); test: free-running")
    t.add_argument("--detour", type=float, default=0.0, help="non-optimal demonstrations: after each step, with this chance, the step written again")
    t.add_argument("--init-from", type=Path, help="ckpt.pt of another run to continue from (model, EMA, optimizer and step)")
    t.add_argument("--decay-from", type=int, default=0, help="linear lr cool-down from this step to 0 at --steps")
    t.add_argument("--out", type=Path, required=True)
    f = sub.add_parser("freeeval", help="frame by frame: stop when the model writes an answer, not after the true number of frames")
    f.add_argument("--run", type=Path, required=True)
    f.add_argument("--max-frames", type=int, default=16)
    f.add_argument("--sample-steps", type=int, default=0, help="denoising steps per frame (default: the run's own)")
    a = ap.parse_args()
    {"check": check_main, "train": train_main, "freeeval": free_main}[a.cmd](a)


if __name__ == "__main__":
    main()
