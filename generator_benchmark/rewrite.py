"""Official prompt rewriters on the benchmark canvases (the Rewrite condition) and the answer deletion (Delete).

Closed systems and FLUX.3 put a language model in front of the generator that sees the canvas and writes the full
prompt. Here each open generator gets its own family's official rewriter (system prompt and, where it is open, model),
so we can measure what the rewriter alone adds; Delete removes the answer from the rewritten prompt and keeps
everything else (rule-based here; delete_answer.py then re-checks every deleted prompt with an LLM).

rewriters (--rewriter):
  pup        FLUX.3 official upsampler: flux-3-image-klein-pup (Qwen3.5-9B), its i2i system prompt and scaffold,
             greedy, thinking off (diffusers' own upsample_prompt). Local GPU. --backend self: the same system prompt
             and scaffold, decoded by the generator's own VLM text encoder (flux3_klein_base: Qwen3-VL-4B).
  qwen_edit  Qwen-Image official edit enhancer (polish_edit_prompt: EDIT_SYSTEM_PROMPT + "User Input ... Rewritten
             Prompt:", JSON {"Rewritten": ...}); official model qwen-vl-max. --backend api uses the REWRITER endpoint
             (qwen3.8-max); --backend self uses the generator's own VLM text encoder (qwen_image_21,
             longcat_image_edit).
  wan_i2v    Wan2.2 official i2v prompt extension (English VL system prompt); --backend api uses the REWRITER endpoint.
             Shared by Wan2.2 and OmniWeaving (--request video).
  pup_quote  --backend self only: pup's prompt plus LongCat's model-card rule (only text in quotes is encoded character
             by character, so new text is quoted and nothing else is). Used for LongCat-Image-Edit.
  h3         MiniMax-H3: the hosted rewriter is not released; the REWRITER model gets the official prompt writing guide
             (rewrite_prompts/h3.txt) as its system prompt.
The request is always the item-independent instruction (tasks.GENERIC_PROMPT or video_generate.VIDEO_PROMPT).
API backend: REWRITER_BASE_URL (OpenAI-compatible server) and REWRITER_API_KEY.

Output (one row per item): item_id, task, level, target, rewriter, backend, raw, prompt, has_answer, prompt_noans,
n_removed. prompt_noans deletes every whole-word, case-insensitive occurrence of the answer (with its quotes when it
is quoted on its own), spelled-out and number-word forms included; it is only produced where the answer is not printed
on the canvas (Factual and Arithmetic).
usage: python -m generator_benchmark.rewrite --rewriter pup --set t1 --out rewrites/pup.jsonl [--shard 0/2]
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path

from . import render, tasks
from .paths import model as model_path, src as src_path

ABLATE = ("qa", "arithmetic")
LEVELS60 = {"qa": ("L1", "L2R", "L3", "L4", "L2F"), "context": ("L5A", "L5B"), "arithmetic": ("A1", "A2"),
            "instruction": ("I1", "I2"), "visual": ("L6A", "L6I", "L6B"), "copy_complete": ("copy",)}
# (task, level, n[, start]): items [start:n] of a test level in frozen order
SETS = {
    # Table 1: the first 60 items of each of the seven levels (Factual = qa L1 + L4, Arithmetic A1, Instruction I1,
    # Context L5A, Visual L6A, Copy)
    "t1": [("qa", "L1", 60), ("qa", "L4", 60), ("arithmetic", "A1", 60), ("instruction", "I1", 60), ("context", "L5A", 60),
           ("visual", "L6A", 60), ("copy_complete", "copy", 60)],
    # 60 items of every one of the 15 test levels (the closed-model reference set)
    "closed60": [(t, lv, 60) for t, lvs in LEVELS60.items() for lv in lvs],
    # 60 video canvases whose answer is not on the canvas (a subset of video_generate's main set)
    "key60": [("qa", "L1", 30), ("qa", "L4", 10), ("arithmetic", "A1", 20)],
    "smoke": [("qa", "L1", 2), ("arithmetic", "A1", 1)],
    "t1rest": None, "t1rest_noans": None,   # set below (T1REST)
}
ABLATE_TASKS = ABLATE
# The rest of every Table 1 level beyond its first 60 items (t1 + t1rest = the full levels); t1rest_noans = its Factual
# and Arithmetic levels, the ones Delete applies to. (task, level, n, start): items [start:n]
T1REST = [("qa", "L1", 10 ** 6, 60), ("qa", "L4", 10 ** 6, 60), ("arithmetic", "A1", 10 ** 6, 60),
          ("instruction", "I1", 10 ** 6, 60), ("context", "L5A", 10 ** 6, 60), ("visual", "L6A", 10 ** 6, 60),
          ("copy_complete", "copy", 10 ** 6, 60)]
T1REST_NOANS = T1REST[:3]
SETS["t1rest"], SETS["t1rest_noans"] = T1REST, T1REST_NOANS
# LongCat-Image(-Edit) model card: target text must be in quotes to trigger its character-level encoding. No hint at the answer.
QUOTE_RULE = ("\n\nThis editing model only draws text that is enclosed in double quotes. Enclose every piece of text that "
              "must newly appear in the image in double quotes, and do not put any other text (existing wording, "
              "questions, labels) in quotes.")

# Qwen-Image src/examples/tools/prompt_utils.py, polish_edit_prompt (verbatim)
QWEN_EDIT_SYSTEM = (Path(__file__).with_name("rewrite_prompts") / "qwen_edit.txt")


def select(name: str) -> list[dict]:
    data = tasks.build_all()
    items = []
    for task, level, n, *start in SETS[name]:
        items += [i for i in data[task]["test"] if i["level"] == level][(start[0] if start else 0):n]
    return items


FILLER = r"(?:then|and|next|followed by|finally|after that|appears?|is written|written|comes?|first|second|last)"
SEP = rf"[^\w]{{0,6}}(?:{FILLER}[^\w]{{0,6}}){{0,3}}"


def fold(text: str) -> str:
    """lower case without accents ("Asunción" -> "asuncion")"""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()


# rewriters add accents the target lacks ("Asunción" for Asuncion): every letter also matches its accented forms
ACCENTED: dict[str, str] = {}
for _cp in range(0xC0, 0x250):
    _ch = chr(_cp)
    if _ch.isalpha() and fold(_ch) and len(fold(_ch)) == 1:
        ACCENTED[fold(_ch)] = ACCENTED.get(fold(_ch), "") + _ch


def letter(c: str) -> str:
    """regex for one answer character: itself, its accented forms (precomposed, or followed by combining marks)"""
    if not c.isalpha():
        return re.escape(c)
    return "[" + re.escape(c.lower() + c.upper() + ACCENTED.get(fold(c), "")) + "]" + "[\u0300-\u036f]*"


def answer_regex(answer: str) -> str:
    return "".join(letter(c) for c in answer.strip())


def spelled(answer: str) -> re.Pattern | None:
    """The answer spelled out one character at a time ("M, then a, d, r, i, d"; first "2" appears, then "9")."""
    chars = [c for c in answer if c.isalnum()]
    if len(chars) < 2:
        return None
    body = SEP.join(rf"(?<![\w]){letter(c)}(?![\w])" for c in chars)
    return re.compile(body, re.IGNORECASE)


# H3-guide rewrites spell the answer with a description between the characters ('the digit "2" is written
# stroke by stroke ...; a moment later the digit "9"', '"M" forms first, followed in order by "a", "d", ...'): a run of
# quoted single characters that spells the answer is deleted wherever its pieces are. Double quotes only (an apostrophe
# would make "card's 'x'" look like a quoted character). Tags such as <Picture 1> / [Shot 1] are never touched.
QUOTED_CHAR = re.compile(r"[\"“”]\s*([^\W_][\u0300-\u036f]*)\s*[\"“”]")
TAG = re.compile(r"<[^<>\n]{1,40}>|\[[^\[\]\n]{1,40}\]")
ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def number_words(n: int) -> list[str]:
    """English spellings of 0 <= n < 1e6 ('twenty-nine', 'twenty nine', 'one hundred (and) five')."""
    def below100(k):
        return [ONES[k]] if k < 20 else [TENS[k // 10]] if k % 10 == 0 else [f"{TENS[k // 10]}-{ONES[k % 10]}", f"{TENS[k // 10]} {ONES[k % 10]}"]

    def below1000(k):
        if k < 100:
            return below100(k)
        head = f"{ONES[k // 100]} hundred"
        return [head] if k % 100 == 0 else [f"{head}{j}{r}" for j in (" ", " and ") for r in below100(k % 100)]
    if not 0 <= n < 10 ** 6:
        return []
    if n < 1000:
        return below1000(n)
    head = [f"{h} thousand" for h in below1000(n // 1000)]
    return head if n % 1000 == 0 else [f"{h}{j}{r}" for h in head for j in (" ", " and ", ", ") for r in below1000(n % 1000)]


def quoted_run(text: str, answer: str) -> list[tuple[int, int]]:
    """spans of consecutive quoted single characters (in the order they appear) that spell the whole answer"""
    chars = [fold(c) for c in answer if c.isalnum()]
    if len(chars) < 2:
        return []
    toks = [(m.start(), m.end(), fold(m.group(1))) for m in QUOTED_CHAR.finditer(text)]
    spans, i = [], 0
    while i + len(chars) <= len(toks):
        if [t[2] for t in toks[i:i + len(chars)]] == chars:
            spans += [(s, e) for s, e, _ in toks[i:i + len(chars)]]
            i += len(chars)
        else:
            i += 1
    return spans


def _mark(k: int) -> str:
    return "\x00TAG" + "".join(chr(97 + int(c)) for c in str(k)) + "\x00"   # letters only: no digit to match


def _protect(text: str):
    """replace tags by placeholders (restored by _restore) so that no deletion touches them"""
    tags = TAG.findall(text)
    for k, t in enumerate(tags):
        text = text.replace(t, _mark(k), 1)
    return text, tags


def _restore(text: str, tags: list[str]) -> str:
    for k, t in enumerate(tags):
        text = text.replace(_mark(k), t, 1)
    return text


def strip_answer(text: str, answer: str) -> tuple[str, int]:
    """Delete the answer from a rewritten prompt, everything else unchanged: a quoted answer loses its quotes too, an
    answer spelled out character by character is deleted as well (also as a run of quoted characters with words in
    between), and so is a number answer written in words. Empty quotes and doubled separators left behind are tidied."""
    if not answer.strip():
        return text, 0
    out, tags = _protect(text)
    a = answer_regex(answer)
    pattern = re.compile(rf"[\"“'‘]\s*{a}\s*[\"”'’]|(?<![\w]){a}(?![\w])", re.IGNORECASE)
    out, n = pattern.subn("", out)
    sp = spelled(answer)
    if sp is not None:
        out, m = sp.subn("", out)
        n += m
    spans = quoted_run(out, answer)
    for s, e in sorted(spans, reverse=True):
        out = out[:s] + out[e:]
    n += len(spans)
    if answer.strip().isdigit():
        for w in sorted(number_words(int(answer.strip())), key=len, reverse=True):
            out, m = re.subn(rf"[\"“]?(?<![\w-]){re.escape(w)}(?![\w-])[\"”]?", "", out, flags=re.IGNORECASE)
            n += m
    out = re.sub(r"[\"“”]\s*[\"“”]", "", out)            # empty quotes
    out = re.sub(r"(?:\s*,){2,}", ",", out)               # ", , ," left by deleted characters
    out = re.sub(r"[ \t]{2,}", " ", out)
    out = re.sub(r" ([,.;:])", r"\1", out)
    out = re.sub(r",([.;:])", r"\1", out)
    return _restore(out, tags).strip(), n


def answer_leaks(text: str, answer: str) -> list[str]:
    """every way the answer is still in the text: word, spelled (short separators), quoted run, number words; 'digits'
    flags the answer's digits as standalone tokens in order within 400 characters (to be read by eye)"""
    if not answer.strip():
        return []
    body, _ = _protect(text)
    out = []
    if re.search(rf"(?<![\w]){answer_regex(answer)}(?![\w])", body, re.IGNORECASE):
        out.append("word")
    sp = spelled(answer)
    if sp is not None and sp.search(body):
        out.append("spelled")
    if quoted_run(body, answer):
        out.append("quoted_run")
    if answer.strip().isdigit():
        if any(re.search(rf"(?<![\w-]){re.escape(w)}(?![\w-])", body, re.IGNORECASE) for w in number_words(int(answer.strip()))):
            out.append("number_words")
        digits = answer.strip()
        if len(digits) >= 2 and re.search(r".{0,400}?".join(rf"(?<![\w.]){d}(?![\w.])" for d in digits), body, re.S):
            out.append("digits")
    return out


def has_answer(text: str, answer: str) -> bool:
    return bool(set(answer_leaks(text, answer)) - {"digits"})


def data_url(image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


class ApiChat:
    """The rewriter model (qwen3.8-max, medium reasoning, streaming) on an OpenAI-compatible endpoint
    (REWRITER_BASE_URL, REWRITER_API_KEY), retried on technical errors."""

    def __init__(self, base_url=None, key_env="REWRITER_API_KEY", model="qwen3.8-max"):
        from .judge.judge import _http_post_stream
        base_url = base_url or os.environ["REWRITER_BASE_URL"]
        self.post, self.url, self.model = _http_post_stream, base_url.rstrip("/") + "/v1/chat/completions", model
        self.headers = {"Content-Type": "application/json", "Authorization": f"Bearer {os.environ[key_env]}"}

    def __call__(self, system: str | None, image, text: str) -> str:
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": data_url(image)}},
                                         {"type": "text", "text": text}]}]
        body = json.dumps({"model": self.model, "messages": messages, "max_tokens": 16384,
                           "reasoning_effort": "medium"}).encode()
        for attempt in range(8):
            try:
                status, raw = self.post(self.url, body, self.headers, 900)
                if status == 200:
                    reply = json.loads(raw)["choices"][0]["message"]["content"]
                    if reply and reply.strip():
                        return reply.strip()
                print(f"api status {status}: {raw[:200]!r}", flush=True)
            except Exception as exc:  # noqa: BLE001 - network errors are retried
                print(f"api error {exc!r}", flush=True)
            time.sleep(min(60, 5 * 2 ** attempt))
        raise RuntimeError("rewriter API failed 8 times")


class LocalVLM:
    """A generator's own VLM text encoder used as a chat model (greedy)."""

    def __init__(self, model: str):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        from .models import ADAPTERS, DTYPE
        path = ADAPTERS[model].path
        self.torch = torch
        self.model = AutoModelForImageTextToText.from_pretrained(path + "/text_encoder", dtype=DTYPE).to("cuda").eval()
        self.processor = AutoProcessor.from_pretrained({"qwen_image_21": path + "/processor",
                                                        "flux3_klein_base": path + "/text_encoder",
                                                        "longcat_image_edit": path + "/text_processor"}[model])

    def __call__(self, system: str | None, image, text: str) -> str:
        messages = ([{"role": "system", "content": [{"type": "text", "text": system}]}] if system else []) + [
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": text}]}]
        inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True,
                                                    return_tensors="pt", enable_thinking=False).to("cuda")
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=1024, do_sample=False)
        return self.processor.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


class Pup:
    def __init__(self):
        import torch
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
        from diffusers.modular_pipelines.flux3.prompt_upsampler import prepare_references, upsample_prompt
        path = model_path("flux-3-image-klein-pup")
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(path, dtype=torch.bfloat16).to("cuda").eval()
        self.processor = AutoProcessor.from_pretrained(path)
        self.prepare, self.upsample = prepare_references, upsample_prompt

    def __call__(self, image, request: str) -> str:
        return self.upsample(self.model, self.processor, request, images=self.prepare(image))


def qwen_edit(chat, image, request: str) -> str:
    """polish_edit_prompt: the whole enhancer prompt goes in the user turn; the system turn is the generic one."""
    text = f"{QWEN_EDIT_SYSTEM.read_text()}\n\nUser Input: {request}\n\nRewritten Prompt:"
    raw = chat("you are a helpful assistant, you should provide useful answers to users.", image, text)
    return raw


def parse_qwen_edit(raw: str) -> str:
    body = raw.replace("```json", "").replace("```", "")
    try:
        return json.loads(body)["Rewritten"].strip().replace("\n", " ")
    except Exception:  # noqa: BLE001 - keep the model's text when it ignores the JSON format
        m = re.search(r'"Rewritten"\s*:\s*"(.*)"', body, re.S)
        return (m.group(1) if m else body).strip().replace("\n", " ")


def wan_i2v_system() -> str:
    """DEFAULT_SYS_PROMPTS["ti2v-5B"]["i2v"]["en"] of wan/utils/prompt_extend.py, read from wan/utils/system_prompt.py
    without importing the wan package (its __init__ needs the model dependencies)."""
    ns: dict = {}
    exec((Path(src_path("Wan2.2")) / "wan/utils/system_prompt.py").read_text(), ns)
    return ns["I2V_A14B_EN_SYS_PROMPT"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rewriter", choices=("pup", "pup_quote", "qwen_edit", "wan_i2v", "h3"), required=True)
    ap.add_argument("--backend", choices=("local", "api", "self"), default="local")
    ap.add_argument("--self-model", default="qwen_image_21")
    ap.add_argument("--set", choices=sorted(SETS), default="t1")
    ap.add_argument("--request", choices=("image", "video"), default="image")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--workers", type=int, default=1, help="parallel API calls (api backend)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    k, n = map(int, a.shard.split("/"))
    items = select(a.set)[k::n]
    done = set()
    if a.out.exists():
        done = {json.loads(l)["item_id"] for l in a.out.read_text().splitlines() if l.strip()}
    todo = [i for i in items if i["item_id"] not in done]
    print(f"{len(items)} items, {len(todo)} to do", flush=True)
    if not todo:
        return
    if a.request == "video":
        from .video_generate import VIDEO_PROMPT as request
    else:
        request = tasks.GENERIC_PROMPT
    if a.rewriter == "pup_quote" and a.backend != "self":
        raise SystemExit("pup_quote needs --backend self")
    if a.rewriter in ("pup", "pup_quote") and a.backend == "self":  # PUP's own system prompt and scaffold, the generator's encoder decodes
        from diffusers.modular_pipelines.flux3.prompt_upsampler import I2I_SCAFFOLD, SYSTEM_PROMPTS
        chat = LocalVLM(a.self_model)
        system = SYSTEM_PROMPTS["i2i"] + (QUOTE_RULE if a.rewriter == "pup_quote" else "")
        one = lambda img: (chat(system, img, I2I_SCAFFOLD[False].format(instruction=request)), None)  # noqa: E731
    elif a.rewriter == "pup":
        pup = Pup()
        one = lambda img: (pup(img, request), None)  # noqa: E731
    else:
        chat = ApiChat() if a.backend == "api" else LocalVLM(a.self_model)
        if a.rewriter == "qwen_edit":
            one = lambda img: (parse_qwen_edit(raw := qwen_edit(chat, img, request)), raw)  # noqa: E731
        elif a.rewriter == "wan_i2v":
            system = wan_i2v_system()
            one = lambda img: (chat(system, img, request), None)  # noqa: E731
        else:
            system = (Path(__file__).with_name("rewrite_prompts") / "h3.txt").read_text()
            one = lambda img: (chat(system, img, request), None)  # noqa: E731
    a.out.parent.mkdir(parents=True, exist_ok=True)

    def row(item):
        prompt, raw = one(render.render_input(item))
        noans, removed = strip_answer(prompt, item["target"])
        return {"item_id": item["item_id"], "task": item["task"], "level": item["level"], "target": item["target"],
                "rewriter": a.rewriter, "backend": a.backend, "request": request,
                "raw": raw, "prompt": prompt, "has_answer": has_answer(prompt, item["target"]),
                "prompt_noans": noans if item["task"] in ABLATE_TASKS else None, "n_removed": removed,
                "noans_has_answer": has_answer(noans, item["target"]), "noans_leaks": answer_leaks(noans, item["target"])}

    with a.out.open("a") as f:
        if a.backend == "api" and a.workers > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(a.workers) as pool:
                for m, r in enumerate(pool.map(row, todo), start=1):
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    f.flush()
                    print(m, r["item_id"], r["has_answer"], flush=True)
        else:
            for m, item in enumerate(todo, start=1):
                r = row(item)
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
                print(m, r["item_id"], r["has_answer"], flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
