"""The Delete condition: remove every trace of the answer from a rewritten prompt, keep everything else.

rewrite.strip_answer deletes the answer string, its spelled-out and number-word forms and runs of quoted characters.
Rewriters also leak the answer in other ways (accented or partial spellings, fragments such as ", D.C.", letter-by-letter
or stroke-by-stroke descriptions), so the rule-based deletion is followed by an LLM pass (qwen3.8-max on the
REWRITER endpoint: REWRITER_BASE_URL, REWRITER_API_KEY):
  --mode recheck  (image editing rewrites) the LLM judges whether prompt_noans still reveals the answer; a flagged
                  prompt is re-deleted by the LLM from scratch and re-judged, up to 4 tries.
  --mode rewrite  (video rewrites, whose descriptions of the writing are long) the LLM deletes the answer from the
                  rewritten prompt directly; the result is checked by rules (rewrite.answer_leaks, no quoted answer
                  character, no answer prefix of two or more characters) and retried, up to 4 tries.
A rewrite that is nothing but the answer becomes the neutral "the answer" (quoted if the rewrite was quoted).
Only rows with a prompt_noans (Factual and Arithmetic) are touched; every other field is kept.
usage: python -m generator_benchmark.delete_answer --mode recheck --rewrites rewrites/pup.jsonl --out rewrites/pup_delete.jsonl
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import time
from pathlib import Path

from .judge.judge import _http_post_stream
from .rewrite import answer_leaks

JUDGE = ("You check prompts for an image editing model. The model must write the answer to a question card WITHOUT being told "
         "the answer. Does the prompt below reveal the answer, fully or partly? Count as revealing: the answer itself; any spelling, "
         "accented or unaccented form, transliteration or translation of it; its digits or a number word for it; any of its "
         "characters quoted or named; a prefix or fragment of it; a description of the shapes or strokes of its characters. "
         "Text that is printed on the card anyway (the question itself) does not count. Reply with exactly YES or NO.")
DELETE_IMAGE = ("You edit image-editing prompts. The prompt describes how to write an answer into the answer box of a question card. "
                "Remove EVERY piece of information that reveals the answer: the answer itself, any of its characters (quoted or not), "
                "accented or alternative spellings, number words or digits of it, prefixes or fragments of it, and any description of "
                "the shapes or strokes of specific characters. Replace such passages with neutral wording such as \"the answer\" so "
                "the sentence stays grammatical. Change NOTHING else. Output only the edited prompt.")
DELETE_VIDEO = ("You edit video-generation prompts. The prompt describes a video in which an answer is written into an answer box "
                "on a question card. Remove EVERY piece of information that reveals the answer: the answer itself, any of its "
                "characters (quoted or not), prefixes or fragments of it, the number or kind of characters if that narrows it "
                "down, and any description of the shapes or strokes of specific characters (e.g. 'the 4 forms from a diagonal', "
                "'a capital Z followed by a lowercase n'). Replace such passages with neutral wording such as 'the answer' or "
                "'the characters of the answer' so the sentence stays grammatical. Change NOTHING else: keep the structure, the "
                "JSON format and every other word exactly as it is. Output only the edited prompt.")
TRIES = 4


class Chat:
    def __init__(self, model: str):
        self.url = os.environ["REWRITER_BASE_URL"].rstrip("/") + "/v1/chat/completions"
        self.headers = {"Content-Type": "application/json",
                        "Authorization": f"Bearer {os.environ['REWRITER_API_KEY']}"}
        self.model = model

    def __call__(self, system: str, text: str, effort: str = "medium") -> str:
        body = json.dumps({"model": self.model, "reasoning_effort": effort, "max_tokens": 16384,
                           "messages": [{"role": "system", "content": system},
                                        {"role": "user", "content": text}]}).encode()
        for attempt in range(8):
            try:
                status, raw = _http_post_stream(self.url, body, self.headers, 900)
                if status == 200:
                    reply = json.loads(raw)["choices"][0]["message"]["content"]
                    if reply and reply.strip():
                        return reply.strip()
                print(f"api status {status}", flush=True)
            except Exception as exc:  # noqa: BLE001 - network errors are retried
                print(f"api error {type(exc).__name__}", flush=True)
            time.sleep(min(60, 5 * 2 ** attempt))
        raise RuntimeError("rewriter API failed 8 times")


def unfence(text: str) -> str:
    return re.sub(r"^```(?:json|\w*)?\s*|\s*```$", "", text.strip())


def rule_leaks(prompt: str, answer: str) -> list[str]:
    out = list(answer_leaks(prompt, answer))
    a = answer.strip()
    if any(re.search(r"[\"'‘’“”]" + re.escape(c) + r"[\"'‘’“”]", prompt, 0 if c.isalpha() else re.I)
           for c in set(a) if c.strip()):
        out.append("quoted_char")
    for k in range(2, len(a)):
        if re.search(r"(?<![A-Za-z0-9])" + re.escape(a[:k]) + r"(?![a-z])", prompt):
            out.append("prefix")
            break
    return out


def bare_answer(prompt: str) -> bool:
    """the deletion left nothing but quotes and punctuation"""
    return not re.sub(r"[\W_]+", "", prompt)


def placeholder(prompt: str) -> str:
    return '"the answer"' if prompt.strip()[:1] in "\"“" else "the answer"


def recheck(chat: Chat, row: dict) -> dict:
    answer, prompt = str(row["target"]), row["prompt_noans"]
    if bare_answer(prompt):
        return {**row, "prompt_noans": placeholder(row["prompt"]), "noans_method": "bare_answer_placeholder"}

    def reveals(p: str) -> bool:
        return chat(JUDGE, f"Answer (hidden from the model): {answer}\n\nPrompt:\n{p}", "low").upper().startswith("YES")
    if not reveals(prompt):
        return row
    new = cand = None
    for _ in range(TRIES):
        cand = unfence(chat(DELETE_IMAGE, f"Answer to remove: {answer}\n\nPrompt:\n{prompt}"))
        if not reveals(cand):
            new = cand
            break
    return {**row, "prompt_noans": new or cand, "noans_method": "llm_recheck", "noans_still_flagged": new is None,
            "prompt_noans_before": prompt}


def redelete(chat: Chat, row: dict) -> dict:
    answer, source = str(row["target"]), row["prompt"]
    last = None
    for _ in range(TRIES):
        cand = unfence(chat(DELETE_VIDEO, f"Answer to remove: {answer}\n\nPrompt:\n{source}"))
        leaks = rule_leaks(cand, answer)
        last = (cand, leaks)
        if not leaks:
            return {**row, "prompt_noans": cand, "noans_method": "llm_delete"}
    return {**row, "prompt_noans": last[0], "noans_method": "llm_delete", "noans_leaks": last[1]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("recheck", "rewrite"), required=True)
    ap.add_argument("--rewrites", type=Path, nargs="+", required=True, help="rewrite.py output rows")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="qwen3.8-max")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    chat = Chat(a.model)
    rows = [json.loads(line) for f in a.rewrites for line in f.read_text().splitlines() if line.strip()]
    one = recheck if a.mode == "recheck" else redelete

    def fix(row: dict) -> dict:
        return one(chat, row) if row.get("prompt_noans") is not None else row
    with cf.ThreadPoolExecutor(a.workers) as ex:
        out = list(ex.map(fix, rows))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    changed = sum(r.get("noans_method") is not None for r in out)
    left = sum(bool(r.get("noans_still_flagged") or r.get("noans_leaks")) for r in out)
    print(json.dumps({"rows": len(out), "noans_rows": sum(r.get("prompt_noans") is not None for r in out),
                      "changed": changed, "still_flagged": left}), flush=True)


if __name__ == "__main__":
    main()
