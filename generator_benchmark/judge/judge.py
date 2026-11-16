"""Judge transport for the benchmark evaluator.

Endpoint and key come from the environment: JUDGE_BASE_URL (an OpenAI-compatible server serving the judge model) and
JUDGE_API_KEY (the bearer token). The key is only placed in the Authorization header and is never logged or cached.

One logical judge request = system prompt + ordered user parts (text and PNG images) sent to an OpenAI-compatible
/v1/chat/completions endpoint (qwen3.8-max, reasoning_effort=high). The reply's message.content must hold one JSON
object (an optional ```json fence is accepted); reasoning_content is stored but never parsed.

Outcome of a request is exactly one of
  ok        - content parsed and accepted by the role validator
  invalid   - the judge answered but the content stayed unparseable / schema-invalid after max_invalid_attempts
              (a judge-side failure; scored as Unknown, never as a miss)
  technical - transport error, HTTP 408/429/5xx, error body, empty content (e.g. reasoning used the whole token
              budget), or a resolved model that is not the requested one, after max_technical_attempts
Re-sends are blind to the private answer: a reply is never re-sent because of what it says, only because it could
not be used. Every attempt is kept in the record.

Records are content-addressed (sha256 of the canonical request incl. model, parameters, prompt and image hashes)
under cache_dir. ok/invalid records are final and replayed; technical records are retried on the next run.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

SCHEMA = "umm-eval-v3/judge-record/1"
FENCE = re.compile(r"^```(?:json)?\s*(\{.*\})\s*```$", re.DOTALL | re.IGNORECASE)


class InvalidReply(ValueError):
    """Raised by role validators (and the JSON parser) for an unusable judge reply."""


@dataclass(frozen=True)
class JudgeConfig:
    model: str = "qwen3.8-max"
    base_url: str = os.environ.get("JUDGE_BASE_URL", "")
    api_key_env: str = "JUDGE_API_KEY"
    reasoning_effort: str | None = "high"
    max_tokens: int = 16384
    timeout_seconds: int = 900
    max_technical_attempts: int = 6
    max_invalid_attempts: int = 3
    # A reply counts only if its "model" field is one of these (default: exactly the requested name).
    accepted_models: tuple[str, ...] = ()
    stream: bool = True

    def accepted(self) -> tuple[str, ...]:
        return self.accepted_models or (self.model,)

    def request_parameters(self) -> dict[str, Any]:
        params: dict[str, Any] = {"max_tokens": self.max_tokens}
        if self.reasoning_effort:
            params["reasoning_effort"] = self.reasoning_effort
        return params


@dataclass(frozen=True)
class Part:
    """One user-message part: text, or a PNG file sent losslessly as a data URI."""
    text: str | None = None
    image: Path | None = None

    @staticmethod
    def t(text: str) -> "Part":
        return Part(text=text)

    @staticmethod
    def img(path: Path | str) -> "Part":
        return Part(image=Path(path))


@dataclass
class JudgeResult:
    status: str                       # ok | invalid | technical
    value: dict[str, Any] | None      # validated facts when ok
    request_sha256: str
    record_path: str
    resolved_model: str | None = None
    attempts: int = 0
    error: str | None = None
    replayed: bool = False
    usage: dict[str, Any] | None = field(default=None)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_json_content(content: Any) -> dict[str, Any]:
    if not isinstance(content, str) or not content.strip():
        raise InvalidReply("empty content")
    text = content.strip()
    match = FENCE.match(text)
    if match:
        text = match.group(1)
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise InvalidReply(f"content is not one JSON object: {exc}") from None
    if not isinstance(value, dict):
        raise InvalidReply("content JSON is not an object")
    return value


class TechnicalError(RuntimeError):
    pass


def _http_post_stream(url: str, body: bytes, headers: dict[str, str], timeout: int) -> tuple[int, bytes]:
    """Send with "stream": true and rebuild one non-streaming chat.completion JSON from the SSE deltas, so the caller
    sees the same shape either way. Streaming keeps the connection busy (no idle gateway timeout) and is faster."""
    payload = json.loads(body)
    payload["stream"] = True
    payload["stream_options"] = {"include_usage": True}
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST", headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()
    content, reasoning, model, finish, usage, rid, error = [], [], None, None, None, None, None
    with response:
        if "text/event-stream" not in (response.headers.get("Content-Type") or ""):
            return response.status, response.read()  # gateway answered without streaming
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if "error" in chunk:
                error = chunk["error"]
                break
            model, rid = chunk.get("model", model), chunk.get("id", rid)
            usage = chunk.get("usage") or usage
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                content.append(delta.get("content") or "")
                reasoning.append(delta.get("reasoning_content") or "")
                finish = choice.get("finish_reason") or finish
    if error is not None:
        return 200, json.dumps({"error": error}).encode()
    return 200, json.dumps({"id": rid, "model": model, "usage": usage, "choices": [{
        "finish_reason": finish, "message": {"content": "".join(content), "reasoning_content": "".join(reasoning)}}]}).encode()


def _http_post(url: str, body: bytes, headers: dict[str, str], timeout: int) -> tuple[int, bytes]:
    request = urllib.request.Request(url, data=body, method="POST", headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


class Judge:
    def __init__(self, config: JudgeConfig, cache_dir: Path, *, post: Callable[..., tuple[int, bytes]] | None = None,
                 sleep: Callable[[float], None] = time.sleep, api_key: str | None = None):
        self.config = config
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if not config.base_url:
            raise ValueError("set JUDGE_BASE_URL (or pass --base-url) to the OpenAI-compatible judge endpoint")
        self.url = config.base_url.rstrip("/") + "/v1/chat/completions"
        self._post = post or (_http_post_stream if config.stream else _http_post)
        self._sleep = sleep
        self._key = api_key if api_key is not None else os.environ.get(config.api_key_env, "")
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    # ----- request construction -------------------------------------------------------------------------------
    def build(self, system_prompt: str, parts: list[Part]) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return (payload sent to the API, hash-only request summary)."""
        content, summary_parts = [], []
        for part in parts:
            if part.image is not None:
                data = part.image.read_bytes()
                if data[:8] != b"\x89PNG\r\n\x1a\n":
                    raise ValueError(f"judge images must be PNG: {part.image}")
                content.append({"type": "image_url",
                                "image_url": {"url": "data:image/png;base64," + base64.b64encode(data).decode(),
                                              "detail": "high"}})
                summary_parts.append({"image_png_sha256": sha256_bytes(data)})
            elif part.text is not None:
                content.append({"type": "text", "text": part.text})
                summary_parts.append({"text": part.text})
            else:
                raise ValueError("empty part")
        payload = {"model": self.config.model, **self.config.request_parameters(),
                   "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": content}]}
        summary = {"endpoint": self.url, "model": self.config.model, "parameters": self.config.request_parameters(),
                   "system_prompt_sha256": sha256_bytes(system_prompt.encode()), "parts": summary_parts}
        return payload, summary

    def record_path(self, request_sha256: str) -> Path:
        return self.cache_dir / request_sha256[:2] / f"{request_sha256}.json"

    def _lock(self, key: str) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(key, threading.Lock())

    # ----- one logical request -------------------------------------------------------------------------------
    def ask(self, system_prompt: str, parts: list[Part], validator: Callable[[dict[str, Any]], dict[str, Any]],
            *, role: str, retry_invalid: bool = False) -> JudgeResult:
        payload, summary = self.build(system_prompt, parts)
        key = sha256_bytes(canonical(summary).encode())
        path = self.record_path(key)
        with self._lock(key):
            record = json.loads(path.read_text()) if path.exists() else None
            if record and (record["status"] == "ok" or (record["status"] == "invalid" and not retry_invalid)):
                return self._result(record, path, replayed=True, validator=validator)
            if record is None:
                record = {"schema": SCHEMA, "request_sha256": key, "role": role, "request": summary,
                          "created_at": _now(), "attempts": []}
            record["status"] = self._run(payload, record, validator)
            self._write(path, record)
            return self._result(record, path, replayed=False, validator=validator)

    def _run(self, payload: dict[str, Any], record: dict[str, Any], validator) -> str:
        body = json.dumps(payload).encode()
        headers = {"Authorization": "Bearer " + self._key, "Content-Type": "application/json"}
        technical = invalid = 0
        while True:
            attempt: dict[str, Any] = {"started_at": _now()}
            record["attempts"].append(attempt)
            started = time.perf_counter()
            try:
                try:
                    status, raw = self._post(self.url, body, headers, self.config.timeout_seconds)
                except Exception as exc:  # transport: timeout, reset, DNS, SSL
                    raise TechnicalError(f"transport {type(exc).__name__}: {exc}") from None
                attempt["http_status"] = status
                text = raw.decode("utf-8", "replace")
                if status != 200:
                    if status in (408, 429) or status >= 500:
                        raise TechnicalError(f"HTTP {status}: {text[:500]}")
                    attempt.update(outcome="fatal", error=f"HTTP {status}: {text[:1000]}")
                    record["error"] = attempt["error"]
                    return "technical"  # 4xx other than 408/429: request itself rejected; not retried
                try:
                    response = json.loads(text)
                except json.JSONDecodeError:
                    raise TechnicalError(f"non-JSON HTTP body: {text[:300]}") from None
                if not isinstance(response, dict) or "error" in response or not response.get("choices"):
                    raise TechnicalError(f"error/empty body: {text[:500]}")
                choice = response["choices"][0]
                message = choice.get("message") or {}
                attempt.update(resolved_model=response.get("model"), finish_reason=choice.get("finish_reason"),
                               usage=response.get("usage"), response_id=response.get("id"),
                               content=message.get("content"), reasoning_content=message.get("reasoning_content"))
                if response.get("model") not in self.config.accepted():
                    raise TechnicalError(f"resolved model {response.get('model')!r} not in {self.config.accepted()}")
                if not (isinstance(message.get("content"), str) and message["content"].strip()):
                    raise TechnicalError(f"empty content (finish_reason={choice.get('finish_reason')!r})")
                try:
                    value = validator(parse_json_content(message["content"]))
                except InvalidReply as exc:
                    invalid += 1
                    attempt.update(outcome="invalid", error=str(exc)[:1000])
                    if invalid >= self.config.max_invalid_attempts:
                        record["error"] = attempt["error"]
                        return "invalid"
                    continue
                attempt.update(outcome="ok", elapsed_seconds=round(time.perf_counter() - started, 3))
                record.update(value=value, resolved_model=response.get("model"), error=None)
                return "ok"
            except TechnicalError as exc:
                technical += 1
                attempt.update(outcome="technical", error=str(exc)[:1000],
                               elapsed_seconds=round(time.perf_counter() - started, 3))
                if technical >= self.config.max_technical_attempts:
                    record["error"] = attempt["error"]
                    return "technical"
                self._sleep(min(300, 15 * 2 ** (technical - 1)))

    def _result(self, record, path, *, replayed, validator) -> JudgeResult:
        value = record.get("value")
        if record["status"] == "ok" and replayed:
            value = validator(dict(value))  # re-validate replayed facts with the current validator
        last = record["attempts"][-1] if record["attempts"] else {}
        return JudgeResult(status=record["status"], value=value if record["status"] == "ok" else None,
                           request_sha256=record["request_sha256"], record_path=str(path),
                           resolved_model=record.get("resolved_model"), attempts=len(record["attempts"]),
                           error=record.get("error"), replayed=replayed, usage=last.get("usage"))

    @staticmethod
    def _write(path: Path, record: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp.", suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(record, handle, ensure_ascii=False, indent=1)
        os.replace(tmp, path)


def config_dict(config: JudgeConfig) -> dict[str, Any]:
    return asdict(config)
