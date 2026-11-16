#!/usr/bin/env python3
"""Clients for the closed image editing models (GPT-Image-2.5, Gemini-3-Pro-Image, Nano-Banana-2.1).

One call = the frozen input image + the frozen prompt text, sent verbatim as a single user message to an
OpenAI-compatible /v1/chat/completions gateway (ChatImageClient: Gemini image models), or (ImagesEditClient) as one
multipart /v1/images/edits request with size=1024x1024 and n=1 (GPT-Image); other request parameters are left at the
gateway defaults. Gateway: IMAGE_API_BASE_URL, key: IMAGE_API_KEY.
The reply must contain exactly one image (markdown data URI or URL). A 2048x2048 reply (~7% of calls) is reduced by
an exact 2x box downscale (recorded as receipt "postprocess", raw bytes kept); any other
size than the protocol size is an error.

Retries are technical only: transport errors, HTTP 429/5xx, and replies without an image. A reply that has an image
is never retried, whatever it shows. Every attempt is written to api_receipt.json (base64 payloads elided) and the
raw returned bytes to api_raw_output.<ext>, both next to the output. The API key comes from the environment variable
named by api_key_env and is never written anywhere.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

DATA_URI = re.compile(r"data:image/([a-zA-Z0-9.+-]+);base64,([A-Za-z0-9+/=\s]+)")
IMAGE_URL = re.compile(r"!\[[^\]]*\]\((https?://[^)\s]+)\)")
RETRY_WAIT = (20, 60, 120, 240)


def downscale_exact_2x(image: Image.Image, size: tuple[int, int]) -> tuple[Image.Image, dict[str, Any] | None]:
    image = image.convert("RGB")
    if image.size == size:
        return image, None
    if image.size != (2 * size[0], 2 * size[1]):
        return image, None  # caller rejects the size
    return image.resize(size, Image.Resampling.BOX), {
        "op": "exact_2x_box_downscale", "from": list(image.size), "to": list(size), "filter": "PIL BOX",
        "decision": "keep every first reply"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _elide(text: str) -> str:
    return DATA_URI.sub(lambda m: f"data:image/{m.group(1)};base64,<{len(m.group(2))} chars elided>", text)


def _download(url: str, timeout: int) -> bytes:
    """Fetch a returned image URL. Some CDNs reject urllib's default User-Agent with 403. A failed
    download is retried here, never by re-generating."""
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except Exception:
            if attempt == 2:
                raise DownloadFailed(f"could not download generated image {url}")
            time.sleep(10 * (attempt + 1))
    raise AssertionError("unreachable")


class DownloadFailed(RuntimeError):
    pass


class NoImageReply(RuntimeError):
    pass


class ChatImageClient:
    PATH = "/v1/chat/completions"
    content_type = "application/json"

    def __init__(self, config: dict[str, Any]):
        self.model = config["api_model"]
        self.url = config["api_base_url"].rstrip("/") + self.PATH
        self.key = os.environ.get(config["api_key_env"], "")
        if not self.key:
            raise RuntimeError(f"API key environment variable {config['api_key_env']} is not set")
        self.timeout = int(config.get("api_timeout_seconds", 600))
        self.max_attempts = int(config.get("api_max_attempts", 5))

    def _post(self, body: bytes) -> tuple[int, bytes]:
        request = urllib.request.Request(self.url, data=body, method="POST", headers={
            "Authorization": "Bearer " + self.key, "Content-Type": self.content_type})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read()

    def _extract(self, payload: dict[str, Any]) -> tuple[str, bytes, str]:
        choices = payload.get("choices") or []
        content = choices[0].get("message", {}).get("content") if choices else None
        if isinstance(content, list):  # some gateways return content parts
            content = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
        content = content or ""
        found = [("data", m.group(1), base64.b64decode(re.sub(r"\s", "", m.group(2)))) for m in DATA_URI.finditer(content)]
        for m in IMAGE_URL.finditer(content):
            found.append(("url", m.group(1).rsplit(".", 1)[-1][:5], _download(m.group(1), self.timeout)))
        if not found:
            raise NoImageReply(f"reply has no image: {_elide(content)[:500]!r}")
        if len(found) != 1:
            raise RuntimeError(f"reply has {len(found)} images; exactly one is required (no selection)")
        kind, ext, raw = found[0]
        return ext.lower().replace("jpeg", "jpg"), raw, _elide(content)

    def _body(self, prompt: str, image_bytes: bytes, size: tuple[int, int]) -> bytes:
        return json.dumps({"model": self.model, "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(image_bytes).decode()}},
        ]}]}).encode()

    def edit(self, *, input_path: Path, prompt: str, receipt_dir: Path, size: tuple[int, int]) -> Image.Image:
        image_bytes = input_path.read_bytes()
        body = self._body(prompt, image_bytes, size)
        receipt: dict[str, Any] = {
            "schema": "pitfalls-pixels/api-image-receipt/v1", "endpoint": self.url, "requested_model": self.model,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "input_sha256": hashlib.sha256(image_bytes).hexdigest(),
            "request_body_sha256": hashlib.sha256(body).hexdigest(),
            "seed_sent": False, "retry_policy": "technical only: transport, HTTP 429/5xx, reply without image",
            "attempts": [],
        }
        path = receipt_dir / "api_receipt.json"
        try:
            for attempt in range(1, self.max_attempts + 1):
                record: dict[str, Any] = {"attempt": attempt, "started_at": _now()}
                receipt["attempts"].append(record)
                started = time.perf_counter()
                retryable = True
                try:
                    status, raw = self._post(body)
                    record["http_status"] = status
                    record["elapsed_seconds"] = round(time.perf_counter() - started, 3)
                    if status != 200:
                        record["error"] = _elide(raw.decode("utf-8", "replace"))[:1000]
                        retryable = status in (408, 429) or status >= 500  # 408 = upstream throttle
                        raise RuntimeError(f"HTTP {status}: {record['error'][:300]}")
                    payload = json.loads(raw)
                    record.update({k: payload.get(k) for k in ("id", "model", "created", "usage")})
                    record["finish_reason"] = (payload.get("choices") or [{}])[0].get("finish_reason")
                    record.update({k: payload[k] for k in ("size", "quality", "output_format", "moderation", "background")
                                   if k in payload})
                    ext, image_raw, content = self._extract(payload)
                    record["reply_content_elided"] = content[:2000]
                    # Name the raw file by its decoded format; URL suffixes are not reliable file extensions.
                    image = Image.open(io.BytesIO(image_raw))
                    raw_name = "api_raw_output." + {"PNG": "png", "JPEG": "jpg", "WEBP": "webp"}.get(image.format, "bin")
                    (receipt_dir / raw_name).write_bytes(image_raw)
                    image.load()
                    receipt.update({"status": "SUCCESS", "resolved_model": payload.get("model"), "raw_output_file": raw_name,
                                    "raw_output_sha256": hashlib.sha256(image_raw).hexdigest(),
                                    "raw_output_format": image.format, "raw_output_size": list(image.size),
                                    "raw_output_mode": image.mode})
                    image, postprocess = downscale_exact_2x(image, size)
                    receipt["postprocess"] = postprocess
                    return image
                except NoImageReply as exc:
                    record["error"] = str(exc)
                except DownloadFailed as exc:
                    record["error"] = str(exc)
                    raise
                except Exception as exc:
                    record.setdefault("error", f"{type(exc).__name__}: {exc}"[:1000])
                    record.setdefault("elapsed_seconds", round(time.perf_counter() - started, 3))
                    if not retryable:
                        raise
                if attempt == self.max_attempts:
                    raise RuntimeError(f"API failed after {attempt} attempts: {record.get('error', '')[:300]}")
                time.sleep(RETRY_WAIT[min(attempt - 1, len(RETRY_WAIT) - 1)])
            raise AssertionError("unreachable")
        except Exception:
            receipt["status"] = "FAILED"
            raise
        finally:
            path.write_text(json.dumps(receipt, ensure_ascii=False, indent=1), encoding="utf-8")


class ImagesEditClient(ChatImageClient):
    PATH = "/v1/images/edits"
    BOUNDARY = "pitfallspixelsboundary7f3a"
    content_type = "multipart/form-data; boundary=" + BOUNDARY

    def _body(self, prompt: str, image_bytes: bytes, size: tuple[int, int]) -> bytes:
        parts = [(name, None, value.encode()) for name, value in
                 (("model", self.model), ("prompt", prompt), ("size", f"{size[0]}x{size[1]}"), ("n", "1"))]
        parts.append(("image", "input.png", image_bytes))
        out = b""
        for name, filename, value in parts:
            disposition = f'form-data; name="{name}"' + (f'; filename="{filename}"' if filename else "")
            out += f"--{self.BOUNDARY}\r\nContent-Disposition: {disposition}\r\n".encode()
            out += b"Content-Type: image/png\r\n\r\n" if filename else b"\r\n"
            out += value + b"\r\n"
        return out + f"--{self.BOUNDARY}--\r\n".encode()

    def _extract(self, payload: dict[str, Any]) -> tuple[str, bytes, str]:
        data = payload.get("data") or []
        if not data:
            raise NoImageReply(f"reply has no image: {_elide(json.dumps(payload))[:500]!r}")
        if len(data) != 1:
            raise RuntimeError(f"reply has {len(data)} images; exactly one is required (no selection)")
        item = data[0]
        if item.get("b64_json"):
            return (payload.get("output_format") or "png").replace("jpeg", "jpg"), base64.b64decode(item["b64_json"]), "b64_json"
        raw = _download(item["url"], self.timeout)
        return item["url"].rsplit(".", 1)[-1][:5].lower().replace("jpeg", "jpg"), raw, "url " + item["url"]
