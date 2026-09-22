"""OpenAI-compatible mimo LLM helper.

Reads gitignored repo `.env` (META_JEV_LLM_*). Manual parse — no python-dotenv.
Never call from eval-afa scoring path.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ENV_KEYS = (
    "META_JEV_LLM_BASE_URL",
    "META_JEV_LLM_MODEL",
    "META_JEV_LLM_API_KEY",
)


def _find_repo_env(start: Path | None = None) -> Path | None:
    """Walk up from *start* (or this file) looking for `.env`."""
    cur = (start or Path(__file__).resolve()).parent
    for _ in range(12):
        candidate = cur / ".env"
        if candidate.is_file():
            return candidate
        if (cur / "pyproject.toml").is_file() and (cur / ".env").is_file():
            return cur / ".env"
        if cur.parent == cur:
            break
        cur = cur.parent
    # also check CWD
    cwd_env = Path.cwd() / ".env"
    if cwd_env.is_file():
        return cwd_env
    return None


def load_dotenv_env(path: str | Path | None = None, *, override: bool = False) -> dict[str, str]:
    """Parse KEY=VALUE lines from `.env` into os.environ (and return dict).

    Does not print or return secret values to callers beyond the dict itself;
    callers must not log API keys.
    """
    env_path = Path(path) if path else _find_repo_env()
    loaded: dict[str, str] = {}
    if env_path is None or not env_path.is_file():
        return loaded
    text = env_path.read_text(encoding="utf-8")
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip()
        if (val.startswith('"') and val.endswith('"')) or (
            val.startswith("'") and val.endswith("'")
        ):
            val = val[1:-1]
        loaded[key] = val
        if override or key not in os.environ:
            os.environ[key] = val
    return loaded


def _cfg() -> tuple[str, str, str]:
    load_dotenv_env()
    base = os.environ.get("META_JEV_LLM_BASE_URL", "").rstrip("/")
    model = os.environ.get("META_JEV_LLM_MODEL", "")
    key = os.environ.get("META_JEV_LLM_API_KEY", "")
    if not base or not model:
        raise RuntimeError(
            "META_JEV_LLM_BASE_URL and META_JEV_LLM_MODEL must be set in .env"
        )
    return base, model, key


def chat_completion(
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.0,
    max_tokens: int = 256,
    timeout: float = 60.0,
) -> dict[str, Any]:
    """POST {BASE_URL}/chat/completions; returns parsed JSON response."""
    base, model, api_key = _cfg()
    url = f"{base}/chat/completions"
    body = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    data = json.dumps(body).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM HTTP {e.code}: {err_body[:500]}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"LLM network error: {e}") from e
    return json.loads(raw)


def ping(timeout: float = 20.0) -> str:
    """Tiny smoke: one-token chat. Returns assistant text snippet (no secrets)."""
    resp = chat_completion(
        [{"role": "user", "content": "Reply with exactly: pong"}],
        max_tokens=16,
        timeout=timeout,
    )
    choices = resp.get("choices") or []
    if not choices:
        return ""
    msg = choices[0].get("message") or {}
    return str(msg.get("content") or "").strip()


__all__ = [
    "ENV_KEYS",
    "chat_completion",
    "load_dotenv_env",
    "ping",
]
