"""Minimal OpenRouter client.

The API key and model id are read only from the OPENROUTER_API_KEY and
OPENROUTER_MODEL environment variables. A local .env file (gitignored) is
loaded into the environment if present, but never overrides a variable that
is already set.

Rate-limited calls (HTTP 429) are retried a few times with a pause, so a run
of back-to-back requests survives OpenRouter's per-minute limit.
"""
import json
import os
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent

MAX_RETRIES = 3          # retries after the first attempt, only for rate limits
BACKOFF_SECONDS = 5      # pause before retry 1; doubles each retry
MAX_WAIT_SECONDS = 60    # never sleep longer than this for one retry


class RateLimitError(RuntimeError):
    pass


def _ssl_context() -> ssl.SSLContext:
    # python.org builds on macOS ship without system root certs; prefer certifi's bundle.
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _load_dotenv(path: Path = ROOT / ".env") -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def load_config(path: Path = ROOT / "config.json") -> dict:
    return json.loads(path.read_text())


def _require_env(name: str) -> str:
    _load_dotenv()
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is not set. Put it in .env or export it in your shell.")
    return value


def get_api_key() -> str:
    return _require_env("OPENROUTER_API_KEY")


def get_model() -> str:
    return _require_env("OPENROUTER_MODEL")


def _retry_wait(headers, attempt: int) -> float:
    """Seconds to pause: Retry-After, then X-RateLimit-Reset (epoch ms), then exponential backoff."""
    default = BACKOFF_SECONDS * 2**attempt
    try:
        if headers and headers.get("Retry-After"):
            wait = float(headers["Retry-After"])
        elif headers and headers.get("X-RateLimit-Reset"):
            wait = float(headers["X-RateLimit-Reset"]) / 1000 - time.time()
        else:
            wait = default
    except ValueError:
        wait = default
    return min(max(wait, 1.0), MAX_WAIT_SECONDS)


def _post(payload: dict) -> dict:
    """One HTTP call. Raises RateLimitError(headers attached) on a 429."""
    request = urllib.request.Request(
        f"{load_config()['openrouter_base_url']}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {get_api_key()}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, context=_ssl_context(), timeout=60) as response:
            body = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:500]
        if exc.code == 429:
            err = RateLimitError(f"OpenRouter rate limit (429): {detail}")
            err.headers = exc.headers
            raise err from exc
        raise RuntimeError(f"OpenRouter HTTP {exc.code}: {detail}") from exc
    # Upstream provider errors can arrive inside a 200 body.
    error = body.get("error") if isinstance(body, dict) else None
    if error:
        if error.get("code") == 429:
            err = RateLimitError(f"OpenRouter upstream rate limit: {error.get('message')}")
            err.headers = None
            raise err
        raise RuntimeError(f"OpenRouter error: {error}")
    return body


def chat(
    messages: list[dict],
    model: str | None = None,
    response_format: dict | None = None,
    sleep=time.sleep,
) -> str:
    payload = {"model": model or get_model(), "messages": messages}
    if response_format:
        payload["response_format"] = response_format
        # Route only to providers that actually enforce the schema.
        payload["provider"] = {"require_parameters": True}

    for attempt in range(MAX_RETRIES + 1):
        try:
            body = _post(payload)
            return body["choices"][0]["message"]["content"]
        except RateLimitError as exc:
            # A daily free-model quota won't reset in seconds; fail fast instead of waiting.
            if attempt == MAX_RETRIES or "per-day" in str(exc):
                raise
            sleep(_retry_wait(exc.headers, attempt))
    raise AssertionError("unreachable")


if __name__ == "__main__":
    print(chat([{"role": "user", "content": "Say hello in five words."}]))
