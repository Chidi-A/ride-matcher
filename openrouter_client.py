"""Minimal OpenRouter client.

The API key is read only from the OPENROUTER_API_KEY environment variable.
A local .env file (gitignored) is loaded into the environment if present,
but never overrides a variable that is already set.
"""
import json
import os
import ssl
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent


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


def get_api_key() -> str:
    _load_dotenv()
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Put it in .env or export it in your shell."
        )
    return key


def chat(messages: list[dict], model: str | None = None) -> str:
    config = load_config()
    request = urllib.request.Request(
        f"{config['openrouter_base_url']}/chat/completions",
        data=json.dumps(
            {"model": model or config["openrouter_model"], "messages": messages}
        ).encode(),
        headers={
            "Authorization": f"Bearer {get_api_key()}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, context=_ssl_context(), timeout=60) as response:
        body = json.load(response)
    return body["choices"][0]["message"]["content"]


if __name__ == "__main__":
    print(chat([{"role": "user", "content": "Say hello in five words."}]))
