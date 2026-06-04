import json
import os
import re
import time

import pytest

openai = pytest.importorskip("openai")
from src.project.config import load_env_file

load_env_file()

CONSULTANT_API_KEY = os.environ.get("CONSULTANT_API_KEY")
CONSULTANT_BASE_URL = os.environ.get("CONSULTANT_BASE_URL")
CONSULTANT_MODEL = os.environ.get("CONSULTANT_MODEL_NAME", "gemini-2.0-flash")


def _key_is_placeholder(key: str) -> bool:
    return not key or "your" in key.lower() or key.startswith("<") or key == "not-needed"


@pytest.fixture
def client():
    if _key_is_placeholder(CONSULTANT_API_KEY or ""):
        pytest.skip("Missing CONSULTANT_API_KEY")
    return openai.OpenAI(api_key=CONSULTANT_API_KEY, base_url=CONSULTANT_BASE_URL)


def robust_call(func, retries=3, delay=5):
    """Retries the API call if a rate limit is hit."""
    for i in range(retries):
        try:
            return func()
        except openai.RateLimitError:
            if i < retries - 1:
                print(f"\n[Rate Limit] Retrying in {delay}s...")
                time.sleep(delay)
            else:
                pytest.skip("Gemini Free Tier Rate Limit: Max retries exceeded.")
        except Exception as e:
            pytest.fail(f"API Error: {e}")


_NO_THINKING = {"chat_template_kwargs": {"enable_thinking": False}}


def test_consultant_basic(client):
    """Test standard text response."""

    def call():
        r = client.chat.completions.create(
            model=CONSULTANT_MODEL,
            messages=[{"role": "user", "content": "Say 'ready'"}],
            extra_body=_NO_THINKING,
        )
        assert "ready" in r.choices[0].message.content.lower()

    robust_call(call)


def test_consultant_json_output(client):
    """Test JSON mode for KELE evaluation."""

    def call():
        r = client.chat.completions.create(
            model=CONSULTANT_MODEL,
            messages=[{"role": "user", "content": 'JSON: {"status": "ok"}'}],
            response_format={"type": "json_object"},
            extra_body=_NO_THINKING,
        )
        content = re.sub(r"^```(?:json)?\s*", "", (r.choices[0].message.content or "").strip())
        content = re.sub(r"\s*```$", "", content)
        data = json.loads(content)
        assert data["status"] == "ok"

    robust_call(call)
