"""LLM judge backed by Google Gemini via the google-genai SDK.

Exposes a single function `llm_judge(prompt)` that returns a tuple of
`(result, usage)` where `result` is shaped like
`{"score": int in [1,10], "reason": str}` and `usage` is shaped like
`{"input_tokens": int, "output_tokens": int}`. `evaluate.py` feeds it the
per-commit prompt built by `build_prompt`.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from google import genai


_MODEL = os.environ.get("LLM_JUDGE_MODEL", "gemini-3.1-flash-lite")
_API_KEY = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY", "YOUR_API_KEY")


_client = genai.Client(
    enterprise=True,
    # api_key=_API_KEY,
)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Best-effort pull of the first {...} JSON object out of a model reply."""
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _extract_usage(response: Any) -> dict[str, int]:
    """Pull input/output token counts off a Gemini response, if present."""
    meta = getattr(response, "usage_metadata", None)
    if meta is None:
        print("no meta data found")
        return {"input_tokens": 0, "output_tokens": 0}
    return {
        "input_tokens": int(getattr(meta, "prompt_token_count", 0) or 0),
        "output_tokens": int(getattr(meta, "candidates_token_count", 0) or 0),
    }


def llm_judge(prompt: str) -> tuple[dict[str, Any], dict[str, int]]:
    """Score a commit prompt with Gemini.

    Returns a tuple `(result, usage)` where `result` is
    `{"score": int, "reason": str}` and `usage` is
    `{"input_tokens": int, "output_tokens": int}`. On any error or
    unparseable reply we fall back to {"score": 0, "reason": "<error message>"}
    so the evaluation pipeline keeps moving.
    """
    try:
        response = _client.models.generate_content(
            model=_MODEL,
            contents=[prompt],
        )
    except Exception as exc:
        return {"score": 0, "reason": f"llm call failed: {exc}"}, {"input_tokens": 0, "output_tokens": 0}

    usage = _extract_usage(response)
    text = (getattr(response, "text", None) or "").strip()
    parsed = _extract_json_object(text)
    if isinstance(parsed, dict):
        return parsed, usage

    return {"score": 0, "reason": f"unparseable judge reply: {text[:140]}"}, usage
