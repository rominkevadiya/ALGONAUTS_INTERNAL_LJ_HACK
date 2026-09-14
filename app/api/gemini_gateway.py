"""Centralized, cached access to the Google Gemini SDK."""

from __future__ import annotations

import hashlib
import logging
import os
import threading
from typing import Any, Dict, Optional, Union

from dotenv import load_dotenv
from PIL import Image

load_dotenv()

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None  # type: ignore[assignment]
    types = None  # type: ignore[assignment]


GEMINI_MODEL = "gemini-3.6-flash"
_LOGGER = logging.getLogger(__name__)
_CLIENT: Any = None
_CLIENT_LOCK = threading.Lock()
_CACHE: Dict[str, str] = {}
_CACHE_LOCK = threading.Lock()
_IN_FLIGHT: Dict[str, threading.Event] = {}


def _result(success: bool, text: str = "", error: Optional[str] = None, *, cache_hit: bool = False, cache_miss: bool = False) -> Dict[str, Any]:
    return {"success": success, "text": text, "error": error, "cache_hit": cache_hit, "cache_miss": cache_miss}


def _image_hash(image: Union[Image.Image, bytes, bytearray]) -> str:
    """Hash image content without retaining or logging the content."""
    digest = hashlib.sha256()
    if isinstance(image, Image.Image):
        normalized = image.convert("RGBA")
        digest.update(normalized.mode.encode("utf-8"))
        digest.update(str(normalized.size).encode("utf-8"))
        digest.update(normalized.tobytes())
    else:
        digest.update(bytes(image))
    return digest.hexdigest()


def _cache_key(*, task_id: str, prompt: str, image: Optional[Union[Image.Image, bytes, bytearray]], temperature: Optional[float], max_output_tokens: Optional[int], prompt_version: Optional[str]) -> str:
    """Build a deterministic key from all response-affecting request inputs."""
    components = {
        "task_id": task_id,
        "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "prompt_version": prompt_version or "",
        "image_hash": _image_hash(image) if image is not None else "",
        "model": GEMINI_MODEL,
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
    }
    return hashlib.sha256(repr(sorted(components.items())).encode("utf-8")).hexdigest()


def _get_client() -> Any:
    """Return the singleton SDK client; it is created only in this module."""
    global _CLIENT
    if genai is None:
        return None
    if _CLIENT is None:
        with _CLIENT_LOCK:
            if _CLIENT is None:
                api_key = os.getenv("GEMINI_API_KEY")
                if not api_key:
                    return None
                _CLIENT = genai.Client(api_key=api_key)
    return _CLIENT


def _error_code(exc: Exception) -> str:
    message = str(exc).upper()
    if "429" in message or "RESOURCE_EXHAUSTED" in message:
        return "rate_limited"
    if any(marker in message for marker in ("TIMEOUT", "UNAVAILABLE", "CONNECTION", "TEMPORARY", "500", "502", "503", "504")):
        return "transient_api_failure"
    return "gemini_api_failure"


def _generate(*, task_id: str, prompt: str, image: Optional[Union[Image.Image, bytes, bytearray]], temperature: Optional[float], max_output_tokens: Optional[int], prompt_version: Optional[str]) -> Dict[str, Any]:
    if genai is None or types is None:
        return _result(False, error="google_genai_not_installed")
    if not os.getenv("GEMINI_API_KEY"):
        return _result(False, error="missing_api_key")
    if not task_id.strip() or not prompt.strip():
        return _result(False, error="invalid_request")

    key = _cache_key(task_id=task_id, prompt=prompt, image=image, temperature=temperature, max_output_tokens=max_output_tokens, prompt_version=prompt_version)
    with _CACHE_LOCK:
        cached = _CACHE.get(key)
        if cached is not None:
            return _result(True, cached, cache_hit=True)
        event = _IN_FLIGHT.get(key)
        request_owner = event is None
        if request_owner:
            event = threading.Event()
            _IN_FLIGHT[key] = event

    if not request_owner:
        # Coalesce simultaneous Streamlit reruns for an identical request.
        event.wait()
        with _CACHE_LOCK:
            cached = _CACHE.get(key)
        if cached is not None:
            return _result(True, cached, cache_hit=True)
        return _result(False, error="gemini_api_failure", cache_miss=True)

    try:
        client = _get_client()
        if client is None:
            return _result(False, error="missing_api_key", cache_miss=True)
        config = types.GenerateContentConfig(temperature=temperature, max_output_tokens=max_output_tokens)
        response = client.models.generate_content(model=GEMINI_MODEL, contents=[image, prompt] if image is not None else prompt, config=config)
        text = getattr(response, "text", None)
        if not isinstance(text, str) or not text.strip():
            _LOGGER.warning("Gemini returned an empty or malformed response for task '%s'.", task_id)
            return _result(False, error="malformed_response", cache_miss=True)
        clean_text = text.strip()
        with _CACHE_LOCK:
            _CACHE[key] = clean_text
        return _result(True, clean_text, cache_miss=True)
    except Exception as exc:
        error = _error_code(exc)
        _LOGGER.warning("Gemini request failed for task '%s' (%s).", task_id, error)
        return _result(False, error=error, cache_miss=True)
    finally:
        with _CACHE_LOCK:
            completed = _IN_FLIGHT.pop(key, None)
            if completed is not None:
                completed.set()


def generate_text(*, task_id: str, prompt: str, temperature: Optional[float] = None, max_output_tokens: Optional[int] = None, prompt_version: Optional[str] = None) -> Dict[str, Any]:
    """Generate text for a named task through the shared Gemini client."""
    return _generate(task_id=task_id, prompt=prompt, image=None, temperature=temperature, max_output_tokens=max_output_tokens, prompt_version=prompt_version)


def generate_multimodal(*, task_id: str, image: Union[Image.Image, bytes, bytearray], prompt: str, temperature: Optional[float] = None, max_output_tokens: Optional[int] = None, prompt_version: Optional[str] = None) -> Dict[str, Any]:
    """Generate text from an image plus prompt through the shared Gemini client."""
    return _generate(task_id=task_id, prompt=prompt, image=image, temperature=temperature, max_output_tokens=max_output_tokens, prompt_version=prompt_version)


def clear_response_cache() -> None:
    """Clear the in-memory response cache (primarily for tests)."""
    with _CACHE_LOCK:
        _CACHE.clear()
