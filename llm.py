"""LLM API wrapper with retry logic — supports OpenAI and Llama (OpenAI-compatible)."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Callable, Optional

from openai import OpenAI
from PyPDF2 import PdfReader
import io

from config import MAX_RETRIES, RETRY_DELAYS, OPENAI_DEFAULT_MODEL, UF_NAVIGATOR_BASE_URL

# --- Legacy Gemini imports (commented out) ---
# from google import genai
# from google.genai import types


def _get_api_key(api_key: str | None = None) -> str:
    """Get API key from param, env var, or raise an error."""
    key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY")
    if not key:
        raise ValueError("No API key provided. Set OPENAI_API_KEY or pass api_key parameter.")
    return key


def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract text content from PDF bytes using PyPDF2."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    text_parts = []
    for page_num, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            text_parts.append(f"--- Page {page_num + 1} ---\n{page_text}")
    return "\n\n".join(text_parts)


def load_pdf_as_text(pdf_path: str) -> str:
    """Read a PDF file and return its text content."""
    pdf_bytes = Path(pdf_path).read_bytes()
    return extract_pdf_text(pdf_bytes)


# --- Legacy Gemini function (commented out) ---
# def load_pdf_as_part(pdf_path: str) -> types.Part:
#     """Read a PDF file and return a Gemini API Part for inline data."""
#     pdf_bytes = Path(pdf_path).read_bytes()
#     return types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf")


def _build_client(api_key: str, provider: str = "openai", base_url: str | None = None) -> OpenAI:
    """Build an OpenAI client (works for OpenAI, UF Navigator, Llama, and other compatible APIs)."""
    key = _get_api_key(api_key)
    kwargs = {"api_key": key}
    if provider == "uf-navigator":
        kwargs["base_url"] = base_url or UF_NAVIGATOR_BASE_URL
    elif provider == "llama" and base_url:
        kwargs["base_url"] = base_url
    elif provider == "llama" and not base_url:
        # Default to Together AI if no base_url given for llama
        kwargs["base_url"] = "https://api.together.xyz/v1"
    return OpenAI(**kwargs)


def call_llm(
    system_prompt: str,
    user_content: list[str],
    max_tokens: int = 8192,
    model: str = OPENAI_DEFAULT_MODEL,
    api_key: str | None = None,
    provider: str = "openai",
    base_url: str | None = None,
    on_thinking: Optional[Callable[[str], None]] = None,
) -> str:
    """Make an LLM API call via OpenAI-compatible interface and return the text response.

    Args:
        system_prompt: System instruction for the model.
        user_content: List of text strings to send as user message parts.
        max_tokens: Maximum output tokens.
        model: Model ID (e.g., 'gpt-4o-mini', 'meta-llama/...').
        api_key: API key for the provider.
        provider: 'openai' or 'llama'.
        base_url: Custom base URL for Llama-compatible providers.
        on_thinking: Optional callback for thinking/reasoning tokens (if supported).

    Returns:
        The model's text response.
    """
    client = _build_client(api_key, provider, base_url)

    # Combine all user content parts into a single user message
    combined_user_text = "\n\n".join(user_content)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": combined_user_text},
    ]

    if on_thinking:
        # Use streaming to capture partial responses
        full_text = ""
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.7,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                full_text += token
                # Send tokens to thinking callback for live display
                on_thinking(token)
        return full_text
    else:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.7,
        )
        return response.choices[0].message.content


# --- Legacy Gemini call functions (commented out) ---
# def call_gemini(
#     system_prompt: str,
#     user_content: list,
#     max_tokens: int = 8192,
#     model: str = "gemini-2.5-pro",
#     api_key: str | None = None,
#     on_thinking: Optional[Callable[[str], None]] = None,
# ) -> str:
#     """Make a Gemini API call and return the text response."""
#     client = genai.Client(api_key=_get_api_key(api_key))
#     config = types.GenerateContentConfig(
#         system_instruction=system_prompt,
#         max_output_tokens=max_tokens,
#         temperature=0.7,
#     )
#     if on_thinking:
#         full_text = ""
#         for chunk in client.models.generate_content_stream(
#             model=model, contents=user_content, config=config
#         ):
#             try:
#                 if chunk.candidates and chunk.candidates[0].content and chunk.candidates[0].content.parts:
#                     for part in chunk.candidates[0].content.parts:
#                         if getattr(part, 'thought', False):
#                             if part.text:
#                                 on_thinking(part.text)
#                         else:
#                             if part.text:
#                                 full_text += part.text
#             except (AttributeError, IndexError):
#                 if hasattr(chunk, 'text') and chunk.text:
#                     full_text += chunk.text
#         return full_text
#     else:
#         response = client.models.generate_content(
#             model=model, contents=user_content, config=config
#         )
#         return response.text


def call_llm_with_retry(
    system_prompt: str,
    user_content: list[str],
    max_tokens: int = 8192,
    model: str = OPENAI_DEFAULT_MODEL,
    api_key: str | None = None,
    provider: str = "openai",
    base_url: str | None = None,
    on_thinking: Optional[Callable[[str], None]] = None,
) -> str:
    """Call LLM API with retry logic for transient errors."""
    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            return call_llm(
                system_prompt, user_content, max_tokens, model,
                api_key, provider, base_url, on_thinking,
            )

        except Exception as e:
            error_str = str(e).lower()
            if any(keyword in error_str for keyword in ["429", "rate", "500", "503", "overloaded", "unavailable"]):
                last_error = e
                wait = RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)]
                print(f"  Transient error. Waiting {wait}s before retry {attempt + 1}/{MAX_RETRIES}...")
                time.sleep(wait)
            else:
                raise

    raise RuntimeError(f"Failed after {MAX_RETRIES} retries. Last error: {last_error}")


# Legacy alias (commented out)
# call_gemini_with_retry = call_llm_with_retry


def parse_llm_json(raw_text: str, step_name: str, model: str, api_key: str | None = None,
                   provider: str = "openai", base_url: str | None = None) -> dict | list:
    """Parse JSON from LLM response, with cleanup and one repair attempt."""
    text = raw_text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        first_newline = text.index("\n")
        text = text[first_newline + 1:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"  Warning: JSON parse failed in {step_name}. Attempting repair...")
        repair_prompt = (
            f"The following text was supposed to be valid JSON but has a syntax error:\n\n"
            f"{text[:4000]}\n\n"
            f"Error: {e}\n\n"
            f"Return ONLY the corrected valid JSON, nothing else."
        )
        repaired = call_llm_with_retry(
            system_prompt="You are a JSON repair tool. Return only valid JSON.",
            user_content=[repair_prompt],
            max_tokens=max(len(text) // 2, 4096),
            model=model,
            api_key=api_key,
            provider=provider,
            base_url=base_url,
        )
        repaired = repaired.strip()
        if repaired.startswith("```"):
            repaired = repaired.split("\n", 1)[1]
        if repaired.endswith("```"):
            repaired = repaired[:-3]
        return json.loads(repaired.strip())
