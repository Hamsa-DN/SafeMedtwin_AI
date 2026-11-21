# """
# SafeMedTwin — Unified LLM Provider
# =====================================
# A single abstraction over Claude (Anthropic) and Gemini (Google), so every
# module that needs an LLM (chatbot, diet personalization, report scanning,
# LLM-vs-SLM comparison) calls one function instead of importing a specific
# vendor SDK directly.

# Provider selection (in order of priority):
# 1. SAFEMEDTWIN_LLM_PROVIDER env var — "anthropic" or "gemini", if set
# 2. Auto-detected: if ANTHROPIC_API_KEY is set, use Claude; else if
#    GEMINI_API_KEY or GOOGLE_API_KEY is set, use Gemini
# 3. Neither set — calls raise a clear RuntimeError, which every caller in
#    this project already catches and falls back to an offline default.

# Both providers are given equivalent prompts and equivalent safety framing
# (no diagnoses, no dosages) — see each call site for the actual prompt
# text; this module only handles the "how do I call whichever provider is
# configured" part.
# """
# import os
# import re
# import socket
# import time

# # Transient failures worth retrying rather than surfacing to the user:
# # 503/429/502/504 from the provider (overloaded or rate-limited — very
# # common on Gemini's free tier), and DNS/connection blips on the client
# # side. Anything else (bad API key, malformed request) fails immediately,
# # since retrying won't help.
# _TRANSIENT_PATTERNS = re.compile(
#     r"\b(503|502|504|429|UNAVAILABLE|RESOURCE_EXHAUSTED|overloaded|high demand|"
#     r"rate.?limit|timeout|timed out|getaddrinfo|temporarily)\b",
#     re.IGNORECASE,
# )


# def _is_transient(exc: Exception) -> bool:
#     if isinstance(exc, (socket.gaierror, socket.timeout, ConnectionError, TimeoutError)):
#         return True
#     return bool(_TRANSIENT_PATTERNS.search(str(exc)))


# def _with_retries(fn, attempts: int = 1, base_delay: float = 1.5):
#     """Run fn(), retrying transient failures with exponential backoff.
#     Re-raises the last exception if every attempt fails, so callers keep
#     their existing graceful-fallback behavior."""
#     last_exc = None
#     for attempt in range(attempts):
#         try:
#             return fn()
#         except Exception as e:
#             last_exc = e
#             if not _is_transient(e) or attempt == attempts - 1:
#                 raise
#             time.sleep(base_delay * (2 ** attempt))  # 1.5s, 3s
#     raise last_exc


# def friendly_error(exc: Exception) -> str:
#     """Turn a raw provider/network exception into something a user can act
#     on. Used by call sites for their fallback messages, so a DNS blip
#     doesn't wrongly tell the user their API key is missing."""
#     text = str(exc)
#     if isinstance(exc, socket.gaierror) or "getaddrinfo" in text:
#         return ("Couldn't reach the AI service — this looks like a network/DNS issue on this "
#                 "machine, not a problem with your API key. Check your internet connection "
#                 "(campus Wi-Fi and VPNs often block this) and try again.")
#     if _TRANSIENT_PATTERNS.search(text):
#         return ("The AI service is temporarily busy or rate-limited (common on Gemini's free "
#                 "tier). This usually clears up on its own — please try again in a moment.")
#     if "api" in text.lower() and "key" in text.lower():
#         return ("The AI service rejected the API key. Check that GEMINI_API_KEY (or "
#                 "ANTHROPIC_API_KEY) is set correctly and is still valid.")
#     return f"Couldn't reach the AI service: {exc}"


# def get_provider() -> str:
#     explicit = os.environ.get("SAFEMEDTWIN_LLM_PROVIDER", "").strip().lower()
#     if explicit in ("anthropic", "gemini"):
#         return explicit
#     if os.environ.get("ANTHROPIC_API_KEY"):
#         return "anthropic"
#     if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
#         return "gemini"
#     raise RuntimeError(
#         "No LLM provider configured. Set ANTHROPIC_API_KEY, or GEMINI_API_KEY "
#         "(or GOOGLE_API_KEY), or explicitly set SAFEMEDTWIN_LLM_PROVIDER."
#     )


# ANTHROPIC_MODEL = os.environ.get("SAFEMEDTWIN_ANTHROPIC_MODEL", "claude-sonnet-5")
# GEMINI_MODEL = os.environ.get("SAFEMEDTWIN_GEMINI_MODEL", "gemini-flash-latest")


# def call_text(system_prompt: str, user_message: str, max_tokens: int = 600) -> str:
#     """Plain text-in, text-out completion. Retries transient failures
#     (provider overload, DNS blips) before giving up. Raises on final
#     failure — callers catch and fall back (every call site here does)."""
#     provider = get_provider()

#     def _run():
#         if provider == "anthropic":
#             import anthropic
#             client = anthropic.Anthropic()
#             resp = client.messages.create(
#                 model=ANTHROPIC_MODEL, max_tokens=max_tokens, system=system_prompt,
#                 messages=[{"role": "user", "content": user_message}],
#             )
#             return "".join(b.text for b in resp.content if hasattr(b, "text"))

#         # gemini
#         from google import genai
#         from google.genai import types
#         api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
#         client = genai.Client(api_key=api_key)
#         resp = client.models.generate_content(
#             model=GEMINI_MODEL,
#             contents=[user_message],
#             config=types.GenerateContentConfig(
#                 system_instruction=system_prompt, max_output_tokens=max_tokens,
#                 # Gemini's newer models "think" by default, and those hidden
#                 # thinking tokens are drawn from the SAME max_output_tokens
#                 # budget as the visible answer — without disabling this, replies
#                 # get silently truncated mid-sentence even with a generous
#                 # token limit. These are short, simple completions that don't
#                 # need multi-step reasoning, so disabling thinking is correct
#                 # here (also faster and cheaper).
#                 thinking_config=types.ThinkingConfig(thinking_budget=0),
#             ),
#         )
#         return resp.text or ""

#     return _with_retries(_run)


# def call_chat(system_prompt: str, messages: list, max_tokens: int = 600) -> str:
#     """Multi-turn chat. `messages` is a list of {"role": "user"|"assistant",
#     "content": str}, most recent last (same shape for both providers).
#     Retries transient failures before giving up."""
#     provider = get_provider()

#     def _run():
#         if provider == "anthropic":
#             import anthropic
#             client = anthropic.Anthropic()
#             resp = client.messages.create(
#                 model=ANTHROPIC_MODEL, max_tokens=max_tokens, system=system_prompt, messages=messages,
#             )
#             return "".join(b.text for b in resp.content if hasattr(b, "text"))

#         # gemini — translate {"role","content"} history into its Content objects,
#         # mapping "assistant" -> "model" (Gemini's naming for the same role)
#         from google import genai
#         from google.genai import types
#         api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
#         client = genai.Client(api_key=api_key)
#         contents = [
#             types.Content(role=("model" if m["role"] == "assistant" else "user"),
#                           parts=[types.Part.from_text(text=m["content"])])
#             for m in messages
#         ]
#         resp = client.models.generate_content(
#             model=GEMINI_MODEL, contents=contents,
#             config=types.GenerateContentConfig(
#                 system_instruction=system_prompt, max_output_tokens=max_tokens,
#                 thinking_config=types.ThinkingConfig(thinking_budget=0),  # see call_text() for why
#             ),
#         )
#         return resp.text or ""

#     return _with_retries(_run)


# def call_vision(system_prompt: str, user_text: str, images: list, max_tokens: int = 900) -> str:
#     """Vision completion. `images` is a list of (bytes, media_type) tuples,
#     e.g. [(png_bytes, "image/png")] — supports multiple images (e.g. PDF
#     pages) in one call for both providers. Retries transient failures."""
#     provider = get_provider()

#     def _run():
#         if provider == "anthropic":
#             import anthropic, base64
#             client = anthropic.Anthropic()
#             content = [{"type": "image", "source": {"type": "base64", "media_type": mt, "data": base64.b64encode(b).decode()}}
#                        for b, mt in images]
#             content.append({"type": "text", "text": user_text})
#             resp = client.messages.create(
#                 model=ANTHROPIC_MODEL, max_tokens=max_tokens, system=system_prompt,
#                 messages=[{"role": "user", "content": content}],
#             )
#             return "".join(b.text for b in resp.content if hasattr(b, "text"))

#         # gemini
#         from google import genai
#         from google.genai import types
#         api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
#         client = genai.Client(api_key=api_key)
#         parts = [types.Part.from_bytes(data=b, mime_type=mt) for b, mt in images]
#         parts.append(types.Part.from_text(text=user_text))
#         resp = client.models.generate_content(
#             model=GEMINI_MODEL,
#             contents=[types.Content(role="user", parts=parts)],
#             config=types.GenerateContentConfig(
#                 system_instruction=system_prompt, max_output_tokens=max_tokens,
#                 thinking_config=types.ThinkingConfig(thinking_budget=0),  # see call_text() for why
#             ),
#         )
#         return resp.text or ""

#     return _with_retries(_run)



"""
SafeMedTwin — Unified LLM Provider
=====================================
A single abstraction over Claude (Anthropic) and Gemini (Google), so every
module that needs an LLM (chatbot, diet personalization, report scanning,
LLM-vs-SLM comparison) calls one function instead of importing a specific
vendor SDK directly.

Provider selection (in order of priority):
1. SAFEMEDTWIN_LLM_PROVIDER env var — "anthropic" or "gemini", if set
2. Auto-detected: if ANTHROPIC_API_KEY is set, use Claude; else if
   GEMINI_API_KEY or GOOGLE_API_KEY is set, use Gemini
3. Neither set — calls raise a clear RuntimeError, which every caller in
   this project already catches and falls back to an offline default.

Both providers are given equivalent prompts and equivalent safety framing
(no diagnoses, no dosages) — see each call site for the actual prompt
text; this module only handles the "how do I call whichever provider is configured"
part.
"""

import os
import re
import socket
import time


# Transient failures worth retrying rather than surfacing to the user:
# 503/429/502/504 from the provider (overloaded or rate-limited), and
# DNS/connection blips on the client side.
_TRANSIENT_PATTERNS = re.compile(
    r"\b(503|502|504|429|UNAVAILABLE|RESOURCE_EXHAUSTED|overloaded|high demand|"
    r"rate.?limit|timeout|timed out|getaddrinfo|temporarily)\b",
    re.IGNORECASE,
)


def _is_transient(exc: Exception) -> bool:
    if isinstance(
        exc,
        (socket.gaierror, socket.timeout, ConnectionError, TimeoutError),
    ):
        return True

    return bool(_TRANSIENT_PATTERNS.search(str(exc)))


def _with_retries(fn, attempts: int = 1, base_delay: float = 1.5):
    """Run fn(), retrying transient failures with exponential backoff.

    Re-raises the last exception if every attempt fails, so callers keep
    their existing graceful-fallback behavior.
    """
    last_exc = None

    for attempt in range(attempts):
        try:
            return fn()

        except Exception as e:
            last_exc = e

            if not _is_transient(e) or attempt == attempts - 1:
                raise

            time.sleep(base_delay * (2 ** attempt))

    raise last_exc


def friendly_error(exc: Exception) -> str:
    """Turn a raw provider/network exception into something a user can act
    on. Used by call sites for their fallback messages.
    """

    text = str(exc)

    if isinstance(exc, socket.gaierror) or "getaddrinfo" in text:
        return (
            "Couldn't reach the AI service — this looks like a network/DNS "
            "issue on this machine, not a problem with your API key. Check "
            "your internet connection (campus Wi-Fi and VPNs often block "
            "this) and try again."
        )

    if _TRANSIENT_PATTERNS.search(text):
        return (
            "The AI service is temporarily busy or rate-limited (common on "
            "Gemini's free tier). This usually clears up on its own — please "
            "try again in a moment."
        )

    if "api" in text.lower() and "key" in text.lower():
        return (
            "The AI service rejected the API key. Check that GEMINI_API_KEY "
            "(or ANTHROPIC_API_KEY) is set correctly and is still valid."
        )

    return f"Couldn't reach the AI service: {exc}"


def get_provider() -> str:
    explicit = os.environ.get("SAFEMEDTWIN_LLM_PROVIDER", "").strip().lower()

    if explicit in ("anthropic", "gemini"):
        return explicit

    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"

    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return "gemini"

    raise RuntimeError(
        "No LLM provider configured. Set ANTHROPIC_API_KEY, or "
        "GEMINI_API_KEY (or GOOGLE_API_KEY), or explicitly set "
        "SAFEMEDTWIN_LLM_PROVIDER."
    )


ANTHROPIC_MODEL = os.environ.get(
    "SAFEMEDTWIN_ANTHROPIC_MODEL",
    "claude-sonnet-5",
)

GEMINI_MODEL = os.environ.get(
    "SAFEMEDTWIN_GEMINI_MODEL",
    "gemini-flash-latest",
)


def call_text(
    system_prompt: str,
    user_message: str,
    max_tokens: int = 600,
) -> str:
    """Plain text-in, text-out completion."""

    provider = get_provider()

    def _run():

        # ---------------------------------------------------------
        # Anthropic
        # ---------------------------------------------------------
        if provider == "anthropic":
            import anthropic

            client = anthropic.Anthropic()

            resp = client.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": user_message,
                    }
                ],
            )

            return "".join(
                b.text
                for b in resp.content
                if hasattr(b, "text")
            )

        # ---------------------------------------------------------
        # Gemini
        # ---------------------------------------------------------
        from google import genai
        from google.genai import types

        api_key = (
            os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )

        client = genai.Client(api_key=api_key)

        resp = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[user_message],
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=max_tokens,
            ),
        )

        return resp.text or ""

    return _with_retries(_run)


def call_chat(
    system_prompt: str,
    messages: list,
    max_tokens: int = 600,
) -> str:
    """Multi-turn chat.

    `messages` is a list of:
        {"role": "user"|"assistant", "content": str}

    Most recent message should be last.
    """

    provider = get_provider()

    def _run():

        # ---------------------------------------------------------
        # Anthropic
        # ---------------------------------------------------------
        if provider == "anthropic":
            import anthropic

            client = anthropic.Anthropic()

            resp = client.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=messages,
            )

            return "".join(
                b.text
                for b in resp.content
                if hasattr(b, "text")
            )

        # ---------------------------------------------------------
        # Gemini
        # ---------------------------------------------------------
        from google import genai
        from google.genai import types

        api_key = (
            os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )

        client = genai.Client(api_key=api_key)

        contents = [
            types.Content(
                role=(
                    "model"
                    if m["role"] == "assistant"
                    else "user"
                ),
                parts=[
                    types.Part.from_text(
                        text=m["content"]
                    )
                ],
            )
            for m in messages
        ]

        resp = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=max_tokens,
            ),
        )

        return resp.text or ""

    return _with_retries(_run)


def call_vision(
    system_prompt: str,
    user_text: str,
    images: list,
    max_tokens: int = 900,
) -> str:
    """Vision completion.

    `images` is a list of:
        (bytes, media_type)

    Example:
        [(png_bytes, "image/png")]

    Supports multiple images/pages.
    """

    provider = get_provider()

    def _run():

        # ---------------------------------------------------------
        # Anthropic
        # ---------------------------------------------------------
        if provider == "anthropic":
            import anthropic
            import base64

            client = anthropic.Anthropic()

            content = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mt,
                        "data": base64.b64encode(b).decode(),
                    },
                }
                for b, mt in images
            ]

            content.append(
                {
                    "type": "text",
                    "text": user_text,
                }
            )

            resp = client.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": content,
                    }
                ],
            )

            return "".join(
                b.text
                for b in resp.content
                if hasattr(b, "text")
            )

        # ---------------------------------------------------------
        # Gemini
        # ---------------------------------------------------------
        from google import genai
        from google.genai import types

        api_key = (
            os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )

        client = genai.Client(api_key=api_key)

        parts = [
            types.Part.from_bytes(
                data=b,
                mime_type=mt,
            )
            for b, mt in images
        ]

        parts.append(
            types.Part.from_text(
                text=user_text
            )
        )

        resp = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                types.Content(
                    role="user",
                    parts=parts,
                )
            ],
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=max_tokens,
            ),
        )

        return resp.text or ""

    return _with_retries(_run)
