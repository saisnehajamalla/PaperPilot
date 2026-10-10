"""LLM provider abstraction (currently Groq, via the OpenAI-compatible SDK).

Every call returns an LLMResponse instead of raising, so callers (and the Streamlit UI)
never crash on a missing key, a timeout, a rate limit, or a malformed response.
"""

from dataclasses import dataclass

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    OpenAI,
    RateLimitError,
)

from config import GROQ_API_KEY, GROQ_BASE_URL, GROQ_MODEL

REQUEST_TIMEOUT_SECONDS = 30


@dataclass
class LLMResponse:
    ok: bool
    content: str | None = None
    # Short, user-safe message -- this is what the UI shows. Never contains a raw JSON
    # body, API key, or organization/account identifier.
    error: str | None = None
    # Full original exception text, kept only for developer diagnostics (e.g. future
    # logging). Never shown in the UI. None for non-provider failures (missing key, empty
    # response) where there's nothing more to add beyond `error` itself.
    raw_error: str | None = None


def _inner_provider_message(e: Exception) -> str:
    """Groq (like OpenAI) nests the real message one level down in the parsed error body
    (`{"error": {"message": ..., "type": ..., "code": ...}}`), and the openai SDK already
    unwraps that "error" key into `e.body` before the exception is constructed. Falls back
    to str(e) (which includes the raw dict) only if body isn't the dict we expect."""
    body = getattr(e, "body", None)
    if isinstance(body, dict):
        message = body.get("message")
        if isinstance(message, str):
            return message
    return str(e)


def translate_provider_error(e: Exception) -> str:
    """Maps a provider exception to a short, actionable, user-safe message.

    Never includes a raw JSON body, API key, or account/organization identifier -- only a
    generic category description. The full original exception text is preserved separately
    (see LLMResponse.raw_error) for developers, not shown here.

    HTTP 429 is deliberately not treated as always meaning "quota exhausted": Groq (and
    OpenAI-compatible providers generally) reuse 429/RateLimitError for both a short-lived
    tokens-per-minute throttle and a hard tokens-per-day quota exhaustion, and both report
    the *same* structured `code` ("rate_limit_exceeded") -- only the embedded message text
    differs ("... per day (TPD) ..." vs "... per minute (TPM) ..."). When that text doesn't
    clearly indicate one or the other, this falls back to the generic rate-limit message
    rather than guessing "quota exhausted".
    """
    if isinstance(e, APITimeoutError):
        return "The AI service took too long to respond. Please try again."
    if isinstance(e, APIConnectionError):
        return "Could not connect to the AI service. Check your connection and try again."
    if isinstance(e, AuthenticationError):
        return "The AI service rejected the configured API credentials. Check the application's configuration."
    if isinstance(e, RateLimitError):
        detail = _inner_provider_message(e).lower()
        if "per day" in detail or "tpd" in detail:
            return "The AI provider's usage quota has been reached. Please try again later."
        return "The AI service is temporarily rate-limited. Please wait before trying again."
    if isinstance(e, APIStatusError):
        if e.status_code == 413:
            return "This request exceeds the AI provider's processing limit. Try reducing the amount of content."
        return "The AI service returned an error and could not complete this request."
    return "An unexpected error occurred while contacting the AI service."


class LLMClient:
    def __init__(self, api_key: str = GROQ_API_KEY, base_url: str = GROQ_BASE_URL, model: str = GROQ_MODEL):
        self.model = model
        self._api_key = api_key
        self._base_url = base_url
        self._client: OpenAI | None = None

    def _get_client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=self._api_key, base_url=self._base_url)
        return self._client

    def _complete(self, system_prompt: str, user_prompt: str, temperature: float, json_mode: bool) -> LLMResponse:
        if not self._api_key:
            return LLMResponse(ok=False, error=(
                "GROQ_API_KEY is not set. Add it to your .env file. "
                "Get a free key at https://console.groq.com/keys"
            ))
        try:
            client = self._get_client()
            kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                timeout=REQUEST_TIMEOUT_SECONDS,
                **kwargs,
            )
            content = response.choices[0].message.content
            if not content:
                return LLMResponse(ok=False, error="The LLM returned an empty response.")
            return LLMResponse(ok=True, content=content)
        except Exception as e:  # last-resort guard: the UI must not crash on an LLM call
            return LLMResponse(ok=False, error=translate_provider_error(e), raw_error=str(e))

    def complete_json(self, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> LLMResponse:
        return self._complete(system_prompt, user_prompt, temperature, json_mode=True)

    def complete_text(self, system_prompt: str, user_prompt: str, temperature: float = 0.3) -> LLMResponse:
        return self._complete(system_prompt, user_prompt, temperature, json_mode=False)
