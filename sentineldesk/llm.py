"""LLM access: Claude through the official Anthropic SDK, plus an offline mock.

If there is no key (or LLM_MODE=mock) the mock runs,
so tests, evals and the dashboard all work offline. Both return the same thing: a dict
parsed from the model's JSON output (see generate.DRAFT_SCHEMA).
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional


class LLMError(Exception):
    pass


class ClaudeLLM:
    name = "anthropic"

    def __init__(self, api_key: str, model: str, effort: str = "low", max_tokens: int = 4000) -> None:
        try:
            import anthropic  # imported lazily so mock mode needs no install
        except ImportError as e:  # pragma: no cover
            raise LLMError("pip install anthropic") from e
        self._anthropic = anthropic
        self.client = anthropic.Anthropic(api_key=api_key, max_retries=1, timeout=45.0)
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens

    def complete_json(self, system: str, user: str, schema: dict) -> dict:
        """One request; the reply is constrained to `schema` via structured outputs."""
        a = self._anthropic
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": schema}},
            )
        except a.RateLimitError as e:
            raise LLMError(f"rate limited: {e}") from e
        except a.APIStatusError as e:
            raise LLMError(f"API error {e.status_code}: {e.message}") from e
        except a.APIConnectionError as e:
            raise LLMError(f"connection error: {e}") from e
        if resp.stop_reason == "refusal":
            raise LLMError("model declined the request")
        if resp.stop_reason == "max_tokens":
            raise LLMError("draft was cut off (max_tokens)")
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        try:
            return json.loads(text)
        except ValueError as e:
            raise LLMError(f"model returned invalid JSON: {text[:200]}") from e


class GeminiLLM:
    """Gemini through the official google-genai SDK (Google Cloud / hackathon credits).

    Same contract as ClaudeLLM: complete_json() returns a dict constrained to the JSON schema.
    If the configured model id isn't available to the key, the newest "flash" model it can use is picked.
    """

    name = "gemini"

    RETRYABLE = {429, 500, 503, 504}

    def __init__(self, api_key: str, model: str = "gemini-3.8-flash", thinking_level: str = "low",
                 fallbacks: Optional[list] = None, timeout_s: float = 30.0) -> None:
        try:
            from google import genai
            from google.genai import errors, types
        except ImportError as e:  # pragma: no cover
            raise LLMError("pip install google-genai") from e
        self._types, self._errors = types, errors
        # one attempt per model, short timeout: on overload we switch model instead of waiting
        self.client = genai.Client(api_key=api_key, http_options=types.HttpOptions(
            timeout=int(timeout_s * 1000), retry_options=types.HttpRetryOptions(attempts=1)))
        self.model = model
        self.fallbacks = [m for m in (fallbacks or []) if m and m != model]
        self.thinking_level = thinking_level

    def _config(self, system: str, schema: dict):
        t = self._types
        cfg = dict(system_instruction=system, response_mime_type="application/json", response_json_schema=schema)
        if self.thinking_level:
            cfg["thinking_config"] = t.ThinkingConfig(thinking_level=self.thinking_level)
        return t.GenerateContentConfig(**cfg)

    def _call(self, model: str, system: str, user: str, schema: dict):
        try:
            return self.client.models.generate_content(model=model, contents=user, config=self._config(system, schema))
        except self._errors.ClientError as e:
            if "thinking" in str(e).lower() and self.thinking_level:  # model without thinking levels
                self.thinking_level = ""
                return self.client.models.generate_content(model=model, contents=user, config=self._config(system, schema))
            raise

    def _fallback_model(self) -> Optional[str]:
        names = [m.name.removeprefix("models/") for m in self.client.models.list()
                 if "generateContent" in (getattr(m, "supported_actions", None) or [])]
        # stable "gemini-<version>-flash" only (no preview/lite/image), newest version first
        stable = [(tuple(int(p) for p in m.group(1).split(".")), n)
                  for n in names if (m := re.fullmatch(r"gemini-(\d+(?:\.\d+)?)-flash", n))]
        return max(stable)[1] if stable else None

    def complete_json(self, system: str, user: str, schema: dict) -> dict:
        chain = [self.model] + self.fallbacks
        last: Exception | None = None
        resp = None
        for model in chain:
            try:
                resp = self._call(model, system, user, schema)
                self.last_model = model
                break
            except self._errors.APIError as e:
                last = e
                code = getattr(e, "code", None)
                if code == 404 and model == self.model:
                    alt = self._fallback_model()
                    if alt and alt not in chain:
                        chain.append(alt)
                    continue
                if code in self.RETRYABLE:
                    continue  # overloaded / rate-limited / slow: try the next model
                raise LLMError(f"Gemini API error {code}: {str(e)[:300]}") from e
            except Exception as e:  # timeouts, network errors
                last = e
                continue
        if resp is None:
            raise LLMError(f"all Gemini models failed ({', '.join(chain)}): {str(last)[:200]}")
        text = resp.text or ""
        try:
            return json.loads(text)
        except ValueError as e:
            raise LLMError(f"Gemini returned invalid JSON: {text[:200]}") from e


class MockLLM:
    """Deterministic stand-in: builds a reply from the top retrieved answer. Not a substitute for Claude."""

    name = "mock"
    model = "mock-rules"

    def complete_json(self, system: str, user: str, schema: dict) -> dict:
        try:
            payload = json.loads(user[user.index("{"):])
        except ValueError:
            payload = {}
        lang = payload.get("reply_language", "en")
        cases = payload.get("retrieved_tickets") or []
        greeting, closing = ("Hallo,", "Viele Grüße,\nIhr Support-Team") if lang == "de" else ("Hello,", "Best regards,\nSupport Team")
        if not cases:
            return {"reply": f"{greeting}\n\n{closing}", "claims": [], "unsupported_requests": []}
        top = cases[0]
        first = _first_sentence(top.get("answer", ""))
        intro = ("Vielen Dank für Ihre Nachricht. In einem ähnlichen Fall hat Folgendes geholfen:" if lang == "de"
                 else "Thank you for reaching out. In a similar case, the following helped:")
        return {
            "reply": f"{greeting}\n\n{intro} {first} [{top['ticket_id']}]\n\n{closing}",
            "claims": [{"sentence": first, "ticket_id": top["ticket_id"], "excerpt": first}],
            "unsupported_requests": [],
        }


def _first_sentence(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    m = re.match(r"(.{20,300}?[.!?])(\s|$)", text)
    return m.group(1) if m else text[:300]


class FallbackLLM:
    """Primary provider, then a backup provider when the primary fails or times out (e.g. Gemini -> Claude)."""

    def __init__(self, primary: Any, backup: Any) -> None:
        self.primary, self.backup, self.active = primary, backup, primary

    @property
    def name(self) -> str:
        return self.active.name

    @property
    def model(self) -> str:
        return self.active.model

    @property
    def last_model(self) -> str:
        return getattr(self.active, "last_model", None) or self.active.model

    def complete_json(self, system: str, user: str, schema: dict) -> dict:
        try:
            self.active = self.primary
            return self.primary.complete_json(system, user, schema)
        except LLMError:
            self.active = self.backup
            return self.backup.complete_json(system, user, schema)


def make_llm(settings: Any) -> Any:
    if settings.llm_mode == "gemini":
        try:
            gem = GeminiLLM(settings.gemini_key, settings.gemini_model, settings.gemini_thinking,
                            settings.gemini_fallbacks, timeout_s=12.0 if settings.anthropic_key else 30.0)
            if settings.anthropic_key:  # stalled Gemini call -> Claude instead of a 30 s wait
                try:
                    return FallbackLLM(gem, ClaudeLLM(settings.anthropic_key, settings.model, settings.effort))
                except LLMError:
                    pass
            return gem
        except LLMError:
            pass
    if settings.llm_mode == "anthropic":
        try:
            return ClaudeLLM(settings.anthropic_key, settings.model, settings.effort)
        except LLMError:
            pass
    return MockLLM()
