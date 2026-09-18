"""
The only component that calls out to an LLM. Everything upstream (retrieval,
slot-filling) is deterministic and free; this is where retrieved evidence +
conversation context get turned into grounded, structured recommendations.

Design choice: we do NOT let the model free-associate recommendations from
its own training data. It is instructed to reason ONLY over the retrieved
knowledge chunks passed in, and to connect at least two of them when
possible (multi-metric reasoning is a hard scoring criterion). It must
return strict JSON matching ChatbotResponse so the app layer never has to
regex-parse prose.
"""

from __future__ import annotations

import json
import os
import time

from google import genai
from google.genai import errors, types

from .schemas import ChatbotResponse, KnowledgeChunk, SiteMetrics

# Google deprecated 2.5 Flash for new API users; gemini-3.6-flash (released
# July 2026) is the current Flash-tier model with free-tier access. If this
# 404s again later, check https://aistudio.google.com for the current
# free-tier model name.
# OLD:
MODEL = "gemini-3.6-flash"
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2.0

# NEW:
MODEL_CHAIN = ["gemini-3.6-flash", "gemini-flash-lite-latest"]
RETRIES_PER_MODEL = 2
RETRY_BACKOFF_SECONDS = 2.0

# Free-tier Flash models occasionally return 503 ("high demand") under load.
# These are transient -- worth a short backoff-and-retry rather than
# surfacing a crash to the user on the first hiccup.
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2.0

SYSTEM_PROMPT = """You are an AI environmental scientist embedded in a biodiversity advisory \
system for Darukaa.Earth. You do not chat like a generic assistant: every recommendation you \
make must be grounded ONLY in the evidence chunks provided to you in the user turn -- never \
invent a statistic or source that is not present in the evidence.

Rules:
1. Use only the provided evidence chunks as the basis for recommendations. If the evidence is \
insufficient, say so in `narrative` rather than fabricating a recommendation.
2. Wherever the site data and evidence allow it, connect at least two different environmental \
variables in your reasoning (e.g. soil health <-> biodiversity, water <-> species survival, \
land use <-> fragmentation). Single-variable, generic answers ("use sustainable practices") \
are explicitly disallowed.
3. Every recommendation must include: action, reasoning (mechanistic, not vague), the specific \
impacted_metrics, expected_effect (quantified where the evidence gives you a number), \
time_horizon (short/medium/long), confidence (low/medium/high), and sources (name + org) \
copied from the evidence chunk(s) you used.
4. Return ONLY valid JSON matching this shape, nothing else -- no markdown fences, no preamble:
{
  "clarifying_question": null,
  "narrative": "1-3 sentences framing the situation",
  "recommendations": [
    {
      "action": "...",
      "reasoning": "...",
      "impacted_metrics": ["..."],
      "expected_effect": "...",
      "time_horizon": "short|medium|long",
      "confidence": "low|medium|high",
      "sources": [{"name": "...", "org": "..."}]
    }
  ]
}
"""


def _format_evidence(chunks: list[KnowledgeChunk]) -> str:
    if not chunks:
        return "No relevant evidence was retrieved from the knowledge base."
    lines = []
    for c in chunks:
        lines.append(f"- [{c.id}] (relevance {c.score:.2f}) {c.text}\n  Source: {c.source}")
    return "\n".join(lines)


def _format_site_data(metrics: SiteMetrics) -> str:
    known = metrics.known_fields()
    if not known:
        return "No structured site data provided yet."
    return json.dumps(known, indent=2)


class ReasoningEngine:
    def __init__(self, api_key: str | None = None):
        self.client = genai.Client(api_key=api_key or os.environ.get("GEMINI_API_KEY"))

    def generate(
        self,
        user_message: str,
        site_metrics: SiteMetrics,
        evidence: list[KnowledgeChunk],
        conversation_context: str,
    ) -> ChatbotResponse:
        user_prompt = f"""Conversation so far:
{conversation_context}

Current known site data:
{_format_site_data(site_metrics)}

Retrieved evidence chunks (only use these):
{_format_evidence(evidence)}

Latest user message: {user_message}

Respond with the JSON object described in your instructions, and nothing else."""

        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            # Ask the API itself to constrain output to JSON, rather than
            # relying purely on prompt instructions -- this is a native
            # Gemini feature and cuts down on malformed-output retries.
            response_mime_type="application/json",
            max_output_tokens=2000,
            # We never pass tools, so automatic function calling has
            # nothing to do here -- disabling it silences the SDK's
            # "use Chat.send_message instead" warning on every call.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

        response = None
        last_error: Exception | None = None
        client_error: errors.ClientError | None = None

        for model in MODEL_CHAIN:
            for attempt in range(1, RETRIES_PER_MODEL + 1):
                try:
                    response = self.client.models.generate_content(
                        model=model, contents=user_prompt, config=config
                    )
                    break
                except errors.ServerError as e:
                    last_error = e
                    if attempt < RETRIES_PER_MODEL:
                        time.sleep(RETRY_BACKOFF_SECONDS * attempt)
                except errors.ClientError as e:
                    client_error = e
                    break
            if response is not None or client_error is not None:
                break

        if client_error is not None:
            return ChatbotResponse(
                narrative=f"The reasoning engine rejected the request (likely an API key or model-name issue): {client_error}"
            )

        if response is None:
            return ChatbotResponse(
                narrative=f"Gemini is temporarily overloaded (high demand on the free tier) across all fallback models and didn't respond. Please try again in a moment. (Last error: {last_error})"
            )

        text = (response.text or "").strip()
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

        try:
            data = json.loads(text)
            return ChatbotResponse(**data)
        except (json.JSONDecodeError, TypeError) as e:
            # Fail loudly but gracefully -- surface the raw text so it's debuggable
            # instead of silently returning an empty response.
            return ChatbotResponse(
                narrative=f"[reasoning engine returned unparseable output: {e}] Raw: {text[:500]}"
            )
