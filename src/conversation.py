"""
Handles multi-turn state: what the user has told us so far (SiteMetrics),
the raw chat history (for the LLM's context), and deciding whether we have
enough signal to reason, or whether we should ask a clarifying question
first (per challenge requirement #2).
"""

from __future__ import annotations

import re

from .schemas import SiteMetrics

CLARIFYING_QUESTIONS = {
    "soil": "Could you share soil organic carbon %, pH, or a rough moisture level (low/moderate/high)?",
    "land_use": "What's the current land use? (e.g. monoculture cropland, intercropped, agroforestry, pasture)",
    "climate": "What's the rainfall pattern (low/moderate/high/erratic) and region type (arid/semi-arid/temperate/tropical)?",
}

# Very lightweight keyword/regex extraction so the system can pull structured
# slots out of free text without needing an LLM call just for parsing.
# This is intentionally simple -- swap in an LLM-based extractor if you want
# more robust free-text understanding.
_PATTERNS = {
    "soil_organic_carbon_pct": r"(?:soil organic carbon|soc)\D{0,10}(\d+(?:\.\d+)?)\s*%",
    "soil_ph": r"\bph\D{0,5}(\d+(?:\.\d+)?)",
    "temperature_anomaly_c": r"(\+?\d+(?:\.\d+)?)\s*(?:degrees?|°c|c)\b.*(?:warmer|anomaly|above)",
}

_KEYWORD_FIELDS = {
    "rainfall": {"low": ["low rainfall", "drought", "dry"], "erratic": ["erratic rainfall", "unpredictable rain"],
                 "high": ["high rainfall", "heavy rain"], "moderate": ["moderate rainfall"]},
    # Order matters: "semi-arid" must be checked before the bare "arid" keyword,
    # since "arid" is a substring of "semi-arid" and dicts preserve insertion order.
    "region_type": {"semi-arid": ["semi-arid", "semiarid"], "arid": ["arid"], "temperate": ["temperate"],
                     "tropical": ["tropical"]},
    "land_use": {"monoculture": ["monoculture", "mono-crop", "single crop"], "agroforestry": ["agroforestry"],
                 "pasture": ["pasture", "grazing"], "cropland": ["cropland", "farmland", "crop"]},
    "pollution_level": {"high": ["high pollution", "heavy pollution"], "moderate": ["some pollution"],
                         "low": ["low pollution", "clean"]},
    "deforestation_rate": {"high": ["heavy deforestation", "rapid deforestation"],
                            "moderate": ["some deforestation"], "none": ["no deforestation"]},
    "soil_moisture": {"low": ["dry soil", "low soil moisture"], "high": ["waterlogged", "high soil moisture"],
                       "moderate": ["moderate soil moisture"]},
}


def extract_metrics_from_text(text: str, existing: SiteMetrics) -> SiteMetrics:
    """Merges any newly-detected values from free text into the existing SiteMetrics."""
    data = existing.model_dump()
    lowered = text.lower()

    for field, pattern in _PATTERNS.items():
        if data.get(field) is not None:
            continue
        m = re.search(pattern, lowered)
        if m:
            data[field] = float(m.group(1))

    for field, options in _KEYWORD_FIELDS.items():
        if data.get(field) is not None:
            continue
        for value, keywords in options.items():
            if any(kw in lowered for kw in keywords):
                data[field] = value
                break

    return SiteMetrics(**data)


class ConversationManager:
    """
    One instance per user session. Holds accumulated SiteMetrics and raw
    chat turns. Call `ingest_user_message` each turn; it updates state and
    tells the caller whether to ask a clarifying question or proceed to
    retrieval + reasoning.
    """

    def __init__(self):
        self.metrics = SiteMetrics()
        self.history: list[tuple[str, str]] = []  # (role, text)
        self.turns_since_last_clarify = 0

    def ingest_user_message(self, text: str) -> tuple[SiteMetrics, str | None]:
        self.history.append(("user", text))
        self.metrics = extract_metrics_from_text(text, self.metrics)

        missing = self.metrics.missing_core_fields()
        # Don't clarify forever -- after one round of clarifying questions,
        # proceed with whatever we have so the conversation doesn't stall.
        if missing and self.turns_since_last_clarify == 0:
            self.turns_since_last_clarify += 1
            question = " ".join(CLARIFYING_QUESTIONS[cat] for cat in missing)
            self.history.append(("assistant_clarify", question))
            return self.metrics, question

        return self.metrics, None

    def ingest_structured_input(self, payload: dict) -> SiteMetrics:
        """For the JSON/structured-input path (challenge requirement #5)."""
        merged = {**self.metrics.model_dump(), **{k: v for k, v in payload.items() if v is not None}}
        self.metrics = SiteMetrics(**merged)
        return self.metrics

    def record_assistant_reply(self, text: str) -> None:
        self.history.append(("assistant", text))

    def context_window(self, max_turns: int = 8) -> str:
        """Simple recency-window memory for the LLM prompt."""
        recent = self.history[-max_turns:]
        return "\n".join(f"{role}: {text}" for role, text in recent)
