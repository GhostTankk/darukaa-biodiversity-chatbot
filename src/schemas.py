"""
Structured data contracts for the system.

Everything the reasoning engine consumes or produces is typed here so that
(a) partial/incomplete user input is easy to detect (missing fields -> None),
and (b) the final chatbot output has a predictable, checkable shape instead
of "just prose from an LLM".
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SiteMetrics(BaseModel):
    """
    The environmental variable slots the system tries to fill before it will
    reason confidently. Every field is optional because a real user will
    almost never provide all of them up front -- ConversationManager decides
    which missing fields are worth asking a clarifying question about.
    """

    soil_organic_carbon_pct: float | None = Field(
        default=None, description="Soil organic carbon, percent by weight"
    )
    soil_ph: float | None = None
    soil_moisture: Literal["low", "moderate", "high"] | None = None

    land_use: str | None = Field(
        default=None,
        description="e.g. monoculture, intercropped, agroforestry, cropland, pasture",
    )
    region_type: Literal["arid", "semi-arid", "temperate", "tropical", "other"] | None = None

    species_richness: float | None = Field(
        default=None, description="Normalized 0-1 index, or raw species count if that's what's available"
    )
    habitat_diversity_index: float | None = Field(default=None, description="Normalized 0-1 index")

    rainfall: Literal["low", "moderate", "high", "erratic"] | None = None
    temperature_anomaly_c: float | None = Field(
        default=None, description="Deviation from local historical baseline, in Celsius"
    )

    pollution_level: Literal["low", "moderate", "high"] | None = None
    deforestation_rate: Literal["none", "low", "moderate", "high"] | None = None

    latitude: float | None = None
    longitude: float | None = None

    def known_fields(self) -> dict:
        return {k: v for k, v in self.model_dump().items() if v is not None}

    def missing_core_fields(self) -> list[str]:
        """
        The knowledge base can't retrieve anything useful without at least
        SOME signal on soil, land use, and climate. This lists which of the
        'core three' categories are still empty -- used to drive clarifying
        questions (see ConversationManager).
        """
        core = {
            "soil": ["soil_organic_carbon_pct", "soil_ph", "soil_moisture"],
            "land_use": ["land_use"],
            "climate": ["rainfall", "region_type", "temperature_anomaly_c"],
        }
        missing = []
        for category, fields in core.items():
            if not any(getattr(self, f) is not None for f in fields):
                missing.append(category)
        return missing


class KnowledgeChunk(BaseModel):
    """A single retrieved piece of evidence, whether from the curated JSON
    knowledge base or from an ingested PDF."""

    id: str
    text: str
    source: str
    score: float


class Source(BaseModel):
    name: str
    org: str


class Recommendation(BaseModel):
    """
    The mandatory shape for every recommendation the system emits (per the
    challenge brief: what to do / why / which metric / which reference).
    """

    action: str
    reasoning: str
    impacted_metrics: list[str]
    expected_effect: str
    time_horizon: Literal["short", "medium", "long"]
    confidence: Literal["low", "medium", "high"]
    sources: list[Source]


class ChatbotResponse(BaseModel):
    clarifying_question: str | None = None
    recommendations: list[Recommendation] = Field(default_factory=list)
    narrative: str = Field(
        default="", description="Short natural-language framing to show above the structured cards"
    )
