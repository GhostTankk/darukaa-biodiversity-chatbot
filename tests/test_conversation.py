from src.conversation import ConversationManager, extract_metrics_from_text
from src.schemas import SiteMetrics


def test_extracts_soc_and_ph_from_free_text():
    metrics = extract_metrics_from_text(
        "Soil organic carbon is 0.3% and pH is 6.5, semi-arid region, low rainfall, monoculture wheat",
        SiteMetrics(),
    )
    assert metrics.soil_organic_carbon_pct == 0.3
    assert metrics.soil_ph == 6.5
    assert metrics.region_type == "semi-arid"
    assert metrics.rainfall == "low"
    assert metrics.land_use == "monoculture"


def test_asks_clarifying_question_when_core_fields_missing():
    convo = ConversationManager()
    _metrics, clarify = convo.ingest_user_message("Biodiversity is declining on my land")
    assert clarify is not None
    assert "soil" in clarify.lower() or "carbon" in clarify.lower() or "ph" in clarify.lower()


def test_does_not_ask_twice_in_a_row():
    convo = ConversationManager()
    convo.ingest_user_message("Biodiversity is declining on my land")
    # Second turn still vague -- should proceed anyway rather than stalling forever
    _, clarify2 = convo.ingest_user_message("Not sure, it's just getting worse")
    assert clarify2 is None


def test_structured_input_merges_into_metrics():
    convo = ConversationManager()
    convo.ingest_structured_input({"soil_organic_carbon_pct": 0.4, "land_use": "monoculture"})
    assert convo.metrics.soil_organic_carbon_pct == 0.4
    assert convo.metrics.land_use == "monoculture"
