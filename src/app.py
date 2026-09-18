"""
Streamlit UI. This is what you deploy for the "live demo URL" submission
requirement (Streamlit Community Cloud is the easiest free option -- see
README).

Two input paths, per challenge requirement #5:
  - Free text chat (main panel)
  - Structured JSON / form input (sidebar) -- also where geo-coordinates
    (bonus requirement) are entered.
"""

from __future__ import annotations

import json

import streamlit as st
from dotenv import load_dotenv

from src.conversation import ConversationManager
from src.knowledge_base import KnowledgeBase
from src.reasoning_engine import ReasoningEngine

load_dotenv()  # local runs: reads .env for GEMINI_API_KEY. On Streamlit Cloud, use the Secrets panel instead (see README).

st.set_page_config(page_title="Darukaa.Earth Biodiversity Advisor", page_icon="🌱", layout="wide")


@st.cache_resource
def get_knowledge_base() -> KnowledgeBase:
    return KnowledgeBase()


@st.cache_resource
def get_reasoning_engine() -> ReasoningEngine:
    return ReasoningEngine()


if "convo" not in st.session_state:
    st.session_state.convo = ConversationManager()
if "display_history" not in st.session_state:
    st.session_state.display_history = []  # list of (role, content) for rendering

kb = get_knowledge_base()

st.title("🌱 Darukaa.Earth — AI Biodiversity Intelligence Chatbot")
st.caption(
    "Ask about a piece of land, its soil, climate, or biodiversity, and get evidence-backed, "
    "multi-metric recommendations -- not generic advice."
)

# --- Sidebar: structured input ---
with st.sidebar:
    st.header("Structured site data")
    st.caption("Fill in what you know. Anything left blank stays unknown; the chat can also fill these in from text.")

    with st.form("structured_form"):
        soc = st.number_input("Soil organic carbon (%)", min_value=0.0, max_value=20.0, value=0.0, step=0.1)
        ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=0.0, step=0.1)
        soil_moisture = st.selectbox("Soil moisture", ["", "low", "moderate", "high"])
        land_use = st.text_input("Land use (e.g. monoculture wheat)")
        region_type = st.selectbox("Region type", ["", "arid", "semi-arid", "temperate", "tropical", "other"])
        rainfall = st.selectbox("Rainfall", ["", "low", "moderate", "high", "erratic"])
        pollution_level = st.selectbox("Pollution level", ["", "low", "moderate", "high"])
        deforestation_rate = st.selectbox("Deforestation rate", ["", "none", "low", "moderate", "high"])
        lat = st.number_input("Latitude (optional)", value=0.0, format="%.5f")
        lon = st.number_input("Longitude (optional)", value=0.0, format="%.5f")
        submitted = st.form_submit_button("Apply structured data")

    if submitted:
        payload = {
            "soil_organic_carbon_pct": soc or None,
            "soil_ph": ph or None,
            "soil_moisture": soil_moisture or None,
            "land_use": land_use or None,
            "region_type": region_type or None,
            "rainfall": rainfall or None,
            "pollution_level": pollution_level or None,
            "deforestation_rate": deforestation_rate or None,
            "latitude": lat or None,
            "longitude": lon or None,
        }
        st.session_state.convo.ingest_structured_input(payload)
        st.success("Structured data applied. Ask a question in the chat to get recommendations.")

    st.divider()
    st.subheader("Or paste raw JSON")
    raw_json = st.text_area("JSON matching SiteMetrics fields", height=120)
    if st.button("Apply JSON"):
        try:
            payload = json.loads(raw_json)
            st.session_state.convo.ingest_structured_input(payload)
            st.success("JSON applied.")
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")

    st.divider()
    with st.expander("Current known site metrics"):
        st.json(st.session_state.convo.metrics.known_fields())

# --- Main panel: chat ---
for role, content in st.session_state.display_history:
    with st.chat_message(role):
        st.markdown(content)

user_message = st.chat_input("Describe your land, ecosystem, or ask a question...")

if user_message:
    st.session_state.display_history.append(("user", user_message))
    with st.chat_message("user"):
        st.markdown(user_message)

    metrics, clarify = st.session_state.convo.ingest_user_message(user_message)

    with st.chat_message("assistant"):
        if clarify:
            st.markdown(clarify)
            st.session_state.display_history.append(("assistant", clarify))
        else:
            with st.spinner("Retrieving evidence and reasoning..."):
                evidence = kb.search(user_message, top_k=4)
                engine = get_reasoning_engine()
                try:
                    response = engine.generate(
                        user_message=user_message,
                        site_metrics=metrics,
                        evidence=evidence,
                        conversation_context=st.session_state.convo.context_window(),
                    )
                except Exception as e:  # noqa: BLE001 -- keep the session alive on any unexpected failure
                    st.error(f"Something went wrong on that request: {e}")
                    st.stop()
            st.session_state.convo.record_assistant_reply(response.narrative)

            rendered = []
            if response.narrative:
                st.markdown(response.narrative)
                rendered.append(response.narrative)

            for i, rec in enumerate(response.recommendations, 1):
                with st.container(border=True):
                    st.markdown(f"**{i}. {rec.action}**")
                    st.markdown(f"*Why:* {rec.reasoning}")
                    cols = st.columns(3)
                    cols[0].metric("Time horizon", rec.time_horizon)
                    cols[1].metric("Confidence", rec.confidence)
                    cols[2].markdown("**Impacts:** " + ", ".join(rec.impacted_metrics))
                    st.markdown(f"*Expected effect:* {rec.expected_effect}")
                    srcs = " • ".join(f"{s.name} ({s.org})" for s in rec.sources)
                    st.caption(f"Sources: {srcs}")
                rendered.append(
                    f"**{i}. {rec.action}**\n\n{rec.reasoning}\n\nImpacts: {', '.join(rec.impacted_metrics)}"
                )

            with st.expander("Retrieved evidence used for this answer"):
                for chunk in evidence:
                    st.markdown(f"- `{chunk.id}` (score {chunk.score:.2f}) — {chunk.source}")

            st.session_state.display_history.append(("assistant", "\n\n".join(rendered)))
