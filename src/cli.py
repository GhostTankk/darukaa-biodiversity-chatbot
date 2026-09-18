"""
Minimal terminal chat loop. Useful for fast local iteration and for
demonstrating the pipeline in a screen recording without setting up
Streamlit. Run with: python -m src.cli
"""

from __future__ import annotations

from dotenv import load_dotenv

from .conversation import ConversationManager
from .knowledge_base import KnowledgeBase
from .reasoning_engine import ReasoningEngine

load_dotenv()  # reads .env in the project root into os.environ; must run before ReasoningEngine() reads GEMINI_API_KEY


def format_response(resp) -> str:
    lines = []
    if resp.narrative:
        lines.append(resp.narrative)
    if resp.clarifying_question:
        lines.append(f"\n> {resp.clarifying_question}")
    for i, rec in enumerate(resp.recommendations, 1):
        lines.append(f"\n[{i}] {rec.action}")
        lines.append(f"    Why: {rec.reasoning}")
        lines.append(f"    Impacts: {', '.join(rec.impacted_metrics)}")
        lines.append(f"    Expected effect: {rec.expected_effect}")
        lines.append(f"    Time horizon: {rec.time_horizon} | Confidence: {rec.confidence}")
        srcs = "; ".join(f"{s.name} ({s.org})" for s in rec.sources)
        lines.append(f"    Sources: {srcs}")
    return "\n".join(lines)


def main():
    print("Darukaa.Earth Biodiversity Advisor (CLI) -- type 'exit' to quit.\n")
    kb = KnowledgeBase()
    convo = ConversationManager()
    engine = ReasoningEngine()

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break

        metrics, clarify = convo.ingest_user_message(user_input)
        if clarify:
            print(f"\nAssistant: {clarify}\n")
            continue

        evidence = kb.search(user_input, top_k=4)
        try:
            response = engine.generate(
                user_message=user_input,
                site_metrics=metrics,
                evidence=evidence,
                conversation_context=convo.context_window(),
            )
        except Exception as e:  # noqa: BLE001 -- last-resort net so one bad turn never kills the session
            print(f"\nAssistant: Something went wrong on that request ({e}). Let's try again.\n")
            continue
        convo.record_assistant_reply(response.narrative)
        print(f"\nAssistant:\n{format_response(response)}\n")


if __name__ == "__main__":
    main()
