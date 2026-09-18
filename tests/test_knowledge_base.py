from src.knowledge_base import KnowledgeBase


def test_kb_loads_all_entries():
    kb = KnowledgeBase()
    assert len(kb.entries) >= 6
    assert kb.index.ntotal == len(kb.entries)


def test_search_returns_relevant_entry_for_low_soc_query():
    kb = KnowledgeBase()
    results = kb.search("my soil organic carbon is very low, monoculture wheat field", top_k=3)
    assert len(results) > 0
    # kb001 (soil_carbon / cover crops) should be highly ranked for this query
    top_ids = [r.id for r in results]
    assert "kb001" in top_ids


def test_search_returns_relevant_entry_for_semiarid_drought_query():
    kb = KnowledgeBase()
    results = kb.search("semi-arid region with low erratic rainfall, crops keep failing", top_k=3)
    top_ids = [r.id for r in results]
    assert "kb003" in top_ids
