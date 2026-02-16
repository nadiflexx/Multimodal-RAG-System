from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from rag.chain.cache import SemanticCache
from rag.chain.contextualizer import QueryContextualizer
from rag.chain.memory import ChatState
from rag.chain.router import SemanticRouter


# ─── MEMORY ───
def test_chat_state():
    state = ChatState(window_size=2)
    state.add_user_message("1")
    state.add_ai_message("2")
    state.add_user_message("3")

    hist = state.get_history()
    assert len(hist) == 2
    assert hist[0]["content"] == "2"
    assert hist[1]["content"] == "3"

    state.clear()
    assert len(state.get_history()) == 0


# ─── ROUTER ───
def test_router():
    with patch("rag.chain.router.LLMFactory"):
        router = SemanticRouter()
        # Mockear la cadena completa
        router.chain = MagicMock()

        # Test Search
        router.chain.invoke.return_value = "SEARCH"
        assert router.route("query") == "SEARCH"

        # Test Fallback
        router.chain.invoke.return_value = "UNKNOWN"
        assert router.route("weird") == "SEARCH"


# ─── CONTEXTUALIZER ───
def test_contextualizer():
    with patch("rag.chain.contextualizer.LLMFactory"):
        ctx = QueryContextualizer()

        # No history
        assert ctx.contextualize("query", []) == "query"

        # With history
        ctx.chain = MagicMock()
        ctx.chain.invoke.return_value = "Contextualized query"
        assert (
            ctx.contextualize("it", [{"role": "user", "content": "prev"}])
            == "Contextualized query"
        )

        # Guardrail (too long)
        ctx.chain.invoke.return_value = "a" * 1000
        short_query = "a"
        assert ctx.contextualize(short_query, ["hist"]) == short_query


# ─── CACHE ───
@pytest.fixture
def mock_cache_embeddings():
    emb = MagicMock()
    # Mock embedding return as list of floats
    emb.embed_query.return_value = [1.0, 0.0]
    return emb


def test_cache_hit_miss(mock_cache_embeddings):
    cache = SemanticCache(mock_cache_embeddings, threshold=0.9)

    # FIX: Mockear _intent_chain explícitamente
    cache._intent_chain = MagicMock()
    cache._intent_chain.invoke.return_value = "intent"

    # PUT
    cache.put("original", "response", [Document("doc")])

    # GET HIT (Same vector [1.0, 0.0])
    res = cache.get("original")
    assert res is not None
    assert res[1] == "response"

    # GET MISS (Different vector)
    mock_cache_embeddings.embed_query.return_value = [0.0, 1.0]  # Orthogonal
    assert cache.get("other") is None


def test_cache_persistence(mock_cache_embeddings, tmp_path):
    json_file = tmp_path / "cache.json"

    # Save
    cache = SemanticCache(mock_cache_embeddings, persist_path=str(json_file))
    # FIX: Mockear _intent_chain explícitamente
    cache._intent_chain = MagicMock()
    cache._intent_chain.invoke.return_value = "intent"

    cache.put("q", "r", [Document("d")])

    assert json_file.exists()

    # Load
    cache2 = SemanticCache(mock_cache_embeddings, persist_path=str(json_file))
    assert len(cache2.entries) == 1

    # Invalidate
    cache2.invalidate()
    assert len(cache2.entries) == 0
    assert not json_file.exists()
