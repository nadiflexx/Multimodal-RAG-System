from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from rag.exceptions import LLMError, RetrievalError
from rag.retrieval.hybrid import HybridRetriever, HybridSearchFactory
from rag.retrieval.hyde import HyDEGenerator
from rag.retrieval.parent import ParentDocumentStore
from rag.retrieval.reranker import Reranker
from rag.retrieval.vector_store import VectorStore


# ─── VECTOR STORE ───
def test_vector_store_methods(mock_embeddings):
    with patch("rag.retrieval.vector_store.Chroma") as mock_chroma:
        mock_db = mock_chroma.return_value
        mock_db._collection.count.return_value = 0

        vs = VectorStore("test_col", MagicMock())

        # Add documents
        vs.add_documents([Document("test")])
        mock_db.add_documents.assert_called()

        # Clear collection
        mock_db._collection.count.return_value = 5
        mock_db._collection.get.return_value = {"ids": ["1"]}
        vs.clear_collection()
        mock_db._collection.delete.assert_called_with(ids=["1"])

        # As retriever
        vs.as_retriever(filter={"page": 1})
        mock_db.as_retriever.assert_called()


# ─── HYBRID ───
def test_hybrid_retriever(sample_docs):
    mock_vs = MagicMock()
    mock_vs.as_retriever.return_value.invoke.return_value = sample_docs[:1]

    hybrid = HybridRetriever(mock_vs, sample_docs, vector_k=1, bm25_k=1)

    # Test direct retrieve
    results = hybrid.retrieve("query")
    assert len(results) > 0

    # Test hyde retrieve
    results_hyde = hybrid.retrieve_with_hyde("query", "hypo")
    assert len(results_hyde) > 0


def test_hybrid_factory(sample_docs):
    # FIX: Pasar documentos no vacíos para que BM25 no falle
    factory = HybridSearchFactory.create(MagicMock(), sample_docs, 5, 5)
    assert isinstance(factory, HybridRetriever)


# ─── HYDE ───
def test_hyde_generator():
    with patch("rag.retrieval.hyde.LLMFactory"):
        gen = HyDEGenerator()
        # FIX: Mockear gen.chain explícitamente
        gen.chain = MagicMock()
        gen.chain.invoke.return_value = "Hypothetical text"

        res = gen.generate("query")
        assert res == "Hypothetical text"


def test_hyde_error():
    with patch("rag.retrieval.hyde.LLMFactory"):
        gen = HyDEGenerator()
        # FIX: Mockear gen.chain antes de asignar side_effect
        gen.chain = MagicMock()
        gen.chain.invoke.side_effect = Exception("LLM fail")
        with pytest.raises(LLMError):
            gen.generate("fail")


# ─── RERANKER ───
def test_reranker_success(sample_docs):
    with patch("rag.retrieval.reranker.Ranker") as mock_ranker:
        instance = mock_ranker.return_value
        # Mock return from FlashRank
        instance.rerank.return_value = [
            {"id": "0", "text": "Doc 1 content", "meta": {}, "score": 0.9}
        ]

        reranker = Reranker()
        reranked = reranker.rerank("query", sample_docs)

        assert len(reranked) == 1
        assert reranked[0].metadata["relevance_score"] == 0.9


def test_reranker_init_fail():
    with patch("rag.retrieval.reranker.Ranker", side_effect=Exception("Load fail")):
        with pytest.raises(RetrievalError):
            Reranker()


# ─── PARENT ───
def test_parent_store(sample_docs):
    store = ParentDocumentStore()
    store.store_parents(sample_docs)

    # FIX: Ajustar el assert al contenido real del fixture sample_docs
    chunk = Document("chunk", metadata={"page": 1})

    # Get parent
    parent = store.get_parent(chunk)
    # sample_docs[0] tiene "Document 1 content about AI."
    assert parent.page_content == "Document 1 content about AI."

    # Get parents for chunks (with expansion)
    chunk_p2 = Document("chunk", metadata={"page": 2})
    parents = store.get_parents_for_chunks([chunk_p2], expand_neighbors=True)
    assert len(parents) == 3
