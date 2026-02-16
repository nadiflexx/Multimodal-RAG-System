from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from rag.exceptions import VectorStoreNotInitializedError
from rag.pipeline import RAGPipeline


@pytest.fixture
def pipeline_mocks():
    """Mocks de todos los componentes internos."""
    with (
        patch("rag.pipeline.LocalPyTorchEmbeddings"),
        patch("rag.pipeline.DocumentLoader") as mock_loader,
        patch("rag.pipeline.VectorStore") as mock_vs,
        patch("rag.pipeline.HybridSearchFactory") as mock_hsf,
        patch("rag.pipeline.Reranker") as mock_rerank,
        patch("rag.pipeline.LLMFactory") as mock_llm_factory,
        patch("rag.pipeline.SemanticRouter") as mock_router,
        patch("rag.pipeline.HyDEGenerator") as mock_hyde,
        patch("rag.pipeline.SemanticCache") as mock_cache,
    ):
        # Setup returns
        mock_loader.return_value.load_pdf.return_value = [Document("chunk")]
        mock_loader.return_value.load_pdf_full_pages.return_value = [Document("full")]

        mock_vs_instance = mock_vs.return_value
        mock_vs_instance.similarity_search.return_value = [
            Document("doc", metadata={"page": 1})
        ]

        mock_hs_instance = mock_hsf.create.return_value
        mock_hs_instance.retrieve.return_value = [Document("doc")]
        mock_hs_instance.retrieve_with_hyde.return_value = [Document("doc")]

        mock_rerank.return_value.rerank.return_value = [
            Document("doc", metadata={"relevance_score": 0.9, "page": 1})
        ]

        mock_router.return_value.route.return_value = "SEARCH"
        mock_hyde.return_value.generate.return_value = "hypo"
        mock_cache.return_value.get.return_value = None

        # FIX DEFINITIVO PARA LLM
        mock_llm_instance = MagicMock()
        ai_msg = AIMessage(content="Generated answer")

        # LangChain puede llamar a .invoke() O llamar al objeto directamente
        mock_llm_instance.invoke.return_value = ai_msg
        mock_llm_instance.return_value = ai_msg

        mock_llm_factory.create.return_value = mock_llm_instance

        yield {
            "loader": mock_loader,
            "vs": mock_vs,
            "hs": mock_hs_instance,
            "router": mock_router,
            "cache": mock_cache,
            "llm": mock_llm_instance,
        }


def test_pipeline_ingestion(pipeline_mocks):
    pipe = RAGPipeline(ingestion_config={"strategy": "recursive"})
    vs = pipe.run_ingestion("data.pdf")

    assert pipe.is_ready
    pipeline_mocks["loader"].return_value.load_pdf.assert_called()
    pipeline_mocks["vs"].return_value.replace_documents.assert_called()


def test_pipeline_not_ready():
    pipe = RAGPipeline()
    with pytest.raises(VectorStoreNotInitializedError):
        pipe.run_retrieval("q")


def test_pipeline_retrieval_flow(pipeline_mocks):
    pipe = RAGPipeline(ingestion_config={"strategy": "recursive"})
    pipe.run_ingestion("data.pdf")  # Init state

    docs = pipe.run_retrieval("query")

    # Verify triple strategy calls
    pipeline_mocks["hs"].retrieve_with_hyde.assert_called()  # Round 1
    pipeline_mocks["hs"].retrieve.assert_called()  # Round 2
    pipeline_mocks["vs"].return_value.similarity_search.assert_called()  # Round 3

    assert len(docs) > 0


def test_conversation_flow_greeting(pipeline_mocks):
    pipeline_mocks["router"].return_value.route.return_value = "GREETING"
    pipe = RAGPipeline()
    pipe._pipeline_state.is_initialized = True  # Fake ready
    pipe._pipeline_state.vector_store = MagicMock()

    resp, docs = pipe.run_conversation_flow("hello")
    assert "Hello" in resp
    assert docs == []


def test_conversation_flow_search(pipeline_mocks):
    pipe = RAGPipeline(ingestion_config={"strategy": "recursive"})
    pipe.run_ingestion("data.pdf")

    resp, docs = pipe.run_conversation_flow("question")

    assert resp == "Generated answer"
    assert docs is not None
    pipeline_mocks["cache"].return_value.put.assert_called()


def test_parent_expansion_logic(pipeline_mocks):
    pipe = RAGPipeline()

    # Case 1: Long chunks, random pages -> No expansion
    long_docs = [
        Document("a" * 500, metadata={"page": 1}),
        Document("b" * 500, metadata={"page": 5}),
    ]
    assert pipe._needs_parent_expansion(long_docs) == False

    # Case 2: Short chunks -> Expansion
    short_docs = [Document("a", metadata={"page": 1})]
    assert pipe._needs_parent_expansion(short_docs) == True

    # Case 3: Consecutive pages -> Expansion
    seq_docs = [
        Document("a" * 500, metadata={"page": 1}),
        Document("b" * 500, metadata={"page": 2}),
    ]
    assert pipe._needs_parent_expansion(seq_docs) == True
