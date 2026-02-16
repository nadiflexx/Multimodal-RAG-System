"""
Main Orchestrator Pipeline for the RAG System.

Coordinates all components: ingestion, retrieval, and generation.
Implements robust state management and error handling.
"""

import re
import unicodedata
from typing import Any

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from loguru import logger

from rag.chain.cache import SemanticCache
from rag.chain.contextualizer import QueryContextualizer
from rag.chain.memory import ChatState
from rag.chain.prompts import get_template
from rag.chain.router import SemanticRouter
from rag.config import settings, timed
from rag.exceptions import (
    IngestionError,
    RetrievalError,
    VectorStoreNotInitializedError,
)
from rag.ingestion.loader import DocumentLoader
from rag.models import Intent, PipelineState
from rag.providers import LLMFactory, LocalPyTorchEmbeddings
from rag.retrieval.hybrid import HybridSearchFactory
from rag.retrieval.hyde import HyDEGenerator
from rag.retrieval.parent import ParentDocumentStore
from rag.retrieval.reranker import Reranker
from rag.retrieval.vector_store import VectorStore


class RAGPipeline:
    """
    Main orchestrator for the RAG system.

    Coordinates:
    1. Ingestion: PDF -> Chunks -> Vector DB
    2. Retrieval: Query -> HyDE -> Hybrid Search -> Reranking
    3. Generation: Context + Query -> LLM -> Response
    """

    def __init__(self, ingestion_config: dict[str, Any] | None = None):
        """
        Initialize the pipeline components.

        Args:
            ingestion_config: Dictionary with configuration keys like
                              'strategy', 'chunk_size', etc.
                              If None, default values are used.
        """
        logger.info("Initializing RAG Pipeline")

        # Default configuration
        config = ingestion_config or {
            "strategy": "semantic",
            "breakpoint_threshold": 85,
            "buffer_size": 3,
        }

        self.embeddings = LocalPyTorchEmbeddings()

        # Configure loader dynamically based on strategy
        if config.get("strategy") == "semantic":
            self.loader = DocumentLoader(
                strategy="semantic",
                embeddings=self.embeddings,
                breakpoint_threshold=config.get("breakpoint_threshold", 85),
                buffer_size=config.get("buffer_size", 3),
            )
        else:
            self.loader = DocumentLoader(
                strategy="recursive",
                chunk_size=config.get("chunk_size", 1000),
                chunk_overlap=config.get("chunk_overlap", 200),
            )

        logger.info(f"Loader configured: {config}")

        self.reranker = Reranker()
        self.llm = LLMFactory.create(temperature=0)
        self.router = SemanticRouter()
        self.hyde = HyDEGenerator()
        self.state = ChatState()
        self.contextualizer = QueryContextualizer()
        self._pipeline_state = PipelineState()
        self.parent_store = ParentDocumentStore()

        self.cache = SemanticCache(
            embeddings=self.embeddings,
            threshold=0.95,
            max_cache_size=500,
            persist_path=str(settings.BASE_DIR / "cache" / "semantic_cache.json"),
        )
        logger.success("RAG Pipeline initialized successfully")

    @property
    def is_ready(self) -> bool:
        """Check if the pipeline is initialized and ready for queries."""
        return (
            self._pipeline_state.is_initialized
            and self._pipeline_state.vector_store is not None
        )

    def run_ingestion(self, filename: str) -> VectorStore:
        """
        Execute the full ingestion process.

        Pipeline:
        1. Load PDF
        2. Generate chunks
        3. Create embeddings
        4. Index in ChromaDB

        Args:
            filename: Name of the file in DATA_DIR.

        Returns:
            The initialized VectorStore instance.

        Raises:
            IngestionError: If any ingestion step fails.
            DocumentNotFoundError: If the file does not exist.
        """
        logger.info(f"Starting ingestion for: {filename}")
        file_path = settings.DATA_DIR / filename

        # 1. Load chunks
        documents = self.loader.load_pdf(file_path)
        if not documents:
            raise IngestionError(f"No documents extracted from {filename}")

        logger.info(f"Generated {len(documents)} chunks")

        # 2. Load full pages (for parent retriever)
        full_pages = self.loader.load_pdf_full_pages(file_path)
        self.parent_store.store_parents(full_pages)

        # 3. Index chunks (replace existing collection)
        safe_name = filename.replace(".", "_").replace(" ", "_").lower()
        vector_store = VectorStore(
            collection_name=safe_name,
            embedding_model=self.embeddings,
        )
        vector_store.replace_documents(documents)

        # 4. Create Hybrid Retriever
        hybrid_retriever = HybridSearchFactory.create(
            vector_store=vector_store,
            documents=documents,
        )

        # 5. Update state
        self._pipeline_state.vector_store = vector_store
        self._pipeline_state.hybrid_retriever = hybrid_retriever
        self._pipeline_state.current_file = filename
        self._pipeline_state.is_initialized = True

        # 6. Invalidate cache (old document answers are invalid)
        self.cache.invalidate()

        logger.success(f"Ingestion complete: {filename} ready for queries")
        return vector_store

    @timed
    def run_retrieval(self, query: str) -> list[Document]:
        """
        Execute triple strategy retrieval pipeline.

        Strategies:
        1. HyDE + Hybrid: For complex conceptual queries.
        2. Direct Hybrid: For exact keyword matching.
        3. Direct Vector: For short/structural queries (pure query embedding).

        The reranker receives ALL candidates and selects the best ones.

        Args:
            query: User query string.

        Returns:
            List of reranked documents.

        Raises:
            VectorStoreNotInitializedError: If ingestion hasn't run.
            RetrievalError: If search fails.
        """
        if not self.is_ready:
            raise VectorStoreNotInitializedError(
                "Pipeline not ready. Run ingestion first."
            )

        assert self._pipeline_state.hybrid_retriever is not None
        assert self._pipeline_state.vector_store is not None

        try:
            retriever = self._pipeline_state.hybrid_retriever
            vector_store = self._pipeline_state.vector_store

            # ── Round 1: HyDE ──
            logger.info("Generating hypothetical document (HyDE)")
            hypothetical_doc = self.hyde.generate(query)

            hyde_candidates = retriever.retrieve_with_hyde(
                original_query=query,
                hypothetical_doc=hypothetical_doc,
            )

            # ── Round 2: Direct Hybrid ──
            logger.info("Executing direct hybrid search")
            direct_candidates = retriever.retrieve(query)

            # ── Round 3: Direct Vector (Pure query against ChromaDB) ──
            logger.info("Executing direct vector search")
            direct_vector = vector_store.similarity_search(query, k=10)
            direct_v_pages = [d.metadata.get("page") for d in direct_vector]
            logger.info(
                f"[DirectVector] pages ({len(direct_vector)}): {direct_v_pages}"
            )

            # ── Merge 3 rounds ──
            all_candidates = self._merge_candidate_lists(
                hyde_candidates, direct_candidates, direct_vector
            )

            logger.info(f"Total unique candidates for reranker: {len(all_candidates)}")

            if not all_candidates:
                logger.warning("No candidates found")
                return []

            # ── Reranking ──
            reranked = self.reranker.rerank(query, all_candidates, top_n=5)

            for i, doc in enumerate(reranked):
                score = doc.metadata.get("relevance_score", 0)
                page = doc.metadata.get("page", "?")
                preview = doc.page_content[:80].replace("\n", " ")
                logger.info(
                    f"  Rerank #{i + 1}: page={page} score={score:.4f} '{preview}...'"
                )

            # ── Filter by minimum score ──
            filtered = [
                d for d in reranked if d.metadata.get("relevance_score", 0) > 0.01
            ]
            if not filtered:
                filtered = reranked[:3]  # Fallback to top 3

            # ── Expand if necessary ──
            if self._needs_parent_expansion(filtered):
                parent_docs = self.parent_store.get_parents_for_chunks(
                    filtered, expand_neighbors=True
                )
                logger.info(
                    f"Expanded {len(filtered)} chunks → {len(parent_docs)} pages"
                )
                return parent_docs

            logger.info(f"Returning {len(filtered)} chunks")
            return filtered

        except VectorStoreNotInitializedError:
            raise
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            raise RetrievalError(f"Could not retrieve documents: {e}")

    def _merge_candidate_lists(self, *lists: list[Document]) -> list[Document]:
        """Merge multiple document lists removing duplicates."""
        seen = {}
        for doc_list in lists:
            for doc in doc_list:
                key = doc.page_content[:150]
                if key not in seen:
                    seen[key] = doc
        return list(seen.values())

    def run_generation(self, context_docs: list[Document], query: str) -> str:
        """Generate response using LLM with provided context."""

        def format_docs(docs: list[Document]) -> str:
            return "\n\n".join(
                [
                    f"[Page {d.metadata.get('page', '?')}]: {d.page_content}"
                    for d in docs
                ]
            )

        logger.info("Generating response with LLM")
        context_text = format_docs(context_docs)
        chain = get_template("chat") | self.llm | StrOutputParser()

        response = chain.invoke({"context": context_text, "question": query})
        logger.debug(f"Generated response ({len(response)} chars)")
        return response

    def _normalize_query(self, query: str) -> str:
        """Normalize query to improve cache hit rate."""
        q = query.lower()
        q = unicodedata.normalize("NFD", q)
        q = "".join(c for c in q if unicodedata.category(c) != "Mn")
        q = re.sub(r"[¿?!¡.,;:\-\"']", "", q)
        q = " ".join(q.split())
        return q.strip()

    def run_conversation_flow(self, user_query: str) -> tuple[str, list[Document]]:
        """
        Execute full conversational flow.

        1. Route Intent
        2. Normalize
        3. Check Cache
        4. Contextualize (if cache miss)
        5. Retrieve + Rerank
        6. Generate
        7. Update Cache & Memory
        """
        if not self.is_ready:
            raise VectorStoreNotInitializedError("Load a document first")

        logger.info(f"Processing query: '{user_query[:50]}...'")

        intent = self.router.route(user_query)
        logger.info(f"Detected intent: {intent}")

        if intent == Intent.GREETING.value:
            return ("Hello! How can I help you today?", [])
        if intent == Intent.OFF_TOPIC.value:
            return ("I can only answer questions related to the document.", [])

        normalized = self._normalize_query(user_query)

        # Cache check - get() returns None or tuple(canonical, response, docs)
        cached = self.cache.get(normalized)
        if cached is not None:
            response, docs = cached
            self.state.add_user_message(user_query)
            self.state.add_ai_message(response)
            return response, docs

        history = self.state.get_history()
        refined_query = self.contextualizer.contextualize(user_query, history)

        docs = self.run_retrieval(refined_query)
        if not docs:
            return "I couldn't find any relevant information.", []

        answer = self.run_generation(docs, refined_query)

        # Store in cache with normalized query
        self.cache.put(normalized, answer, docs)

        self.state.add_user_message(user_query)
        self.state.add_ai_message(answer)

        return answer, docs

    def reset(self) -> None:
        """Reset pipeline state and memory."""
        logger.info("Resetting pipeline state")
        self._pipeline_state = PipelineState()
        self.state.clear()
        logger.success("Pipeline reset complete")

    def _needs_parent_expansion(self, docs: list[Document]) -> bool:
        """
        Decide if chunks need expansion to full pages.

        Heuristics:
        1. If average chunk length is very short (< 200 chars).
        2. If chunks come from consecutive pages (likely spanning content).
        """
        if not docs:
            return False

        avg_length = sum(len(d.page_content) for d in docs) / len(docs)
        if avg_length < 200:
            return True

        pages = sorted(set(d.metadata.get("page", 0) for d in docs))
        for i in range(len(pages) - 1):
            if pages[i + 1] - pages[i] == 1:
                return True

        return False
