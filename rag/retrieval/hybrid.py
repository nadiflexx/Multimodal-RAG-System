"""
Hybrid Retrieval Strategy combining Vector Search and Keyword-based Search (BM25).
"""

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from loguru import logger

from rag.retrieval.vector_store import VectorStore


class HybridRetriever:
    """
    Retriever that combines semantic search (Vector) and keyword search (BM25).

    It executes both retrieval strategies independently and merges the results,
    removing duplicates. This ensures that both conceptual matches and exact
    keyword matches are captured.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        documents: list[Document],
        vector_k: int = 10,
        bm25_k: int = 15,
    ):
        """
        Initialize the HybridRetriever.

        Args:
            vector_store: Initialized VectorStore instance.
            documents: List of documents to index for BM25.
            vector_k: Number of documents to retrieve from vector store.
            bm25_k: Number of documents to retrieve from BM25.
        """
        self.vector_k = vector_k
        self.bm25_k = bm25_k

        # Initialize Vector Retriever
        self.vector_retriever = vector_store.as_retriever(k=vector_k)

        # Initialize BM25 Retriever (in-memory)
        self.bm25_retriever = BM25Retriever.from_documents(documents)
        self.bm25_retriever.k = bm25_k

        logger.info(f"Hybrid retriever ready (vector_k={vector_k}, bm25_k={bm25_k})")

    def retrieve(self, query: str) -> list[Document]:
        """
        Execute standard hybrid search using the same query for both retrievers.

        Args:
            query: The search query string.

        Returns:
            Merged list of unique documents from both sources.
        """
        vector_docs = self.vector_retriever.invoke(query)
        keyword_docs = self.bm25_retriever.invoke(query)

        self._log_pages("Direct", vector_docs, keyword_docs)

        return self._merge(vector_docs, keyword_docs)

    def retrieve_with_hyde(
        self, original_query: str, hypothetical_doc: str
    ) -> list[Document]:
        """
        Execute advanced hybrid search using HyDE strategy.

        - Vector search uses the hypothetical document (better for semantics).
        - BM25 search uses the original query (better for exact keywords).

        Args:
            original_query: The user's original query.
            hypothetical_doc: The generated hypothetical document.

        Returns:
            Merged list of unique documents.
        """
        vector_docs = self.vector_retriever.invoke(hypothetical_doc)
        keyword_docs = self.bm25_retriever.invoke(original_query)

        self._log_pages("HyDE", vector_docs, keyword_docs)

        merged = self._merge(vector_docs, keyword_docs)
        logger.info(
            f"HyDE retrieval: {len(vector_docs)} vector + "
            f"{len(keyword_docs)} keyword → {len(merged)} merged"
        )
        return merged

    def _merge(
        self, vector_docs: list[Document], keyword_docs: list[Document]
    ) -> list[Document]:
        """
        Merge two lists of documents, removing duplicates based on content preview.

        Args:
            vector_docs: Results from vector search.
            keyword_docs: Results from keyword search.

        Returns:
            Deduplicated list of documents.
        """
        seen = {}
        # Merge lists, giving priority to vector docs order if needed,
        # but here we just collect unique docs.
        for doc in vector_docs + keyword_docs:
            # Use first 150 chars as a unique key for deduplication
            key = doc.page_content[:150]
            if key not in seen:
                seen[key] = doc
        return list(seen.values())

    def _log_pages(
        self,
        strategy: str,
        vector_docs: list[Document],
        keyword_docs: list[Document],
    ) -> None:
        """Helper to log which pages were retrieved by each method."""
        v = [d.metadata.get("page") for d in vector_docs]
        k = [d.metadata.get("page") for d in keyword_docs]
        logger.info(f"[{strategy}] Vector pages: {v}")
        logger.info(f"[{strategy}] BM25 pages:   {k}")


class HybridSearchFactory:
    """Factory for creating HybridRetriever instances."""

    @staticmethod
    def create(
        vector_store: VectorStore,
        documents: list[Document],
        vector_k: int = 10,
        bm25_k: int = 15,
    ) -> HybridRetriever:
        """
        Create a configured HybridRetriever.

        Args:
            vector_store: The vector store instance.
            documents: List of documents for BM25 indexing.
            vector_k: Number of vector results.
            bm25_k: Number of BM25 results.

        Returns:
            A ready-to-use HybridRetriever.
        """
        logger.debug(f"Creating hybrid retriever with {len(documents)} docs")
        return HybridRetriever(
            vector_store,
            documents,
            vector_k=vector_k,
            bm25_k=bm25_k,
        )
