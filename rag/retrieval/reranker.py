"""FlashRank reranker for relevance filtering."""

from flashrank import Ranker, RerankRequest  # type: ignore[import-untyped]
from langchain_core.documents import Document
from loguru import logger

from rag.config import timed
from rag.exceptions import RetrievalError


class Reranker:
    """
    Wraps FlashRank for local, privacy-first reranking.

    Uses a Cross-Encoder model to re-score documents based on their
    actual relevance to the query, improving precision significantly
    compared to simple vector similarity.
    """

    def __init__(self, model_name: str = "ms-marco-MiniLM-L-12-v2") -> None:
        """
        Initialize the Reranker with a specific model.

        Args:
            model_name: Name of the FlashRank model to use.
                        Defaults to a lightweight MS MARCO model.

        Raises:
            RetrievalError: If the model cannot be loaded.
        """
        try:
            # cache_dir stores the downloaded model locally
            self.ranker = Ranker(model_name=model_name, cache_dir="./models_cache")
        except Exception as e:
            logger.error(f"Failed to initialize reranker: {e}")
            raise RetrievalError(f"Could not load reranker model: {e}")

    @timed
    def rerank(
        self, query: str, documents: list[Document], top_n: int = 3
    ) -> list[Document]:
        """
        Re-rank a list of documents based on relevance to the query.

        Args:
            query: The user query string.
            documents: List of candidate documents (from retrieval).
            top_n: Number of top documents to return.

        Returns:
            List of re-ranked documents, sorted by relevance score.

        Raises:
            RetrievalError: If the reranking process fails.
        """
        if not documents:
            return []

        try:
            # FlashRank expects a specific dictionary format
            passages = [
                {
                    "id": str(i),
                    "text": doc.page_content,
                    "meta": doc.metadata,
                }
                for i, doc in enumerate(documents)
            ]

            rerank_request = RerankRequest(query=query, passages=passages)
            results = self.ranker.rerank(rerank_request)

            # Slice to get only top N results
            results = results[:top_n]

            # Convert back to LangChain Documents
            reranked_docs = []
            for res in results:
                doc = Document(page_content=res["text"], metadata=res["meta"])
                doc.metadata["relevance_score"] = float(res["score"])
                reranked_docs.append(doc)

            return reranked_docs

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            raise RetrievalError(f"Reranking failed: {e}")
