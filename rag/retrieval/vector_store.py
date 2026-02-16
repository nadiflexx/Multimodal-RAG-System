"""
ChromaDB vector store wrapper with support for re-indexing and direct search.
"""

import shutil
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from loguru import logger

from rag.config import settings


class VectorStore:
    """
    Wrapper around ChromaDB to manage vector storage and retrieval.

    Handles persistence, re-indexing, and provides methods for both
    retriever-based access (LangChain standard) and direct similarity search.
    """

    def __init__(self, collection_name: str, embedding_model: Embeddings):
        """
        Initialize the VectorStore.

        Args:
            collection_name: Name of the ChromaDB collection.
            embedding_model: Embedding model instance to use for vectorization.
        """
        self.collection_name = collection_name
        self.embedding_model = embedding_model

        self.db = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_model,
            persist_directory=str(settings.CHROMA_PATH),
            collection_metadata={"hnsw:space": "cosine"},
        )

    def as_retriever(
        self,
        search_type: str = "similarity",
        k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> BaseRetriever:
        """
        Return a LangChain Retriever object for use in pipelines.

        Args:
            search_type: Type of search ("similarity" or "mmr").
            k: Number of documents to retrieve.
            filter: Metadata filter dictionary (e.g., {"page": 1}).

        Returns:
            A configured BaseRetriever instance.
        """
        search_kwargs: dict[str, Any] = {"k": k}
        if filter:
            search_kwargs["filter"] = filter

        return self.db.as_retriever(
            search_type=search_type, search_kwargs=search_kwargs
        )

    def get_count(self) -> int:
        """Return the total number of documents in the collection."""
        return self.db._collection.count()

    def add_documents(self, documents: list[Document]) -> None:
        """
        Add documents to the store if the collection is empty.
        (Idempotent add).
        """
        if self.get_count() > 0:
            logger.info(
                f"Collection '{self.collection_name}' already has data. Skipping."
            )
            return

        logger.info(f"Indexing {len(documents)} chunks...")
        self.db.add_documents(documents)
        logger.success("Indexing complete.")

    def similarity_search(self, query: str, k: int = 10) -> list[Document]:
        """
        Perform a direct similarity search using the query string.

        Useful for short or structural queries where HyDE might introduce noise.
        Bypasses the retriever abstraction to access the vector store directly.

        Args:
            query: The query text.
            k: Number of documents to return.

        Returns:
            List of matching Documents.
        """
        return self.db.similarity_search(query, k=k)

    def clear_collection(self) -> None:
        """
        Delete all documents in the current collection.
        Used when re-ingesting a file to ensure a clean state.
        """
        count = self.get_count()
        if count == 0:
            return

        logger.info(f"Clearing collection '{self.collection_name}' ({count} documents)")

        all_ids = self.db._collection.get()["ids"]
        if all_ids:
            self.db._collection.delete(ids=all_ids)

        logger.success(f"Collection '{self.collection_name}' cleared")

    def replace_documents(self, documents: list[Document]) -> None:
        """
        Replace all documents in the collection with new ones.

        Steps:
        1. Clear existing documents.
        2. Index new documents.

        Args:
            documents: List of new Documents to index.
        """
        self.clear_collection()

        logger.info(f"Indexing {len(documents)} new chunks...")
        self.db.add_documents(documents)
        logger.success("Reindexing complete.")

    def query(self, text: str, k: int = 5) -> list[tuple[Document, float]]:
        """
        Perform similarity search and return scores.

        Returns:
            List of tuples (Document, score).
        """
        logger.debug(f"Querying: '{text}'")
        return self.db.similarity_search_with_score(text, k=k)

    @staticmethod
    def delete_all_collections() -> None:
        """
        Physically delete the entire ChromaDB directory from disk.
        Useful for a hard reset of the system.
        """
        chroma_path = Path(settings.CHROMA_PATH)
        if chroma_path.exists():
            shutil.rmtree(chroma_path)
            logger.info(f"Deleted ChromaDB directory: {chroma_path}")
