"""
Parent Document Retriever.

Strategy: Indexes small chunks for precise search,
but returns the "parent" document (full page) to
provide full context to the LLM.
"""

from langchain_core.documents import Document
from loguru import logger


class ParentDocumentStore:
    """
    Stores parent documents (full pages) and allows retrieving them
    when a child chunk is found.
    """

    def __init__(self) -> None:
        """Initialize the store with empty dictionaries for parents and chunks."""
        self.parents: dict[int, Document] = {}
        self.page_chunks: dict[int, list[Document]] = {}

    def store_parents(self, full_pages: list[Document]) -> None:
        """
        Store full pages (without chunking) as parent documents.

        Args:
            full_pages: List of Documents, one per page.
        """
        for doc in full_pages:
            page_num = doc.metadata.get("page", 0)
            self.parents[page_num] = doc

        logger.info(f"Stored {len(full_pages)} parent documents")

    def get_parent(self, chunk: Document) -> Document:
        """
        Given a chunk, return the full page it came from.

        Args:
            chunk: The chunk found by retrieval.

        Returns:
            The corresponding full page Document.
        """
        page_num = chunk.metadata.get("page", 0)

        if page_num in self.parents:
            return self.parents[page_num]

        logger.warning(f"Parent not found for page {page_num}")
        return chunk

    def get_parents_for_chunks(
        self,
        chunks: list[Document],
        expand_neighbors: bool = True,
    ) -> list[Document]:
        """
        Given a list of chunks, return the corresponding full pages (deduplicated).

        Args:
            chunks: List of chunks found by retrieval.
            expand_neighbors: If True, includes adjacent pages (useful for
                              tables/content that spans across pages).

        Returns:
            List of full page documents, deduplicated.
        """
        seen_pages = set()
        parent_docs = []

        for chunk in chunks:
            page_num = chunk.metadata.get("page", 0)

            pages_to_fetch = [page_num]

            if expand_neighbors:
                if page_num - 1 in self.parents:
                    pages_to_fetch.append(page_num - 1)
                if page_num + 1 in self.parents:
                    pages_to_fetch.append(page_num + 1)

            for p in pages_to_fetch:
                if p not in seen_pages and p in self.parents:
                    seen_pages.add(p)
                    parent_docs.append(self.parents[p])

        logger.info(f"Expanded {len(chunks)} chunks → {len(parent_docs)} parent pages")

        return parent_docs
