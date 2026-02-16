"""
Document loader with configurable chunking strategy.

Handles PDF loading and processing with:
- Text cleaning
- Metadata enrichment (without polluting content)
- Semantic chunking
"""

from pathlib import Path
from typing import Literal

from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger
from pypdf import PdfReader

from rag.exceptions import DocumentNotFoundError, IngestionError


class DocumentLoader:
    """
    Load and process PDF documents with configurable chunking strategy.

    Attributes:
        strategy: "recursive" or "semantic"
        text_splitter: Instance of the configured splitter
    """

    text_splitter: SemanticChunker | RecursiveCharacterTextSplitter

    def __init__(
        self,
        strategy: Literal["recursive", "semantic"] = "semantic",
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        embeddings=None,
        buffer_size: int = 3,
        breakpoint_threshold: float = 85,
    ):
        """
        Initialize the loader with a chunking configuration.

        Args:
            strategy: Chunking strategy type.
                - "recursive": Cuts by characters (fast, predictable).
                - "semantic": Cuts by meaning using embeddings (smart, experimental).
            chunk_size: Only for "recursive". Maximum chunk size in characters.
            chunk_overlap: Only for "recursive". Character overlap between chunks.
            embeddings: Only for "semantic". Embeddings model used to calculate
                       sentence similarity.
            buffer_size: Only for "semantic". Number of sentences to buffer/group
                        before calculating similarity (smoothes out noise).
            breakpoint_threshold: Only for "semantic". Percentile threshold for
                                 splitting. 85 means split when difference is in
                                 the top 15% of all sentence differences.
        """
        self.strategy = strategy

        if strategy == "semantic":
            if embeddings is None:
                raise ValueError(
                    "SemanticChunker requires an embeddings model. "
                    "Pass embeddings=LocalPyTorchEmbeddings()"
                )

            # Mypy is happy because text_splitter is typed as Union
            self.text_splitter = SemanticChunker(
                embeddings=embeddings,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=breakpoint_threshold,
                buffer_size=buffer_size,
            )

            logger.info(
                f"DocumentLoader initialized with SemanticChunker "
                f"(threshold={breakpoint_threshold}, buffer={buffer_size})"
            )

        else:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                add_start_index=True,
            )

            logger.debug(
                f"DocumentLoader initialized with RecursiveCharacterTextSplitter "
                f"(chunk_size={chunk_size}, overlap={chunk_overlap})"
            )

    def load_pdf(self, file_path: Path) -> list[Document]:
        """
        Load and process a PDF into chunks.

        Args:
            file_path: Path to the PDF file.

        Returns:
            List of chunked documents with metadata.

        Raises:
            DocumentNotFoundError: If the file does not exist.
            IngestionError: If PDF reading fails.
        """
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            raise DocumentNotFoundError(f"PDF file not found: {file_path}")

        logger.info(f"Loading PDF: {file_path.name}")

        raw_docs = []
        try:
            with open(file_path, "rb") as file:
                reader = PdfReader(file)
                total_pages = len(reader.pages)

                logger.debug(f"PDF contains {total_pages} pages")

                for page_num, page in enumerate(reader.pages):
                    text = page.extract_text()

                    if not text or text.strip() == "":
                        logger.warning(f"Page {page_num + 1} is empty, skipping")
                        continue

                    clean_text = " ".join(text.split())

                    metadata = {
                        "source": file_path.name,
                        "page": page_num + 1,
                        "is_title_page": page_num == 0,
                        "total_pages": total_pages,
                    }

                    doc = Document(page_content=clean_text, metadata=metadata)
                    raw_docs.append(doc)

        except DocumentNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Error reading PDF: {e}")
            raise IngestionError(f"Failed to read PDF {file_path}: {e}")

        if not raw_docs:
            logger.warning(f"No content extracted from {file_path.name}")
            return []

        logger.debug(f"Splitting {len(raw_docs)} pages into chunks")
        chunks = self.text_splitter.split_documents(raw_docs)

        for idx, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = idx

        logger.success(
            f"Loaded {file_path.name}: {len(raw_docs)} pages → {len(chunks)} chunks"
        )

        return chunks

    def load_pdf_full_pages(self, file_path: Path) -> list[Document]:
        """
        Load PDF without chunking (one Document per page).
        Used as "parent documents" for the Parent Retriever.

        Args:
            file_path: Path to the PDF file.

        Returns:
            List of full-page documents.

        Raises:
            DocumentNotFoundError: If the file does not exist.
            IngestionError: If PDF reading fails.
        """
        if not file_path.exists():
            raise DocumentNotFoundError(f"PDF not found: {file_path}")

        pages = []
        try:
            with open(file_path, "rb") as file:
                reader = PdfReader(file)
                for page_num, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if not text or text.strip() == "":
                        continue

                    clean_text = " ".join(text.split())
                    doc = Document(
                        page_content=clean_text,
                        metadata={
                            "source": file_path.name,
                            "page": page_num + 1,
                        },
                    )
                    pages.append(doc)
        except Exception as e:
            logger.error(f"Error reading PDF pages: {e}")
            raise IngestionError(f"Failed to read PDF pages {file_path}: {e}")

        logger.info(f"Loaded {len(pages)} full pages (no chunking)")
        return pages
