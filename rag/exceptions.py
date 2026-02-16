"""
Custom exceptions for the RAG pipeline.

Provides a hierarchy of exceptions to handle errors granularly across different
components of the system (Ingestion, Retrieval, Generation, Infrastructure).

Hierarchy:
    RAGException
    ├── ConfigurationError
    ├── DocumentNotFoundError
    ├── VectorStoreNotInitializedError
    ├── IngestionError
    ├── RetrievalError
    ├── EmbeddingError
    ├── LLMError
    └── CacheError
"""


class RAGException(Exception):
    """
    Base exception for all RAG pipeline errors.

    Attributes:
        message: Explanation of the error.
        original_error: The underlying exception that caused this error (optional).
    """

    def __init__(self, message: str, original_error: Exception | None = None):
        super().__init__(message)
        self.message = message
        self.original_error = original_error

    def __str__(self) -> str:
        if self.original_error:
            return f"{self.message} (Caused by: {self.original_error})"
        return self.message


class ConfigurationError(RAGException):
    """Raised when there is a configuration error (eg: missing environment variables)"""

    pass


class DocumentNotFoundError(RAGException):
    """Raised when the specified document file cannot be found."""

    pass


class VectorStoreNotInitializedError(RAGException):
    """Raised when attempting to query the vector store before ingestion."""

    pass


class IngestionError(RAGException):
    """Raised when the document ingestion process fails (loading or chunking)."""

    pass


class RetrievalError(RAGException):
    """Raised when the retrieval process fails (search or reranking)."""

    pass


class EmbeddingError(RAGException):
    """Raised when embedding generation fails."""

    pass


class LLMError(RAGException):
    """Raised when communication with the LLM provider fails."""

    pass


class CacheError(RAGException):
    """Raised when semantic cache operations fail."""

    pass
