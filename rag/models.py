"""
Data Transfer Objects (DTOs) shared across modules.

Centralizes schemas to avoid circular imports and ensure clear contracts between layers.
"""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from langchain_core.documents import Document


class Intent(StrEnum):
    """User intents detected by the SemanticRouter."""

    SEARCH = "SEARCH"
    GREETING = "GREETING"
    OFF_TOPIC = "OFF_TOPIC"


@dataclass
class RetrievalResult:
    """Result of the retrieval pipeline execution."""

    documents: list[Document] = field(default_factory=list)
    query_used: str = ""
    hyde_doc: str | None = None


@dataclass
class PipelineState:
    """
    Internal state of the RAG pipeline.

    Attributes:
        vector_store: Instance of the VectorStore (None if not initialized).
        hybrid_retriever: Configured HybridRetriever instance.
        current_file: Name of the currently loaded file.
        is_initialized: Flag indicating if the pipeline is ready for queries.
    """

    vector_store: Any | None = None
    hybrid_retriever: Any | None = None
    current_file: str | None = None
    is_initialized: bool = False
