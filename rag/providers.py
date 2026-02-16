"""
Factories for external services: LLM and Embeddings.

Centralizes instance creation with:
- Thread-safe singleton for embeddings
- Factory pattern for LLMs
- Credential validation
"""

from threading import Lock

import torch
from langchain_core.embeddings import Embeddings
from langchain_groq import ChatGroq
from loguru import logger
from sentence_transformers import SentenceTransformer

from rag.config import settings
from rag.exceptions import EmbeddingError, LLMError

# ═══════════════════════════════════════════════════════
# LLM FACTORY
# ═══════════════════════════════════════════════════════


class LLMFactory:
    """
    Factory to create configured LLM instances.

    Example:
        >>> llm = LLMFactory.create()
        >>> llm_creative = LLMFactory.create(temperature=0.7)
    """

    @staticmethod
    def create(
        model_name: str = "llama-3.3-70b-versatile",
        temperature: float = 0,
    ) -> ChatGroq:
        """
        Creates a configured LLM instance.

        Args:
            model_name: Name of the model in Groq API.
            temperature: Randomness control (0=deterministic, 1=creative).

        Returns:
            Configured ChatGroq instance.

        Raises:
            LLMError: If API key is not configured or initialization fails.
        """
        # Validation: Check if key exists (but don't expose it yet)
        if not settings.GROQ_API_KEY or not settings.GROQ_API_KEY.get_secret_value():
            logger.error("GROQ_API_KEY is not set or empty")
            raise LLMError(
                "GROQ_API_KEY is not configured. Please set it in .env file."
            )

        try:
            logger.debug(f"Creating LLM instance: {model_name} (temp={temperature})")

            llm = ChatGroq(
                model=model_name,
                temperature=temperature,
                api_key=settings.GROQ_API_KEY,
            )

            logger.success(f"LLM instance created: {model_name}")
            return llm

        except Exception as e:
            logger.error(f"Failed to create LLM instance: {e}")
            raise LLMError(f"Could not initialize LLM {model_name}: {e}")


# ═══════════════════════════════════════════════════════
# EMBEDDINGS SINGLETON
# ═══════════════════════════════════════════════════════


class LocalPyTorchEmbeddings(Embeddings):
    """
    Thread-safe Singleton for local embeddings.

    Maintains a model cache by name to avoid reloads.

    Example:
        >>> embeddings = LocalPyTorchEmbeddings()
        >>> embeddings_custom = LocalPyTorchEmbeddings("paraphrase-multilingual...")
    """

    _instances: dict[str, "LocalPyTorchEmbeddings"] = {}
    _lock: Lock = Lock()

    def __new__(cls, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        """Create or return singleton instance for the specified model."""
        with cls._lock:
            if model_name not in cls._instances:
                logger.debug(f"Creating new singleton instance for model: {model_name}")
                instance = super().__new__(cls)
                instance._initialize_model(model_name)
                cls._instances[model_name] = instance
            return cls._instances[model_name]

    def _initialize_model(self, model_name: str) -> None:
        """Initialize the model with optimal device auto-detection."""
        try:
            device = "cpu"
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"

            logger.info(
                f"Loading embedding model '{model_name}' on device: {device.upper()}"
            )

            self._model: SentenceTransformer = SentenceTransformer(
                model_name, device=device
            )
            self.model_name = model_name

            logger.success(f"Model '{model_name}' loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise EmbeddingError(f"Could not initialize model {model_name}: {e}")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple documents."""
        if not texts:
            return []

        try:
            with torch.no_grad():
                embeddings = self._model.encode(
                    texts, convert_to_tensor=True, show_progress_bar=False
                )
                return embeddings.cpu().tolist()
        except Exception as e:
            logger.error(f"Failed to embed documents: {e}")
            raise EmbeddingError(f"Document embedding failed: {e}")

    def embed_query(self, text: str) -> list[float]:
        """Generate embedding for a single query."""
        try:
            with torch.no_grad():
                embedding = self._model.encode(
                    [text], convert_to_tensor=True, show_progress_bar=False
                )
                return embedding.cpu().tolist()[0]
        except Exception as e:
            logger.error(f"Failed to embed query: {e}")
            raise EmbeddingError(f"Query embedding failed: {e}")
