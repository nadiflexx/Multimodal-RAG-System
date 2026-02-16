import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

# Asegurar que el src está en el path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


@pytest.fixture(autouse=True)
def mock_env_vars():
    """Establece variables de entorno dummy para todos los tests."""
    with patch.dict(
        os.environ, {"GROQ_API_KEY": "dummy_key", "LANGCHAIN_TRACING_V2": "false"}
    ):
        yield


@pytest.fixture
def mock_settings():
    """Mock de settings."""
    with patch("rag.config.settings") as mock_settings:
        mock_settings.GROQ_API_KEY.get_secret_value.return_value = "dummy_key"
        mock_settings.CHROMA_PATH = Path("/tmp/chroma")
        mock_settings.DATA_DIR = Path("/tmp/data")
        mock_settings.LANGCHAIN_TRACING_V2 = False
        yield mock_settings


@pytest.fixture
def mock_llm():
    """Mock del ChatGroq devolviendo un objeto AIMessage válido."""
    with patch("rag.providers.ChatGroq") as mock:
        instance = mock.return_value
        # FIX: invoke debe devolver un AIMessage, no un MagicMock genérico
        # esto evita el ValidationError de Pydantic en StrOutputParser
        instance.invoke.return_value = AIMessage(content="Respuesta simulada")
        yield instance


@pytest.fixture
def mock_embeddings():
    """Mock de LocalPyTorchEmbeddings."""
    # Mockear SentenceTransformer para evitar descarga
    with patch("rag.providers.SentenceTransformer") as mock_transformer:
        # Mockear el singleton limpiando el dict
        with patch("rag.providers.LocalPyTorchEmbeddings._instances", {}):
            instance = mock_transformer.return_value
            # Configurar para que devuelva floats, no MagicMocks, para evitar errores de numpy
            instance.encode.return_value.cpu.return_value.tolist.return_value = [
                [0.1, 0.2]
            ]
            yield instance


@pytest.fixture
def sample_docs():
    """Documentos de prueba."""
    return [
        Document(
            page_content="Document 1 content about AI.",
            metadata={"page": 1, "source": "test.pdf"},
        ),
        Document(
            page_content="Document 2 content about Python.",
            metadata={"page": 2, "source": "test.pdf"},
        ),
        Document(
            page_content="Document 3 content about Testing.",
            metadata={"page": 3, "source": "test.pdf"},
        ),
    ]
