from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from rag.exceptions import DocumentNotFoundError, IngestionError
from rag.ingestion.loader import DocumentLoader


@pytest.fixture
def mock_pdf_reader():
    with patch("rag.ingestion.loader.PdfReader") as mock:
        page = MagicMock()
        page.extract_text.return_value = "Contenido de prueba"
        mock.return_value.pages = [page]
        yield mock


def test_loader_init_semantic():
    with patch("rag.ingestion.loader.SemanticChunker") as mock_chunker:
        # FIX: Pasar embeddings mock
        loader = DocumentLoader(strategy="semantic", embeddings=MagicMock())
        assert loader.strategy == "semantic"
        mock_chunker.assert_called()


def test_loader_init_semantic_no_embeddings():
    with pytest.raises(ValueError):
        DocumentLoader(strategy="semantic", embeddings=None)


def test_loader_init_recursive():
    loader = DocumentLoader(strategy="recursive")
    assert loader.strategy == "recursive"


def test_load_pdf_success(mock_pdf_reader):
    # FIX: Usar recursive para no necesitar embeddings en este test
    loader = DocumentLoader(strategy="recursive")
    with patch("pathlib.Path.exists", return_value=True):
        with patch("builtins.open", mock_open(read_data=b"pdf_data")):
            docs = loader.load_pdf(Path("dummy.pdf"))
            assert len(docs) > 0
            assert docs[0].page_content == "Contenido de prueba"


def test_load_pdf_not_found():
    # FIX: Usar recursive
    loader = DocumentLoader(strategy="recursive")
    with patch("pathlib.Path.exists", return_value=False):
        with pytest.raises(DocumentNotFoundError):
            loader.load_pdf(Path("no_existe.pdf"))


def test_load_pdf_error(mock_pdf_reader):
    # FIX: Usar recursive
    loader = DocumentLoader(strategy="recursive")
    with patch("pathlib.Path.exists", return_value=True):
        mock_pdf_reader.side_effect = Exception("PDF corrupto")
        with patch("builtins.open", mock_open()):
            with pytest.raises(IngestionError):
                loader.load_pdf(Path("bad.pdf"))


def test_load_pdf_full_pages(mock_pdf_reader):
    # FIX: Usar recursive
    loader = DocumentLoader(strategy="recursive")
    with patch("pathlib.Path.exists", return_value=True):
        with patch("builtins.open", mock_open()):
            docs = loader.load_pdf_full_pages(Path("dummy.pdf"))
            assert len(docs) == 1
