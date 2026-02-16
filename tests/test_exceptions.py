from rag.exceptions import ConfigurationError, RAGException


def test_rag_exception_str():
    # Test mensaje simple
    exc = RAGException("Error base")
    assert str(exc) == "Error base"

    # Test con error original (chaining)
    original = ValueError("Valor malo")
    exc_chained = ConfigurationError("Config mal", original_error=original)
    assert "Config mal" in str(exc_chained)
    assert "Valor malo" in str(exc_chained)
