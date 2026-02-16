import os
from unittest.mock import patch

from rag.config import Settings, timed


def test_settings_load_from_env():
    mock_env = {
        "GROQ_API_KEY": "test_key",
        "LANGCHAIN_TRACING_V2": "true",
        "LANGCHAIN_PROJECT": "test_project",
    }

    with patch.dict(os.environ, mock_env):
        settings = Settings()
        assert settings.GROQ_API_KEY.get_secret_value() == "test_key"
        assert settings.LANGCHAIN_TRACING_V2 is True
        assert settings.LANGCHAIN_PROJECT == "test_project"


def test_timed_decorator():
    # FIX: Patch logger directamente para verificar la llamada
    with patch("rag.config.logger") as mock_logger:

        @timed
        def slow_func():
            return "done"

        result = slow_func()

        assert result == "done"
        # Verificar que logger.debug fue llamado
        mock_logger.debug.assert_called()
        # Verificar que el mensaje contiene el nombre de la función
        args, _ = mock_logger.debug.call_args
        assert "slow_func" in args[0]
