"""
Centralized configuration for the RAG system.

Includes:
- Pydantic Settings with validation.
- Logging configuration (Loguru).
"""

import functools
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from loguru import logger
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()

# ═══════════════════════════════════════════════════════
# SETTINGS
# ═══════════════════════════════════════════════════════


class Settings(BaseSettings):
    """
    Application settings managed by Pydantic.
    Reads values from environment variables or the .env file.
    """

    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    CHROMA_PATH: Path = BASE_DIR / "chroma_db"
    DATA_DIR: Path = BASE_DIR / "data"

    # Required field
    GROQ_API_KEY: SecretStr = Field(..., min_length=1)

    # LangChain Observability (Optional)
    # Note: These are read for validation, but LangChain reads them directly from env
    LANGCHAIN_TRACING_V2: bool = Field(default=False)
    LANGCHAIN_ENDPOINT: str = Field(default="https://api.smith.langchain.com")
    LANGCHAIN_API_KEY: SecretStr | None = Field(default=None)
    LANGCHAIN_PROJECT: str = Field(default="RAG_Workshop_Pro")

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


try:
    # We ignore [call-arg] because Mypy complains about missing arguments (GROQ_API_KEY)
    # but Pydantic loads them dynamically from the .env file at runtime.
    settings = Settings()  # type: ignore[call-arg]

    if settings.LANGCHAIN_TRACING_V2:
        print(
            f"🔭 Observability enabled: Logging to project "
            f"'{settings.LANGCHAIN_PROJECT}' @ {settings.LANGCHAIN_ENDPOINT}"
        )

except Exception as e:
    print("❌ ERROR: Could not load configuration.")
    raise e


# ═══════════════════════════════════════════════════════
# LOGGING (Loguru)
# ═══════════════════════════════════════════════════════

LOG_DIR = settings.BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Remove default handler
logger.remove()

# Console Handler (INFO+)
logger.add(
    sys.stdout,
    format=(
        "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
    ),
    level="INFO",
    colorize=True,
)

# File Handler (DEBUG+) with daily rotation
logger.add(
    LOG_DIR / "rag_{time:YYYY-MM-DD}.log",
    rotation="00:00",
    retention="7 days",
    level="DEBUG",
    format=(
        "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
        "{name}:{function}:{line} - {message}"
    ),
)


# ═══════════════════════════════════════════════════════
# TIMER DECORATOR
# ═══════════════════════════════════════════════════════


def timed(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to measure execution time of a function."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = (time.perf_counter() - start) * 1000
        logger.debug(f"⏱️  [{func.__name__}] executed in {elapsed:.2f}ms")
        return result

    return wrapper
