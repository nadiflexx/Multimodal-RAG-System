"""
Hypothetical Document Embeddings (HyDE) Generator.

Generates a hypothetical answer/document given a query. This hypothetical
document is then used for vector retrieval instead of the raw query,
improving semantic matching by bridging the gap between query space
and document space.
"""

from langchain_core.output_parsers import StrOutputParser
from loguru import logger

from rag.chain.prompts import get_template
from rag.exceptions import LLMError
from rag.providers import LLMFactory


class HyDEGenerator:
    """
    Generator for hypothetical documents using an LLM.

    Uses a specialized prompt to instruct the LLM to hallucinate a plausible
    answer to the user's question, which captures the semantic meaning better
    for retrieval purposes.
    """

    def __init__(self) -> None:
        """Initialize the generator with an LLM and the HyDE prompt template."""
        # Use slightly higher temperature for creativity in generating the hypothesis
        self.llm = LLMFactory.create(temperature=0.3)

        self.template = get_template("hyde")

        self.chain = self.template | self.llm | StrOutputParser()

    def generate(self, query: str) -> str:
        """
        Generate a hypothetical document for the given query.

        Args:
            query: The user's search query.

        Returns:
            A string containing the hypothetical document/answer.

        Raises:
            LLMError: If the LLM generation fails.
        """
        try:
            return self.chain.invoke({"question": query})
        except Exception as e:
            logger.error(f"HyDE generation failed: {e}")
            raise LLMError(f"Could not generate hypothetical document: {e}")
