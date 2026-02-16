from langchain_core.output_parsers import StrOutputParser
from loguru import logger

from rag.chain.prompts import get_template
from rag.models import Intent
from rag.providers import LLMFactory


class SemanticRouter:
    """
    A semantic router that classifies user intent into one of three categories:
    - SEARCH: The user is asking for information in the document
    - GREETING: The user is greeting or saying goodbye
    - OFF_TOPIC: The user is talking about unrelated topics

    This class uses a language model to classify the intent of a given query.
    """

    def __init__(self):
        self.llm = LLMFactory.create(temperature=0)

        self.template = get_template("router")

        self.chain = self.template | self.llm | StrOutputParser()

    def route(self, query: str) -> str:
        """
        Classifies the user's intent.

        Returns:
            One of: "SEARCH", "GREETING", "OFF_TOPIC"

        Raises:
            LLMError: If the classification fails
        """
        try:
            intent = self.chain.invoke({"question": query}).strip().upper()

            # Validate
            valid_intents = {i.value for i in Intent}
            if intent not in valid_intents:
                logger.warning(f"Unknown intent '{intent}', defaulting to SEARCH")
                return Intent.SEARCH.value

            return intent

        except Exception as e:
            logger.error(f"Router classification failed: {e}")
            # Fallback: treat as SEARCH intent
            return Intent.SEARCH.value
