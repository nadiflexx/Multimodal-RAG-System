from langchain_core.output_parsers import StrOutputParser
from loguru import logger

from rag.chain.prompts import get_template
from rag.providers import LLMFactory


class QueryContextualizer:
    """
    Contextualizes a query by resolving ambiguous references in conversation history.
    """

    def __init__(self):
        """Initialize the QueryContextualizer with an LLM and a prompt template."""

        self.llm = LLMFactory.create(temperature=0.1)

        self.template = get_template("contextualizer")

        self.chain = self.template | self.llm | StrOutputParser()

    def contextualize(self, query: str, history: list) -> str:
        """
        Reformulates the query by resolving ambiguous pronouns.

        Returns:
            Query contextualized or the original if there is no history

        Raises:
            LLMError: if contextualization fails (fallback to original query)
        """
        if not history:
            return query

        try:
            history_str = "\n".join(
                [f"{msg['role']}: {msg['content']}" for msg in history]
            )

            new_query = self.chain.invoke(
                {"chat_history": history_str, "question": query}
            )

            # # Guard: Rechazar expansiones excesivas
            # if len(new_query) > len(query) * 6:
            #     logger.warning(
            #         f"Contextualizer expanded too much: "
            #         f"'{query}' → '{new_query[:80]}...' "
            #         f"(rejected, using original)"
            #     )
            #     return query

            if new_query.strip() != query.strip():
                print(f"🔄 Reformulated: '{query}' ---> '{new_query}'")

            return new_query

        except Exception as e:
            logger.error(f"Contextualization failed: {e}, using original query")
            return query
