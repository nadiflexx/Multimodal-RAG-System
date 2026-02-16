class ChatState:
    """
    Manages conversation memory (Short-term memory).
    Maintains a sliding window of the last 6 messages.
    """

    def __init__(self, window_size: int = 6):
        self.history: list[dict[str, str]] = []
        self.window_size = window_size

    def add_user_message(self, message: str) -> None:
        """
        Adds a user message to the conversation history.
        """
        self.history.append({"role": "user", "content": message})

    def add_ai_message(self, message: str) -> None:
        """
        Adds an assistant message to the conversation history.
        """
        self.history.append({"role": "assistant", "content": message})

    def get_history(self) -> list[dict[str, str]]:
        """Returns the last N messages (sliding window)."""
        return self.history[-self.window_size :]

    def clear(self) -> None:
        """Clears the conversation history."""
        self.history = []

    @property
    def message_count(self) -> int:
        """Returns the number of messages in the history."""
        return len(self.history)
