# data/validation/schema.py
"""
Validation schemas for the training dataset.
We use Pydantic v2 because it is production-standard and demonstrates
proper separation between validation logic and the rest of the codebase.
"""
from enum import Enum
from pydantic import BaseModel, field_validator, model_validator


class Role(str, Enum):
    """Valid roles in a chat conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    """A single message in a conversation."""
    role: Role
    content: str

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        """Content cannot be empty or contain only whitespace."""
        if not v.strip():
            raise ValueError("Message content cannot be empty or whitespace")
        return v.strip()


class Conversation(BaseModel):
    """
    A complete conversation ready for training.
    Format: list of messages in chronological order.
    """
    messages: list[Message]
    source: str = "unknown"       # Origin of the data
    language: str = "en"          # Detected language
    quality_score: float = 1.0    # Quality score (0 to 1)

    @model_validator(mode="after")
    def validate_conversation_structure(self) -> "Conversation":
        """
        Validates that the conversation has a coherent structure:
        - At least 2 messages (one user and one assistant)
        - Must end with an assistant message
        - Cannot contain consecutive messages from the same role
        """
        messages = self.messages

        if len(messages) < 2:
            raise ValueError("Conversation must have at least 2 messages")

        if messages[-1].role != Role.ASSISTANT:
            raise ValueError("Conversation must end with an assistant message")

        # Check for consecutive messages with the same role
        for i in range(1, len(messages)):
            if messages[i].role == messages[i - 1].role:
                raise ValueError(
                    f"Consecutive messages with the same role at index {i}: "
                    f"{messages[i].role}"
                )

        return self

    @property
    def num_turns(self) -> int:
        """Number of conversation turns (user/assistant pairs)."""
        return sum(1 for m in self.messages if m.role == Role.USER)

    @property
    def total_chars(self) -> int:
        """Total number of characters in the conversation."""
        return sum(len(m.content) for m in self.messages)