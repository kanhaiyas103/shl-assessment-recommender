"""Stateless conversation domain models."""

from dataclasses import dataclass

from shl_agent.models.enums import ConversationRole


@dataclass(frozen=True, slots=True)
class ConversationMessage:
    """A normalized message supplied by the API boundary."""

    role: ConversationRole
    content: str

    def __post_init__(self) -> None:
        """Reject content that cannot contribute to a conversation."""
        if not self.content.strip():
            raise ValueError("message content must not be blank")


@dataclass(frozen=True, slots=True)
class Conversation:
    """A complete conversation carried by one stateless request."""

    messages: tuple[ConversationMessage, ...]

    def __post_init__(self) -> None:
        """Enforce the assignment's message-order and turn-budget invariants."""
        if not self.messages:
            raise ValueError("conversation must contain at least one message")
        if len(self.messages) > 8:
            raise ValueError("conversation cannot exceed eight messages")
        if self.messages[0].role is not ConversationRole.USER:
            raise ValueError("conversation must start with a user message")
        if self.messages[-1].role is not ConversationRole.USER:
            raise ValueError("latest message must be from the user")
        if any(
            current.role is following.role
            for current, following in zip(self.messages, self.messages[1:], strict=False)
        ):
            raise ValueError("conversation roles must alternate")

    @property
    def remaining_message_budget(self) -> int:
        """Return messages remaining under the evaluator's hard cap."""
        return 8 - len(self.messages)
