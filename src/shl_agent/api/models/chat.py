"""Strict schemas required by the assignment replay harness."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

from shl_agent.models.enums import TestType

MessageContent = Annotated[str, Field(min_length=1, max_length=20_000)]


class StrictApiModel(BaseModel):
    """Base schema that rejects evaluator-breaking fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ChatMessage(StrictApiModel):
    """One user or assistant message from the complete conversation history."""

    role: Literal["user", "assistant"]
    content: MessageContent


class ChatRequest(StrictApiModel):
    """Stateless chat request containing the complete ordered history."""

    messages: Annotated[list[ChatMessage], Field(min_length=1, max_length=8)]

    @model_validator(mode="after")
    def validate_conversation_order(self) -> "ChatRequest":
        """Reject histories that cannot represent a user-driven conversation."""
        if self.messages[0].role != "user":
            raise ValueError("conversation must start with a user message")
        if self.messages[-1].role != "user":
            raise ValueError("latest message must be from the user")
        if any(
            current.role == following.role
            for current, following in zip(self.messages, self.messages[1:], strict=False)
        ):
            raise ValueError("user and assistant messages must alternate")
        return self


class RecommendationResponse(StrictApiModel):
    """Canonical catalog identity exposed by the API."""

    name: Annotated[str, Field(min_length=1)]
    url: HttpUrl
    test_type: TestType


class ChatResponse(StrictApiModel):
    """Exact response contract for POST /chat."""

    reply: Annotated[str, Field(min_length=1)]
    recommendations: Annotated[list[RecommendationResponse], Field(max_length=10)]
    end_of_conversation: bool


class HealthResponse(StrictApiModel):
    """Exact response contract for GET /health."""

    status: Literal["ok"] = "ok"
