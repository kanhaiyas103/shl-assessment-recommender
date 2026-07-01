"""Tests for evaluator-facing API schema invariants."""

import pytest
from pydantic import ValidationError

from shl_agent.api.models import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    RecommendationResponse,
)
from shl_agent.models.enums import TestType as AssessmentTestType


def test_chat_request_accepts_valid_alternating_history() -> None:
    request = ChatRequest(
        messages=[
            ChatMessage(role="user", content="I need a Java assessment."),
            ChatMessage(role="assistant", content="What seniority?"),
            ChatMessage(role="user", content="Mid-level."),
        ]
    )

    assert len(request.messages) == 3


@pytest.mark.parametrize(
    "messages",
    [
        [ChatMessage(role="assistant", content="How can I help?")],
        [
            ChatMessage(role="user", content="Java"),
            ChatMessage(role="user", content="Mid-level"),
        ],
        [
            ChatMessage(role="user", content="Java"),
            ChatMessage(role="assistant", content="What seniority?"),
        ],
    ],
)
def test_chat_request_rejects_invalid_order(messages: list[ChatMessage]) -> None:
    with pytest.raises(ValidationError):
        ChatRequest(messages=messages)


def test_chat_response_serializes_exact_contract() -> None:
    response = ChatResponse(
        reply="A grounded shortlist.",
        recommendations=[
            RecommendationResponse(
                name="Java 8 (New)",
                url="https://www.shl.com/solutions/products/product-catalog/view/java-8-new/",
                test_type=AssessmentTestType.KNOWLEDGE_AND_SKILLS,
            )
        ],
        end_of_conversation=False,
    )

    assert response.model_dump(mode="json") == {
        "reply": "A grounded shortlist.",
        "recommendations": [
            {
                "name": "Java 8 (New)",
                "url": "https://www.shl.com/solutions/products/product-catalog/view/java-8-new/",
                "test_type": "K",
            }
        ],
        "end_of_conversation": False,
    }


def test_api_models_reject_extra_fields() -> None:
    with pytest.raises(ValidationError):
        HealthResponse.model_validate({"status": "ok", "ready": True})
