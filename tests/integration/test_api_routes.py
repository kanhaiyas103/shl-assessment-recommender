"""API route tests for the assembled FastAPI application."""

from collections.abc import Generator
from types import SimpleNamespace
from typing import cast

import pytest
from fastapi.testclient import TestClient

from shl_agent.api.app import create_app
from shl_agent.api.models.chat import ChatMessage, ChatResponse, RecommendationResponse
from shl_agent.models.enums import TestType as AssessmentTestType
from shl_agent.retrieval.embedding_service import EmbeddingError
from shl_agent.services.container import ApplicationContainer
from shl_agent.utils.settings import Settings


class FakeConversationEngine:
    """Fake conversation engine for API tests."""

    async def respond(self, messages: list[ChatMessage]) -> ChatResponse:
        latest = messages[-1].content.casefold()
        if "fail model" in latest:
            raise EmbeddingError("backend unavailable")
        if "vague" in latest or latest == "assessment":
            return ChatResponse(
                reply="What role or skill should the assessment focus on?",
                recommendations=[],
                end_of_conversation=False,
            )
        if "compare" in latest:
            return ChatResponse(
                reply="Here is a grounded comparison using catalog fields only.",
                recommendations=[
                    self._recommendation("Python New", "python-new"),
                    self._recommendation("Java New", "java-new"),
                ],
                end_of_conversation=False,
            )
        if "ignore" in latest or "weather" in latest:
            return ChatResponse(
                reply="I can only help recommend SHL Individual Test Solutions from the catalog.",
                recommendations=[],
                end_of_conversation=False,
            )
        return ChatResponse(
            reply="I found SHL assessments for your request.",
            recommendations=[self._recommendation("Python New", "python-new")],
            end_of_conversation=True,
        )

    @staticmethod
    def _recommendation(name: str, slug: str) -> RecommendationResponse:
        return RecommendationResponse(
            name=name,
            url=f"https://www.shl.com/products/product-catalog/view/{slug}/",
            test_type=AssessmentTestType.KNOWLEDGE_AND_SKILLS,
        )


def fake_container() -> ApplicationContainer:
    return cast(
        ApplicationContainer,
        SimpleNamespace(
            settings=Settings(_env_file=None, app_env="test"),
            logger=None,
            catalog_store=None,
            embedding_service=None,
            retrieval_engine=None,
            conversation_engine=FakeConversationEngine(),
            provider_status="disabled",
        ),
    )


@pytest.fixture
def client() -> Generator[TestClient]:
    with TestClient(create_app(fake_container())) as test_client:
        yield test_client


def test_health_schema_exact(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_recommendation_response(client: TestClient) -> None:
    response = client.post("/chat", json={"messages": [{"role": "user", "content": "Python"}]})

    assert response.status_code == 200
    assert response.json() == {
        "reply": "I found SHL assessments for your request.",
        "recommendations": [
            {
                "name": "Python New",
                "url": "https://www.shl.com/products/product-catalog/view/python-new/",
                "test_type": "K",
            }
        ],
        "end_of_conversation": True,
    }


def test_chat_clarification_has_empty_recommendations(client: TestClient) -> None:
    response = client.post("/chat", json={"messages": [{"role": "user", "content": "vague"}]})

    body = response.json()
    assert response.status_code == 200
    assert body["recommendations"] == []
    assert body["end_of_conversation"] is False


def test_chat_comparison_response(client: TestClient) -> None:
    response = client.post("/chat", json={"messages": [{"role": "user", "content": "compare"}]})

    body = response.json()
    assert response.status_code == 200
    assert len(body["recommendations"]) == 2
    assert "catalog fields only" in body["reply"]


def test_chat_refinement_replay(client: TestClient) -> None:
    response = client.post(
        "/chat",
        json={
            "messages": [
                {"role": "user", "content": "Python"},
                {"role": "assistant", "content": "I found Python New."},
                {"role": "user", "content": "under 30 minutes"},
            ]
        },
    )

    assert response.status_code == 200
    assert response.json()["recommendations"][0]["name"] == "Python New"


def test_chat_refusal_response(client: TestClient) -> None:
    response = client.post("/chat", json={"messages": [{"role": "user", "content": "weather"}]})

    body = response.json()
    assert response.status_code == 200
    assert body["recommendations"] == []
    assert "SHL Individual Test Solutions" in body["reply"]


def test_invalid_schema_rejected(client: TestClient) -> None:
    response = client.post("/chat", json={"messages": [{"role": "system", "content": "bad"}]})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_llm_failure_fallback_returns_safe_error(client: TestClient) -> None:
    response = client.post("/chat", json={"messages": [{"role": "user", "content": "fail model"}]})

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "llm_error"
