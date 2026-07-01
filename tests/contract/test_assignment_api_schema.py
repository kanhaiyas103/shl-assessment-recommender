"""Contract tests for exact assignment API schemas."""

from types import SimpleNamespace
from typing import cast

from fastapi.testclient import TestClient

from shl_agent.api.app import create_app
from shl_agent.api.models.chat import ChatMessage, ChatResponse, RecommendationResponse
from shl_agent.models.enums import TestType as AssessmentTestType
from shl_agent.services.container import ApplicationContainer
from shl_agent.utils.settings import Settings


class ContractConversationEngine:
    """Return deterministic schema-valid contract responses."""

    async def respond(self, _messages: list[ChatMessage]) -> ChatResponse:
        return ChatResponse(
            reply="Here is one recommendation.",
            recommendations=[
                RecommendationResponse(
                    name="Python New",
                    url="https://www.shl.com/products/product-catalog/view/python-new/",
                    test_type=AssessmentTestType.KNOWLEDGE_AND_SKILLS,
                )
            ],
            end_of_conversation=True,
        )


def client() -> TestClient:
    container = cast(
        ApplicationContainer,
        SimpleNamespace(
            settings=Settings(_env_file=None, app_env="test"),
            logger=None,
            catalog_store=None,
            embedding_service=None,
            retrieval_engine=None,
            conversation_engine=ContractConversationEngine(),
            provider_status="disabled",
        ),
    )
    return TestClient(create_app(container))


def test_chat_success_schema_has_exact_keys() -> None:
    with client() as test_client:
        response = test_client.post(
            "/chat",
            json={"messages": [{"role": "user", "content": "Hiring a Java developer"}]},
        )

    body = response.json()
    assert set(body) == {"reply", "recommendations", "end_of_conversation"}
    assert set(body["recommendations"][0]) == {"name", "url", "test_type"}
    assert 1 <= len(body["recommendations"]) <= 10
    assert body["recommendations"][0]["url"].startswith("https://www.shl.com/")
    assert body["end_of_conversation"] is True
