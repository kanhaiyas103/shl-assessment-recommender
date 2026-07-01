"""Tests for deterministic ChatResponse construction."""

import pytest

from shl_agent.models.assessment import Assessment
from shl_agent.models.enums import TestType as AssessmentTestType
from shl_agent.services.conversation_engine import OutputValidator


def assessment(
    url: str = "https://www.shl.com/products/product-catalog/view/python-new/",
) -> Assessment:
    return Assessment(
        "python",
        "Python New",
        url,
        (AssessmentTestType.KNOWLEDGE_AND_SKILLS,),
        "Python skills.",
    )


def test_output_validator_builds_schema_from_assessments_only() -> None:
    response = OutputValidator().build_response(
        reply="Here are matches.",
        recommendations=(assessment(), assessment()),
        end_of_conversation=True,
    )

    assert response.recommendations[0].name == "Python New"
    assert len(response.recommendations) == 1
    assert response.end_of_conversation is True


def test_output_validator_rejects_non_catalog_urls() -> None:
    unsafe = assessment()
    object.__setattr__(unsafe, "url", "https://evil.example.com/test")

    with pytest.raises(ValueError, match="SHL catalog"):
        OutputValidator().build_response(
            reply="Bad",
            recommendations=(unsafe,),
            end_of_conversation=False,
        )
