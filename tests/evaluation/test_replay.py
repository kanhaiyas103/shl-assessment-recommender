"""Tests for FastAPI trace replay."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from shl_agent.evaluation.models import ConversationTrace, TraceTurn
from shl_agent.evaluation.replay import TraceReplayer


def test_replay_sends_complete_history_and_stops_on_end() -> None:
    app = FastAPI()
    observed_lengths: list[int] = []

    @app.post("/chat")
    async def chat(payload: dict[str, list[dict[str, str]]]) -> dict[str, object]:
        observed_lengths.append(len(payload["messages"]))
        return {
            "reply": "done",
            "recommendations": [
                {
                    "name": "Core Java",
                    "url": "https://www.shl.com/products/product-catalog/view/core-java-new/",
                    "test_type": "K",
                }
            ],
            "end_of_conversation": True,
        }

    trace = ConversationTrace(
        trace_id="C1",
        source_path="C1.md",
        turns=(
            TraceTurn(1, "Need Java", "", (), False),
            TraceTurn(2, "Thanks", "", (), True),
        ),
    )

    with TestClient(app) as client:
        result = TraceReplayer(client, max_messages=8).replay(trace)

    assert observed_lengths == [1]
    assert result.stopped_reason == "end_of_conversation"
    assert result.turns[0].schema_compliant is True


def test_replay_records_turn_limit_before_invalid_request() -> None:
    app = FastAPI()

    @app.post("/chat")
    async def chat(_payload: dict[str, list[dict[str, str]]]) -> dict[str, object]:
        return {"reply": "continue", "recommendations": [], "end_of_conversation": False}

    trace = ConversationTrace(
        trace_id="C1",
        source_path="C1.md",
        turns=(
            TraceTurn(1, "First", "", (), False),
            TraceTurn(2, "Second", "", (), False),
        ),
    )

    with TestClient(app) as client:
        result = TraceReplayer(client, max_messages=2).replay(trace)

    assert result.stopped_reason == "turn_limit_reached"
    assert result.failures == ("turn 2: request would exceed 2 messages",)


def test_replay_records_schema_violation() -> None:
    app = FastAPI()

    @app.post("/chat")
    async def chat(_payload: dict[str, list[dict[str, str]]]) -> dict[str, object]:
        return {"reply": "", "recommendations": [], "end_of_conversation": False}

    trace = ConversationTrace(
        trace_id="C1",
        source_path="C1.md",
        turns=(TraceTurn(1, "Need Java", "", (), False),),
    )

    with TestClient(app) as client:
        result = TraceReplayer(client, max_messages=8).replay(trace)

    assert result.turns[0].schema_compliant is False
    assert result.turns[0].failure == "schema violation"


def test_replay_many_returns_one_result_per_trace() -> None:
    app = FastAPI()

    @app.post("/chat")
    async def chat(_payload: dict[str, list[dict[str, str]]]) -> dict[str, object]:
        return {"reply": "done", "recommendations": [], "end_of_conversation": True}

    traces = (
        ConversationTrace("C1", "C1.md", (TraceTurn(1, "Need Java", "", (), False),)),
        ConversationTrace("C2", "C2.md", (TraceTurn(1, "Need SQL", "", (), False),)),
    )

    with TestClient(app) as client:
        results = TraceReplayer(client, max_messages=8).replay_many(traces)

    assert [result.trace.trace_id for result in results] == ["C1", "C2"]
