"""Replay public traces through the production FastAPI application."""

from __future__ import annotations

import time
from collections.abc import Iterable
from typing import Any

from fastapi.testclient import TestClient

from shl_agent.evaluation.models import (
    ApiRecommendation,
    ApiTurnResult,
    ConversationTrace,
    TraceReplayResult,
)

_RESPONSE_KEYS = {"reply", "recommendations", "end_of_conversation"}
_RECOMMENDATION_KEYS = {"name", "url", "test_type"}


class TraceReplayer:
    """Replay traces via POST /chat with complete stateless history."""

    def __init__(self, client: TestClient, max_messages: int) -> None:
        self._client = client
        self._max_messages = max_messages

    def replay_many(self, traces: Iterable[ConversationTrace]) -> tuple[TraceReplayResult, ...]:
        """Replay multiple traces."""
        return tuple(self.replay(trace) for trace in traces)

    def replay(self, trace: ConversationTrace) -> TraceReplayResult:
        """Replay one trace until completion or the assignment turn limit."""
        history: list[dict[str, str]] = []
        observed_turns: list[ApiTurnResult] = []
        failures: list[str] = []
        stopped_reason = "trace_exhausted"

        for turn in trace.turns:
            next_history = [*history, {"role": "user", "content": turn.user_message}]
            if len(next_history) > self._max_messages:
                stopped_reason = "turn_limit_reached"
                failures.append(
                    f"turn {turn.turn_number}: request would exceed {self._max_messages} messages"
                )
                break

            started = time.perf_counter()
            response = self._client.post("/chat", json={"messages": next_history})
            latency_ms = round((time.perf_counter() - started) * 1000, 3)
            if response.status_code != 200:
                observed_turns.append(
                    ApiTurnResult(
                        turn_number=turn.turn_number,
                        request_messages=len(next_history),
                        reply="",
                        recommendations=(),
                        end_of_conversation=False,
                        latency_ms=latency_ms,
                        schema_compliant=False,
                        failure=f"HTTP {response.status_code}",
                    )
                )
                stopped_reason = "api_error"
                break

            body = response.json()
            schema_compliant = _is_schema_compliant(body)
            recommendations = _parse_recommendations(body.get("recommendations", []))
            reply = body.get("reply", "")
            end_of_conversation = body.get("end_of_conversation", False)
            observed_turns.append(
                ApiTurnResult(
                    turn_number=turn.turn_number,
                    request_messages=len(next_history),
                    reply=reply if isinstance(reply, str) else "",
                    recommendations=recommendations,
                    end_of_conversation=(
                        end_of_conversation if isinstance(end_of_conversation, bool) else False
                    ),
                    latency_ms=latency_ms,
                    schema_compliant=schema_compliant,
                    failure=None if schema_compliant else "schema violation",
                )
            )

            history = [
                *next_history,
                {"role": "assistant", "content": reply if isinstance(reply, str) else ""},
            ]
            if end_of_conversation is True:
                stopped_reason = "end_of_conversation"
                break

        return TraceReplayResult(
            trace=trace,
            turns=tuple(observed_turns),
            stopped_reason=stopped_reason,
            failures=tuple(failures),
        )


def _is_schema_compliant(body: Any) -> bool:
    if not isinstance(body, dict) or set(body) != _RESPONSE_KEYS:
        return False
    if not isinstance(body["reply"], str) or not body["reply"]:
        return False
    if not isinstance(body["end_of_conversation"], bool):
        return False
    recommendations = body["recommendations"]
    if not isinstance(recommendations, list) or len(recommendations) > 10:
        return False
    return all(
        isinstance(item, dict)
        and set(item) == _RECOMMENDATION_KEYS
        and all(isinstance(item[key], str) and item[key] for key in _RECOMMENDATION_KEYS)
        for item in recommendations
    )


def _parse_recommendations(raw: Any) -> tuple[ApiRecommendation, ...]:
    if not isinstance(raw, list):
        return ()
    recommendations: list[ApiRecommendation] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        url = item.get("url")
        test_type = item.get("test_type")
        if isinstance(name, str) and isinstance(url, str) and isinstance(test_type, str):
            recommendations.append(ApiRecommendation(name=name, url=url, test_type=test_type))
    return tuple(recommendations)
