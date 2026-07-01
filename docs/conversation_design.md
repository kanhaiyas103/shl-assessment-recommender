# Phase 6 conversation design

The conversation layer is stateless. Every request carries the full `messages[]` history, and
`ConversationHistoryResolver` reconstructs current requirements from that history on each call.
No server-side memory is required.

## Hallucination prevention

- Retrieval and recommendation objects are separated from prose generation.
- The response composer can write conversational text, but it never constructs recommendation
  identities.
- `OutputValidator` builds `RecommendationResponse` objects only from trusted `Assessment`
  instances already present in catalog metadata.
- URLs are accepted only when they come from SHL catalog records.
- Comparison output uses only grounded fields: name, test type, duration, remote testing, and
  adaptive/IRT support.
- Prompt-injection and off-topic requests are refused before retrieval.

## Deterministic recommendation objects

The engine selects recommendations from `RetrievalEvidence.results`, then `OutputValidator`
deduplicates by assessment ID, caps output at 10, and maps trusted catalog fields into the strict
API schema. The LLM is not allowed to create or modify recommendation objects.

## Clarification budget

`ConversationPolicy` enforces the assignment's eight-message maximum and limits clarification
questions to two. `RecommendationReadinessPolicy` clarifies only when the request has no actionable
anchor or retrieval evidence is weak and there is enough remaining conversation budget. This avoids
unnecessary clarification and biases toward completing the task within the evaluator's turn limit.

## Refinement behavior

`RefinementEngine` merges new constraints into existing requirements. Valid previous constraints
are preserved unless the user explicitly removes or replaces them. This supports natural turns such
as "make it under 30 minutes" after a previous skill request.
