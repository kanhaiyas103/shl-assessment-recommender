# Approach

## Problem

The assignment asks for a conversational agent that recommends SHL Individual Test
Solutions. The main risks are hallucinated assessments, invented URLs, and losing context
because the API is stateless. The design therefore treats the SHL catalog as the only
source of truth and reconstructs conversation state from the complete message history on
every request.

## Architecture

The system uses clean architecture boundaries: `api` handles HTTP, `services` handles
conversation decisions, `retrieval` handles candidate generation, `scraper` builds offline
catalog artifacts, and `models` contains framework-independent domain objects. FastAPI
lifespan loads the catalog, FAISS index, embedding model, retrieval engine, and
conversation engine once at startup.

## Catalog pipeline

The catalog pipeline creates a canonical `Assessment` model from the SHL catalog source,
filters to Individual Test Solutions, validates required fields, removes duplicates, and
writes a versioned `catalog.json` plus manifest and report. Later phases never scrape at
request time.

## Embedding pipeline

Each assessment is converted into deterministic semantic text with labeled sections such
as name, description, test type, skills, languages, duration, remote testing, and adaptive
support. SentenceTransformers generates embeddings, FAISS stores vectors, and metadata
keeps vector order mapped to catalog IDs.

## Retrieval

The retrieval engine combines deterministic requirement normalization, bounded query
expansion, FAISS semantic search, lexical search, soft metadata scoring, Reciprocal Rank
Fusion, and candidate reranking. It returns `RetrievalEvidence`, not final prose, so the
conversation layer can decide whether to clarify, recommend, compare, refine, or refuse.

## Conversation engine

The API is stateless. `ConversationHistoryResolver` rebuilds current requirements from
the full message list. `RecommendationReadinessPolicy` asks one focused clarification only
when a missing fact can materially improve retrieval. `ConversationPolicy` enforces the
assignment turn limit and completion semantics.

## Prompt design

The current implementation keeps recommendation objects deterministic and uses grounded
composition for conversational text. If an LLM provider is enabled later, it should receive
only resolved requirements, retrieval evidence, and retrieved catalog records. It should
never construct recommendation objects or receive the full catalog.

## Hallucination prevention

Recommendation names, URLs, and test types are built only from trusted catalog metadata.
`OutputValidator` rejects non-SHL URLs, limits recommendations to ten, removes duplicates,
and preserves schema compliance. Prompt-injection and off-topic requests are refused before
retrieval output is exposed.

## Evaluation

The evaluation harness parses the provided public Markdown conversation traces, replays
each conversation through `POST /chat` using FastAPI `TestClient`, preserves full stateless
history, and computes Recall@10, schema compliance, recommendation count validity, turn
count, clarification rate, refusal/comparison/refinement indicators, latency, and
hallucination rate. Latest snapshot: Mean Recall@10 `0.253`, schema compliance `100%`,
hallucination rate `0%`.

## Trade-offs

The implementation favors deterministic behavior and catalog safety over free-form LLM
creativity. This reduces hallucination risk and makes the design easier to defend, but it
means some nuanced recommendations require additional deterministic mappings or future
LLM-assisted interpretation. FAISS and SentenceTransformers add startup cost, but loading
them once during lifespan keeps per-request latency predictable.

## AI tools used

AI assistance was used to design, implement, test, and document the project iteratively.
All generated code was reviewed through Ruff, MyPy, Pytest, contract tests, and the public
trace evaluation harness. The system itself does not rely on AI-generated catalog truth.

## Future work

Improve final confirmation behavior, tune long-trace handling under the assignment message
limit, add CI/CD, add richer domain-to-catalog mappings, and optionally enable an LLM
provider for response phrasing while keeping recommendation objects deterministic.
