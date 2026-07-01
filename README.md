# SHL Assessment Recommendation Agent

A production-ready, stateless FastAPI service that recommends SHL Individual Test
Solutions from a scraped SHL catalog. The system combines an offline catalog pipeline,
SentenceTransformer embeddings, FAISS vector search, lexical search, deterministic
ranking, and a conversation policy layer that keeps recommendations grounded in catalog
metadata.

The application is built for the SHL AI Intern take-home assignment. It exposes exactly
the required public endpoints:

- `GET /health`
- `POST /chat`

## Project overview

The service helps a user clarify assessment needs, compare assessments, refine previous
requirements, and receive one to ten catalog-backed recommendations. The API is fully
stateless: every `/chat` request contains the complete conversation history, and the
server stores no per-user conversation state.

The most important design rule is that catalog identity is deterministic. The assistant
may generate conversational text, but recommendation objects are built only from trusted
SHL catalog records. This prevents invented assessment names and invented URLs.

## Architecture

```text
Client / evaluator
  -> FastAPI API layer
  -> ConversationEngine
  -> ConversationHistoryResolver
  -> RecommendationReadinessPolicy
  -> HybridRetrievalEngine
       -> RequirementResolver
       -> QueryExpansionService
       -> SemanticRetriever / FAISS
       -> LexicalRetriever
       -> MetadataFilter
       -> RankFusionService
       -> CandidateRanker
       -> RetrievalEvidenceBuilder
  -> GroundedResponseComposer
  -> OutputValidator
  -> ChatResponse
```

Application resources are loaded once during FastAPI lifespan startup:

- validated catalog JSON
- catalog manifest
- FAISS metadata
- FAISS index
- SentenceTransformer embedding model
- retrieval engine
- conversation engine
- optional LLM provider configuration

## Tech stack

- Python 3.12
- FastAPI
- Pydantic v2
- BeautifulSoup
- Sentence Transformers
- FAISS
- NumPy
- Uvicorn
- Docker
- Render deployment config
- Ruff, MyPy, Pytest, pre-commit

No LangChain is used. The orchestration is plain Python to keep the design explicit and
interview-defensible.

## Features

- Stateless conversation API
- SHL Individual Test Solutions only
- Catalog-only recommendation names and URLs
- Clarification for materially missing facts
- Recommendation refinement from complete history
- Grounded comparison behavior
- Off-topic and prompt-injection refusal
- Hybrid retrieval with semantic, lexical, metadata, RRF, and reranking signals
- FAISS index built offline
- JSON structured logging
- Contract, unit, integration, and evaluation tests
- Docker and Render deployment support

## Project structure

```text
src/shl_agent/
  api/          FastAPI app and assignment schemas
  evaluation/   Public trace parser, replay harness, metrics, reports, CLI
  models/       Framework-independent domain models
  retrieval/    Text building, embeddings, FAISS, hybrid retrieval, ranking
  scraper/      Offline SHL catalog pipeline
  services/     Conversation engine and dependency composition
  utils/        Settings and structured logging

data/
  processed/    Validated catalog artifact and manifest
  indexes/      FAISS index, vector metadata, embedding manifest/report
  evaluation/   Latest evaluation report and summary

docs/           Architecture and design notes
deployment/     Render configuration
tests/          Unit, integration, contract, and evaluation tests
```

## RAG pipeline

1. Build `data/processed/catalog.json` from the SHL catalog source.
2. Convert every assessment into deterministic embedding text.
3. Generate SentenceTransformer embeddings.
4. Store vectors in `data/indexes/index.faiss`.
5. Store vector-to-assessment mapping separately in `data/indexes/metadata.json`.
6. At runtime, retrieve candidates before response composition.
7. Construct recommendation objects only from catalog metadata.

## Conversation flow

1. Validate incoming `messages`.
2. Reconstruct current requirements from the full stateless history.
3. Classify the latest user intent.
4. Refuse off-topic or prompt-injection requests.
5. Ask one focused clarification only when it can materially improve retrieval.
6. Retrieve and rank candidates.
7. Compose a grounded reply.
8. Validate recommendation objects against catalog metadata.
9. Return the exact assignment response schema.

## Retrieval flow

```text
ConversationRequirements
  -> RequirementResolver
  -> QueryExpansionService
  -> SemanticRetriever
  -> LexicalRetriever
  -> MetadataFilter
  -> RankFusionService
  -> CandidateRanker
  -> RetrievalEvidence
```

RRF is used because semantic, lexical, and metadata signals have different score scales.
Rank fusion preserves recall better than hand-tuned weighted averaging when one signal is
strong and another is sparse.

## Installation

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Optional provider extras:

```powershell
python -m pip install -e ".[dev,gemini]"
python -m pip install -e ".[dev,openrouter]"
```

## Configuration

Configuration is loaded from `SHL_`-prefixed environment variables. See
[.env.example](.env.example).

Important variables:

| Variable | Default | Purpose |
|---|---|---|
| `SHL_APP_ENV` | `local` | `local`, `test`, or `production` |
| `SHL_LOG_LEVEL` | `INFO` | Structured log level |
| `SHL_CATALOG_PATH` | `data/processed/catalog.json` | Validated catalog |
| `SHL_FAISS_INDEX_PATH` | `data/indexes/index.faiss` | FAISS index |
| `SHL_INDEX_METADATA_PATH` | `data/indexes/metadata.json` | Vector metadata |
| `SHL_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Runtime model |
| `SHL_LLM_PROVIDER` | `disabled` | `disabled`, `gemini`, or `openrouter` |
| `SHL_LLM_API_KEY` | empty | Required only when a provider is enabled |
| `SHL_MAX_CONVERSATION_MESSAGES` | `8` | Assignment conversation limit |

Secrets are typed as `SecretStr` and are never logged.

## Running locally

```powershell
uvicorn shl_agent.api.app:create_app --factory --host 0.0.0.0 --port 8000
```

Health check:

```powershell
curl http://localhost:8000/health
```

## Docker

```powershell
docker build -t shl-assessment-agent .
docker run --rm -p 8000:8000 --env-file .env shl-assessment-agent
```

The image runs as a non-root user and loads catalog/index artifacts from `data/`.

## Render deployment

Render configuration is provided in [deployment/render.yaml](deployment/render.yaml).

Deployment expectations:

- runtime: Docker
- health check path: `/health`
- startup: `uvicorn shl_agent.api.app:create_app --factory`
- port: uses Render `PORT` when present, otherwise `SHL_PORT` or `8000`
- required artifacts: `data/processed/catalog.json`, `data/indexes/index.faiss`,
  and `data/indexes/metadata.json`

## API documentation

### `GET /health`

Response:

```json
{
  "status": "ok"
}
```

### `POST /chat`

Request:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Hiring a Java developer"
    }
  ]
}
```

Response:

```json
{
  "reply": "I found SHL assessments for java...",
  "recommendations": [
    {
      "name": "Core Java (Advanced Level) (New)",
      "url": "https://www.shl.com/products/product-catalog/view/core-java-advanced-level-new/",
      "test_type": "K"
    }
  ],
  "end_of_conversation": false
}
```

Clarification responses return an empty recommendation list:

```json
{
  "reply": "Should I prioritize backend, frontend, or balanced full-stack coverage?",
  "recommendations": [],
  "end_of_conversation": false
}
```

Error responses are secret-safe:

```json
{
  "error": {
    "code": "validation_error",
    "message": "Request schema is invalid.",
    "request_id": "..."
  }
}
```

See [docs/api.md](docs/api.md) for the full API contract.

## Offline pipelines

Build catalog:

```powershell
shl-build-catalog
```

Build FAISS index:

```powershell
shl-build-index
```

Run evaluation:

```powershell
shl-evaluate --traces-zip C:\Users\KIIT\Downloads\sample_conversations.zip --output-dir data\evaluation
```

Latest evaluation snapshot:

- Mean Recall@10: `0.253`
- Schema compliance: `1.000`
- Hallucination rate: `0.000`
- Pass rate: `0.200`
- Average clarification count: `0.50`

## Quality checks

```powershell
ruff check .
ruff format --check .
mypy
pytest
```

## Future improvements

- Improve final confirmation behavior so closing turns repeat the final shortlist.
- Better handling of long public traces under the eight-message assignment limit.
- Add richer deterministic domain-to-catalog mappings.
- Add provider-backed natural-language composition while preserving deterministic
  recommendation objects.
- Add CI/CD pipeline to run lint, type checks, tests, Docker build, and evaluation.
