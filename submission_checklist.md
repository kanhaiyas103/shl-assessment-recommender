# Submission Checklist

## Application

- [x] FastAPI application factory implemented.
- [x] `GET /health` returns exactly `{"status": "ok"}`.
- [x] `POST /chat` accepts complete stateless message history.
- [x] `POST /chat` returns exactly `reply`, `recommendations`, and `end_of_conversation`.
- [x] Recommendation objects return exactly `name`, `url`, and `test_type`.
- [x] Clarification/refusal responses return empty recommendation arrays.

## Catalog and retrieval artifacts

- [x] `data/processed/catalog.json` generated.
- [x] `data/processed/catalog_manifest.json` generated.
- [x] `data/indexes/index.faiss` generated.
- [x] `data/indexes/metadata.json` generated.
- [x] Embedding manifest/report generated.

## Quality

- [x] Ruff check passes.
- [x] Ruff format check passes.
- [x] MyPy passes.
- [x] Pytest passes.
- [x] API contract tests pass.
- [x] Evaluation harness completed.

## Evaluation

- [x] Public trace ZIP parser implemented.
- [x] Conversation replay uses FastAPI `TestClient`.
- [x] Recall@10 computed from expected final catalog URLs.
- [x] Hallucination rate verified.
- [x] `data/evaluation/evaluation_report.json` generated.
- [x] `data/evaluation/evaluation_summary.md` generated.

## Documentation

- [x] README completed.
- [x] `APPROACH.md` completed.
- [x] API documentation completed.
- [x] Architecture documentation completed.
- [x] Submission checklist completed.

## Deployment

- [x] Dockerfile present.
- [x] Docker container runs as non-root user.
- [x] Docker startup command uses FastAPI app factory.
- [x] Docker command honors hosted `PORT` with fallback.
- [x] Render blueprint present.
- [x] Render health check path is `/health`.
- [x] Required runtime artifacts are allowlisted for submission.

## Manual final checks

- [ ] Build Docker image locally.
- [ ] Run Docker container locally.
- [ ] Call `/health` against the container.
- [ ] Call `/chat` against the container.
- [ ] Configure Render environment variables.
- [ ] Deploy to Render.
- [ ] Confirm Render health check passes.
