"""FastAPI application factory and public assignment routes."""

import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from http import HTTPStatus

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from shl_agent.api.models.chat import ChatRequest, ChatResponse, HealthResponse
from shl_agent.retrieval.catalog_artifact import CatalogArtifactError
from shl_agent.retrieval.embedding_service import EmbeddingError
from shl_agent.retrieval.runtime_store import RetrievalStoreError
from shl_agent.services.container import ApplicationContainer, build_application_container

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Attach request IDs and structured latency logs."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Log every request without exposing sensitive values."""
        request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
        request.state.request_id = request_id
        started = time.perf_counter()
        response = await call_next(request)
        latency_ms = round((time.perf_counter() - started) * 1000, 3)
        response.headers["x-request-id"] = request_id
        logger.info(
            "HTTP request completed",
            extra={
                "request_id": request_id,
                "endpoint": request.url.path,
                "latency_ms": latency_ms,
                "status_code": response.status_code,
            },
        )
        return response


def error_response(
    request: Request,
    status_code: int,
    code: str,
    message: str,
) -> JSONResponse:
    """Return a secret-safe error payload."""
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code,
                "message": message,
                "request_id": getattr(request.state, "request_id", None),
            }
        },
    )


async def validation_exception_handler(
    request: Request,
    _exc: Exception,
) -> JSONResponse:
    """Handle invalid request schemas without stack traces."""
    logger.warning(
        "Request validation failed",
        extra={
            "request_id": getattr(request.state, "request_id", None),
            "endpoint": request.url.path,
        },
    )
    return error_response(
        request,
        HTTPStatus.UNPROCESSABLE_ENTITY,
        "validation_error",
        "Request schema is invalid.",
    )


async def catalog_exception_handler(request: Request, _exc: Exception) -> JSONResponse:
    """Handle catalog artifact startup/runtime failures."""
    logger.exception(
        "Catalog failure",
        extra={"request_id": getattr(request.state, "request_id", None)},
    )
    return error_response(
        request,
        HTTPStatus.SERVICE_UNAVAILABLE,
        "catalog_error",
        "Catalog is unavailable.",
    )


async def retrieval_exception_handler(request: Request, _exc: Exception) -> JSONResponse:
    """Handle retrieval metadata failures."""
    logger.exception(
        "Retrieval failure",
        extra={"request_id": getattr(request.state, "request_id", None)},
    )
    return error_response(
        request,
        HTTPStatus.SERVICE_UNAVAILABLE,
        "retrieval_error",
        "Retrieval index is unavailable.",
    )


async def embedding_exception_handler(request: Request, _exc: Exception) -> JSONResponse:
    """Handle embedding/model failures."""
    logger.exception(
        "Embedding failure",
        extra={"request_id": getattr(request.state, "request_id", None)},
    )
    return error_response(
        request,
        HTTPStatus.SERVICE_UNAVAILABLE,
        "llm_error",
        "Model provider is unavailable.",
    )


async def internal_exception_handler(request: Request, _exc: Exception) -> JSONResponse:
    """Handle unexpected failures without exposing implementation details."""
    logger.exception(
        "Unhandled error",
        extra={"request_id": getattr(request.state, "request_id", None)},
    )
    return error_response(
        request,
        HTTPStatus.INTERNAL_SERVER_ERROR,
        "internal_error",
        "An internal error occurred.",
    )


def create_app(container: ApplicationContainer | None = None) -> FastAPI:
    """Create the API shell and attach application-scoped dependencies."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        resolved_container = container or build_application_container()
        app.state.container = resolved_container
        logger.info(
            "Application startup complete",
            extra={
                "provider_status": resolved_container.provider_status,
                "catalog_path": str(resolved_container.settings.catalog_path),
                "index_path": str(resolved_container.settings.faiss_index_path),
            },
        )
        yield

    app = FastAPI(
        title="SHL Assessment Recommendation Agent",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.add_middleware(RequestLoggingMiddleware)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(CatalogArtifactError, catalog_exception_handler)
    app.add_exception_handler(RetrievalStoreError, retrieval_exception_handler)
    app.add_exception_handler(EmbeddingError, embedding_exception_handler)
    app.add_exception_handler(Exception, internal_exception_handler)

    @app.get("/health")
    async def health() -> HealthResponse:
        return HealthResponse()

    @app.post("/chat")
    async def chat(request: ChatRequest, raw_request: Request) -> ChatResponse:
        resolved_container: ApplicationContainer = raw_request.app.state.container
        response = await resolved_container.conversation_engine.respond(request.messages)
        logger.info(
            "Chat response generated",
            extra={
                "request_id": getattr(raw_request.state, "request_id", None),
                "endpoint": "/chat",
                "retrieved_assessment_ids": [
                    recommendation.name for recommendation in response.recommendations
                ],
                "provider_status": resolved_container.provider_status,
            },
        )
        return response

    return app
