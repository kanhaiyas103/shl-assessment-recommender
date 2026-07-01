FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m pip install --upgrade pip \
    && python -m pip wheel --wheel-dir /wheels .

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SHL_HOST=0.0.0.0 \
    SHL_PORT=8000

RUN groupadd --system app \
    && useradd --system --gid app --create-home app

WORKDIR /app

COPY --from=builder /wheels /wheels
RUN python -m pip install --no-cache-dir /wheels/* \
    && rm -rf /wheels

COPY --chown=app:app data ./data

USER app

EXPOSE 8000

CMD ["sh", "-c", "uvicorn shl_agent.api.app:create_app --factory --host 0.0.0.0 --port ${PORT:-${SHL_PORT:-8000}}"]
