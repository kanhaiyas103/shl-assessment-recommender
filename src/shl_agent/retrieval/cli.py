"""Command-line entry point for building the FAISS embedding index."""

import asyncio
import json
import logging
import sys
from dataclasses import asdict

from shl_agent.retrieval.catalog_artifact import CatalogArtifactRepository
from shl_agent.retrieval.embedding_service import EmbeddingService, SentenceTransformerBackend
from shl_agent.retrieval.faiss_index import FaissIndexBuilder
from shl_agent.retrieval.index_pipeline import EmbeddingIndexPipeline
from shl_agent.retrieval.text_builder import AssessmentTextBuilder
from shl_agent.utils.logging import configure_logging
from shl_agent.utils.settings import get_settings

logger = logging.getLogger(__name__)


async def build_index() -> int:
    """Build the configured embedding index and print a summary."""
    settings = get_settings()
    configure_logging(settings.log_level)

    embedding_service = EmbeddingService(
        SentenceTransformerBackend(
            model_name=settings.embedding_model,
            batch_size=settings.embedding_batch_size,
        )
    )
    pipeline = EmbeddingIndexPipeline(
        catalog_repository=CatalogArtifactRepository(
            settings.catalog_path,
            settings.catalog_manifest_path,
        ),
        text_builder=AssessmentTextBuilder(),
        embedding_service=embedding_service,
        index_builder=FaissIndexBuilder(),
        index_path=settings.faiss_index_path,
        metadata_path=settings.index_metadata_path,
        manifest_path=settings.embedding_manifest_path,
        report_path=settings.embedding_report_path,
    )

    manifest, report = await pipeline.build()
    sys.stdout.write(
        json.dumps(
            {"manifest": asdict(manifest), "report": asdict(report)},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    logger.info("Embedding index build command finished")
    return 0


def main() -> int:
    """Run the async embedding index build command."""
    return asyncio.run(build_index())


if __name__ == "__main__":
    raise SystemExit(main())
