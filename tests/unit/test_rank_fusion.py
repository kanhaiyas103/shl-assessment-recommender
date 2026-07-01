"""Tests for Reciprocal Rank Fusion."""

from shl_agent.retrieval.metadata_filter import MetadataSignal
from shl_agent.retrieval.rank_fusion import RankFusionService
from shl_agent.retrieval.semantic_retriever import CandidateSignal


def test_rank_fusion_rewards_multi_signal_candidates() -> None:
    fused = RankFusionService().fuse(
        semantic={
            "a": CandidateSignal("a", 0.9, 1, ("python",)),
            "b": CandidateSignal("b", 0.8, 2, ("python",)),
        },
        lexical={"a": CandidateSignal("a", 1.0, 1, ("python",))},
        metadata={
            "a": MetadataSignal("a", 1.0, ("duration",), 1, 1),
            "b": MetadataSignal("b", 0.5, (), 1, 0),
        },
    )

    assert fused[0].assessment_id == "a"
    assert fused[0].fused_score > fused[1].fused_score
    assert "duration" in fused[0].matched_terms
