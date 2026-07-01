"""Deterministic lexical retrieval over catalog text and metadata."""

import re
from collections import Counter
from dataclasses import dataclass
from typing import Protocol

from shl_agent.models.assessment import Assessment
from shl_agent.retrieval.requirements import CanonicalRequirements
from shl_agent.retrieval.semantic_retriever import CandidateSignal

_TOKEN_RE = re.compile(r"[a-z0-9+#.]{2,}")


@dataclass(frozen=True, slots=True)
class LexicalDocument:
    """Tokenized lexical view of one assessment."""

    assessment_id: str
    tokens: Counter[str]
    phrase_text: str


class LexicalStore(Protocol):
    """Catalog listing capability required by lexical retrieval."""

    def all_assessments(self) -> tuple[Assessment, ...]:
        """Return all catalog assessments."""
        raise NotImplementedError


class LexicalRetriever:
    """Keyword, exact-name, abbreviation, category, and skill retriever."""

    def __init__(self, store: LexicalStore) -> None:
        self._documents = tuple(
            self._document(assessment) for assessment in store.all_assessments()
        )

    def retrieve(
        self,
        requirements: CanonicalRequirements,
        *,
        limit: int = 80,
    ) -> dict[str, CandidateSignal]:
        """Return lexical matches with normalized scores."""
        query_terms = self._query_terms(requirements)
        scored: list[CandidateSignal] = []
        for document in self._documents:
            score, matched = self._score(document, requirements, query_terms)
            if score > 0:
                scored.append(
                    CandidateSignal(document.assessment_id, score, 0, tuple(sorted(matched)))
                )

        ordered = sorted(scored, key=lambda item: (-item.score, item.assessment_id))[:limit]
        return {
            signal.assessment_id: CandidateSignal(
                signal.assessment_id,
                signal.score,
                rank,
                signal.matched_terms,
            )
            for rank, signal in enumerate(ordered, start=1)
        }

    @staticmethod
    def _document(assessment: Assessment) -> LexicalDocument:
        text = " ".join(
            (
                assessment.name,
                assessment.description,
                " ".join(test_type.value for test_type in assessment.test_types),
                " ".join(assessment.job_levels),
                " ".join(assessment.languages),
            )
        ).casefold()
        return LexicalDocument(
            assessment_id=assessment.assessment_id,
            tokens=Counter(_TOKEN_RE.findall(text)),
            phrase_text=text,
        )

    @staticmethod
    def _query_terms(requirements: CanonicalRequirements) -> tuple[str, ...]:
        terms = (
            *requirements.skills,
            *requirements.competencies,
            *requirements.assessment_names,
            *(test_type.value.casefold() for test_type in requirements.test_types),
            requirements.role or "",
            requirements.seniority or "",
        )
        return tuple(term for term in terms if term)

    @staticmethod
    def _score(
        document: LexicalDocument,
        requirements: CanonicalRequirements,
        query_terms: tuple[str, ...],
    ) -> tuple[float, set[str]]:
        score = 0.0
        matched: set[str] = set()
        for name in requirements.assessment_names:
            if name and name in document.phrase_text:
                score += 1.0
                matched.add(name)
        for term in query_terms:
            term_tokens = _TOKEN_RE.findall(term.casefold())
            token_hits = sum(document.tokens.get(token, 0) for token in term_tokens)
            if token_hits:
                score += min(0.45, 0.12 * token_hits)
                matched.add(term)
            if len(term) >= 3 and term in document.phrase_text:
                score += 0.3
                matched.add(term)
        return min(1.0, score), matched
