# Phase 5 retrieval design

The retrieval engine is intentionally deterministic and LLM-free. Its job is to maximize
candidate recall before the later conversation/recommendation layer decides whether to clarify,
recommend, compare, or refuse.

## Pipeline

1. `RequirementResolver` normalizes resolved conversation requirements into a canonical retrieval
   representation.
2. `QueryExpansionService` creates bounded deterministic retrieval views, such as original query,
   normalized role, skills, competencies, assessment categories, abbreviations, and synonyms.
3. `SemanticRetriever` searches FAISS for each expanded query and returns similarity-only results.
4. `LexicalRetriever` searches exact names, technologies, abbreviations, categories, and skills.
5. `MetadataFilter` scores constraints softly to preserve recall unless a candidate is explicitly
   excluded.
6. `RankFusionService` combines semantic, lexical, and metadata signals with Reciprocal Rank
   Fusion.
7. `CandidateRanker` computes rerank scores from fused relevance, requirement coverage, diversity,
   metadata match, and semantic/lexical balance.
8. `RetrievalEvidenceBuilder` returns calibrated `RetrievalEvidence` for the future readiness
   policy.

## Why Reciprocal Rank Fusion

RRF is used instead of weighted averaging because semantic, lexical, and metadata scores live on
different scales. A cosine similarity, an exact-name lexical score, and a metadata constraint score
are not naturally comparable. RRF uses rank positions rather than raw score magnitudes, so a strong
candidate from any retriever remains visible, while candidates found by multiple retrievers rise
above single-signal matches. This is a better fit for Recall@10 than a brittle hand-tuned weighted
average.

## Why Top-30 internally

The retrieval engine keeps 30 candidates internally because final recommendation will later return
only 1-10 assessments. Keeping 30 preserves enough room for clarification, comparison, diversity,
duration/language filtering, and LLM grounding without forcing the retriever to make the final
business decision too early.

## Recall@10 improvements

- Query expansion catches alternate wording such as `js`, `javascript`, and category language.
- Semantic retrieval covers fuzzy intent and role descriptions.
- Lexical retrieval protects exact technology and exact assessment-name matches.
- Metadata constraints are soft by default, preventing over-filtering.
- RRF promotes candidates that appear in multiple independent retrieval views.
- Candidate ranking balances relevance with diversity so near-duplicate assessment families do not
  crowd out useful alternatives.
