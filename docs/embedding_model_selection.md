# Embedding model selection

Phase 4 compares practical SentenceTransformer choices for the SHL catalog index. The index
contains fewer than 500 assessments, so the best model is the one that preserves semantic recall
while staying small and fast enough for Render deployment.

| Model | Dimension | Retrieval quality | Recall@10 suitability | Speed | Memory / deployment size | Notes |
|---|---:|---|---|---|---|---|
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | Good general semantic matching | Strong for short assessment descriptions and skill queries | Fastest | Smallest | Best operational trade-off for this assignment |
| `BAAI/bge-small-en-v1.5` | 384 | Very good retrieval-focused embeddings | Strong, often better than MiniLM for retrieval wording | Fast | Small | Good alternative, but may require query instruction conventions |
| `BAAI/bge-base-en-v1.5` | 768 | Best semantic capacity among these three | Strongest expected recall when latency is less constrained | Slower | Larger | More expensive for a small catalog and lightweight deployment |

Chosen model: `sentence-transformers/all-MiniLM-L6-v2`.

Why: the catalog is small, the semantic documents are carefully labeled, and the take-home project
will be evaluated through API responsiveness as well as result quality. MiniLM gives 384-dimensional
vectors, quick CPU inference, low memory use, and widely available SentenceTransformer support. It
is a defensible production default for Render while leaving the model name configurable for later
experiments with BGE models.
