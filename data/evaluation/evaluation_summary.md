# SHL Evaluation Summary

- Trace count: 10
- Overall Mean Recall@10: 0.253
- Pass rate: 0.200
- Schema compliance: 1.000
- Hallucination rate: 0.000
- Average turn count: 3.40
- Average clarification count: 0.50
- Unnecessary clarification rate: 0.000
- Average latency: 256.05 ms

## Behavior summary

The harness replays each public trace through POST /chat using complete stateless conversation history and scores the final recommendation set against the reference catalog URLs in the trace.

## Hallucination summary

No hallucinated recommendations were detected.

## Worst-performing traces

- C10: Recall@10=0.000
- C2: Recall@10=0.000
- C6: Recall@10=0.000

## Best-performing traces

- C3: Recall@10=1.000 failures=1
- C4: Recall@10=0.400
- C8: Recall@10=0.400

## Suggestions for improvement

- Inspect low-recall traces and compare missing expected URLs against retrieval evidence.
- Reduce unnecessary clarifications when the reference trace already recommends.
- Keep hallucination rate at zero by preserving catalog-only output validation.
