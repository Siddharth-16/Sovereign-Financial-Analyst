# Sovereign Financial Analyst — Dev Evaluation

| Metric                      |    Rule-Based |       Agentic |
| --------------------------- | ------------: | ------------: |
| Company routing accuracy    |        100.0% |        100.0% |
| Section routing accuracy    |        100.0% |        100.0% |
| Metadata retrieval hit rate |        100.0% |        100.0% |
| Evidence retrieval recall@8 | 60.9% (28/46) | 71.7% (33/46) |
| Evidence question success   |         57.1% |         64.3% |
| Answer fact completeness    | 45.1% (32/71) | 81.7% (58/71) |
| Manual answer groundedness  | 64.3% (18/28) | 78.6% (22/28) |
| Incorrect abstentions       |             0 |             0 |
| Tool errors                 |           N/A |             0 |
| Avg tool calls / question   |           N/A |           1.0 |
| Mean application latency    |        22.54s |        37.64s |
| Median application latency  |        21.81s |        45.70s |

## Interpretation

Both approaches achieved 100% company and section routing accuracy on the
28-question development/regression benchmark.

The rule-based pipeline was faster, with a mean application latency of
22.54 seconds, but retrieved less gold evidence and produced substantially
less complete answers.

The agentic pipeline improved evidence recall from 60.9% to 71.7% and
answer fact completeness from 45.1% to 81.7%. Manual answer groundedness
also improved from 64.3% to 78.6%, while maintaining 100% routing accuracy
and recording zero tool errors.

These gains came with a latency tradeoff: mean application latency increased
from 22.54 seconds for rule-based execution to 37.64 seconds for agentic
execution.

The groundedness errors also differed by approach. The rule-based pipeline
was generally stronger on Risk Factors but produced several unsupported or
incorrect claims in Business, MD&A, and Financial Statements. The agentic
pipeline reduced these errors, and all seven Financial Statements answers
were manually judged grounded against their retrieved context.

The rule-based pipeline performs retrieval directly and therefore does not
use agent tool calls. Tool-call metrics apply only to the agentic pipeline.

## Evaluation Notes

- Evaluation uses the `dev-v5-tesla-2025-gold-2026-08-14` development/regression
  benchmark with 28 questions spanning Business, Risk Factors, MD&A, and
  Financial Statements.
- Evidence retrieval is evaluated against 46 manually annotated gold evidence
  items at `k=8`.
- Fact completeness is evaluated against 71 expected facts.
- Groundedness was manually evaluated by comparing each generated answer
  against the exact retrieved filing context available during that same run.
- An answer was marked ungrounded if it contained any substantive factual
  claim that was unsupported or contradicted by the retrieved context.
- Missing expected information affects completeness, not groundedness.
- The benchmark was used during development and should not be interpreted as
  a held-out test set.
