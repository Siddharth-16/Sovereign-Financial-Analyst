# Sovereign Financial Analyst — Dev Evaluation

| Metric                      |    Rule-Based |       Agentic |
| --------------------------- | ------------: | ------------: |
| Company routing accuracy    |        100.0% |        100.0% |
| Section routing accuracy    |        100.0% |        100.0% |
| Metadata retrieval hit rate |        100.0% |        100.0% |
| Evidence retrieval recall@8 | 63.0% (29/46) | 69.6% (32/46) |
| Evidence question success   |         57.1% |         60.7% |
| Answer fact completeness    | 43.0% (34/79) | 75.9% (60/79) |
| Groundedness                |           N/A |           N/A |
| Incorrect abstentions       |             0 |             0 |
| Tool errors                 |           N/A |             0 |
| Avg tool calls / question   |           N/A |           1.0 |
| Mean application latency    |        16.38s |        43.44s |
| Median application latency  |        14.05s |        51.52s |

## Interpretation

Both approaches achieved 100% company and section routing accuracy on the
28-question development/regression benchmark.

The rule-based pipeline was faster, with a mean application latency of
16.38 seconds, but retrieved less gold evidence and produced substantially
less complete answers.

The agentic pipeline improved evidence recall from 63.0% to 69.6% and
answer fact completeness from 43.0% to 75.9%.

These gains came with a latency tradeoff: mean application latency increased
from 16.38 seconds for rule-based execution to 43.44 seconds for agentic
execution.

The rule-based pipeline performs retrieval directly and therefore does not
use agent tool calls. Tool-call metrics apply only to the agentic pipeline.

## Evaluation Notes

- Evaluation uses the `dev-v6-reviewed-2026-08-15` development/regression
  benchmark with 28 questions spanning Business, Risk Factors, MD&A, and
  Financial Statements.
- Evidence retrieval is evaluated against 46 annotated gold evidence
  items at `k=8`.
- Fact completeness is evaluated against 79 expected facts.
- Missing expected information affects completeness.
- The benchmark was used during development and should not be interpreted as
  a held-out test set.
