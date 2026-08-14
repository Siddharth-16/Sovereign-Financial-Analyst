# Sovereign Financial Analyst — Dev Evaluation

| Metric                      |    Rule-Based |       Agentic |
| --------------------------- | ------------: | ------------: |
| Company routing accuracy    |        100.0% |        100.0% |
| Section routing accuracy    |        100.0% |        100.0% |
| Metadata retrieval hit rate |        100.0% |        100.0% |
| Evidence retrieval recall@8 | 63.0% (29/46) | 69.6% (32/46) |
| Evidence question success   |         57.1% |         57.1% |
| Answer fact completeness    | 49.3% (35/71) | 83.1% (59/71) |
| Manual answer groundedness  | 71.4% (20/28) | 71.4% (20/28) |
| Incorrect abstentions       |             0 |             0 |
| Tool errors                 |             0 |             0 |
| Avg tool calls / question   |           N/A |           1.0 |
| Mean application latency    |         23.4s |         35.7s |
| Median application latency  |         22.7s |         37.1s |

## Interpretation

Both approaches achieved 100% company and section routing accuracy on the
28-question development regression benchmark.

The rule-based pipeline was faster, with a mean latency of 23.4 seconds,
but retrieved less gold evidence and produced substantially less complete
answers.

The agentic pipeline improved evidence recall from 63.0% to 69.6% and
answer fact completeness from 49.3% to 83.1%, while maintaining the same
71.4% manual answer groundedness and zero tool errors.

The groundedness errors differed by approach. The rule-based pipeline was
strong on Risk Factors but weaker on Financial Statements, while the
agentic pipeline's deterministic financial-answer path produced fully
grounded financial-statement answers.

The rule-based pipeline performs retrieval directly and therefore does not
use agent tool calls. Tool-call metrics apply only to the agentic pipeline.

## Evaluation Notes

- Evaluation uses a 28-question development/regression benchmark spanning
  Business, Risk Factors, MD&A, and Financial Statements.
- Evidence retrieval is evaluated against 46 gold evidence items.
- Fact completeness is evaluated against 71 expected facts.
- Groundedness was manually evaluated by comparing each generated answer
  with the exact retrieved filing context available during generation.
- Missing expected information affects completeness, not groundedness.
- The benchmark was used during development and should not be interpreted
  as a held-out test set.
