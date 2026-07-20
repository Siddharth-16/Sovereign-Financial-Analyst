# Sovereign Financial Analyst -- Phase 1 Eval Report

Questions evaluated: **28**

## Headline numbers

- **Retrieval recall@k:** 100.0% (28/28)
- **Company routing accuracy:** 100.0%
- **Section routing accuracy:** 100.0%
- **Groundedness (faithfulness):** 92.9% of graded answers (28 graded)

## By section

| Section | N | Retrieval recall@k | Groundedness |
|---|---|---|---|
| business | 7 | 100.0% | 100.0% |
| risk_factors | 7 | 100.0% | 71.4% |
| mdna | 7 | 100.0% | 100.0% |
| financial_statements | 7 | 100.0% | 100.0% |

## Misses

- **risk-05** (risk_factors): "What risks does Meta describe related to data privacy and regulation?"
  - groundedness fail: The provided context only mentions potential future changes to existing restrictions, but does not mention uncertainty in complying with new restrictions. unsupported claims: Uncertainty in complying with new restrictions and requirements under the DMA.
- **risk-06** (risk_factors): "What risk factors does Pfizer disclose related to drug development and clinical trials?"
  - groundedness fail: The provided context only mentions risk factors related to litigation, government investigations, and commercial matters, but does not explicitly mention the impact of technical or advisory committee recommendations on product sales and regulatory approvals, nor the specific challenges posed by regulatory agency requirements in the approval process. unsupported claims: Receiving or maintaining favorable recommendations from technical or advisory committees, such as the ACIP or FDA Advisory Committee, which may impact product sales and regulatory approvals.; Regulatory agency requirements resulting in a more challenging, expensive, and lengthy regulatory approval process due to requests for additional or more extensive clinical trials.
