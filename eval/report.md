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
| business | 7 | 100.0% | 85.7% |
| risk_factors | 7 | 100.0% | 85.7% |
| mdna | 7 | 100.0% | 100.0% |
| financial_statements | 7 | 100.0% | 100.0% |

## Misses

- **biz-01** (business): "What are Nvidia's main business segments and how does the company describe its core products?"
  - groundedness fail: The provided context does not mention 'Gaming' and 'Professional Visualization' as separate business segments. unsupported claims: Nvidia's main business segments are: 1. Datacenter, 2. Gaming, 3. Professional Visualization
- **risk-05** (risk_factors): "What risks does Meta describe related to data privacy and regulation?"
  - groundedness fail: The provided context only mentions the DMA as a specific example of evolving regulation, but does not mention uncertainty around future regulatory actions or compliance with emerging regulations. unsupported claims: Uncertainty around future regulatory actions and their impact on Meta's business.; Compliance with emerging regulations and laws that may require changes in how Meta collects, uses, or discloses user data.
