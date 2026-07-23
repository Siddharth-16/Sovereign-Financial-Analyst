# Sovereign Financial Analyst -- Phase 1 Eval Report

Questions evaluated: **28**

## Headline numbers

- **Retrieval recall@k:** 100.0% (28/28)
- **Company routing accuracy:** 100.0%
- **Section routing accuracy:** 100.0%
- **Groundedness (faithfulness):** 89.3% of graded answers (28 graded)

## By section

| Section | N | Retrieval recall@k | Groundedness |
|---|---|---|---|
| business | 7 | 100.0% | 85.7% |
| risk_factors | 7 | 100.0% | 100.0% |
| mdna | 7 | 100.0% | 85.7% |
| financial_statements | 7 | 100.0% | 85.7% |

## Misses

- **biz-01** (business): "What are Nvidia's main business segments and how does the company describe its core products?"
  - groundedness fail: The provided context does not mention automotive products or the Drive PX 2 platform. unsupported claims: Automotive products, including Drive software and hardware platforms for autonomous vehicles; Drive PX 2 platform for autonomous driving
- **mdna-04** (mdna): "What does Broadcom's MD&A discuss regarding gross margin trends?"
  - groundedness fail: The provided context does not support the claim that the gross margin trend for Broadcom's semiconductor solutions segment has been decreasing. unsupported claims: The gross margin trend for Broadcom's semiconductor solutions segment has been decreasing; from 79% in fiscal year 2023 to 58% in fiscal year 2025; the decrease is attributed to strong demand for networking products, primarily AI networking products, and custom AI accelerators, which have lower gross margins compared to other products
- **fin-02** (financial_statements): "What does Walmart's balance sheet show regarding total liabilities?"
  - groundedness fail: The answer states a specific figure for total liabilities, but CONTEXT only shows 'Accrued liabilities' as part of total liabilities and does not provide the total amount. unsupported claims: $260,823 million
