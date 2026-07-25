# Sovereign Financial Analyst -- Phase 1 Eval Report

Questions evaluated: **28**

## Headline numbers

- **Retrieval recall@k:** 92.9% (26/28)
- **Company routing accuracy:** 92.9%
- **Section routing accuracy:** 100.0%
- **Groundedness (faithfulness):** 82.1% of graded answers (28 graded)

## By section

| Section | N | Retrieval recall@k | Groundedness |
|---|---|---|---|
| business | 7 | 85.7% | 85.7% |
| risk_factors | 7 | 100.0% | 100.0% |
| mdna | 7 | 85.7% | 71.4% |
| financial_statements | 7 | 100.0% | 71.4% |

## Misses

- **biz-01** (business): "What are Nvidia's main business segments and how does the company describe its core products?"
  - groundedness fail: The provided context does not mention 'Professional Visualization' as a business segment, but rather mentions it alongside 'Data Center' and 'Gaming'. Additionally, the descriptions provided for each segment do not accurately reflect the language used in the context to describe those segments. unsupported claims: Nvidia's main business segments are: * Datacenter, Professional Visualization, Gaming; The company describes its core products as follows: * Data Center: The company will have a broader and faster Data Center product launch cadence to meet a growing and diverse set of AI opportunities.; Professional Visualization: Product transitions are complex and can impact revenue as the company often ships both new and prior architecture products simultaneously.; Gaming: Deployment of new products to customers creates additional challenges due to the complexity of its technologies, which has impacted and may in the future impact the timing of customer purchases or otherwise impact demand.
- **biz-02** (business): "What is Tesla's business model and what are its primary product lines?"
  - retrieval miss: routed to (company=tesla, inc., section=business), expected (company=tesla, section=business). no citations returned by the agent's tool call(s)
- **mdna-01** (mdna): "What does Microsoft's MD&A say about revenue growth drivers?"
  - groundedness fail: The MD&A Analysis answer includes claims about Office Consumer products and cloud services revenue increase and Microsoft 365 Consumer subscriber growth, but these are not supported by the provided CONTEXT. unsupported claims: Office Consumer products and cloud services revenue increased; Microsoft 365 Consumer subscribers grew
- **mdna-02** (mdna): "What does Amazon's MD&A discuss about operating margin trends?"
  - groundedness fail: The MD&A discusses operating income for different segments, but does not explicitly discuss 'operating margin trends'. Additionally, the increase in AWS operating income is mentioned for 2022, not 2024. unsupported claims: operating margin trends; AWS operating income also increased in 2024
- **mdna-04** (mdna): "What does Broadcom's MD&A discuss regarding gross margin trends?"
  - retrieval miss: routed to (company=broadcom inc., section=mdna), expected (company=broadcom, section=mdna). no citations returned by the agent's tool call(s)
- **fin-01** (financial_statements): "What do Johnson & Johnson's financial statements show about total assets?"
  - groundedness fail: The provided context does not contain any information about total assets or the specific asset categories mentioned in the answer. unsupported claims: Cash and cash equivalents: $14.1 billion; Accounts receivable: $13.5 billion; Inventory: $6.3 billion; Property, plant, and equipment: $24.9 billion; Intangible assets: $34.4 billion; Goodwill: $23.8 billion; Other assets: $7.2 billion
- **fin-06** (financial_statements): "What does Pfizer's income statement show regarding net income?"
  - groundedness fail: The answer states that net income for 2023 was $9,562 million and decreased from $92,631 million in 2022, but the context only shows net income before allocation to noncontrolling interests as $2,158 million for 2023. unsupported claims: Net income for the year ended December 31, 2023 was $9,562 million.; This represents a decrease from $92,631 million in 2022.
