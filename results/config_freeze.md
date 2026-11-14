
# LAB7 — Configuration Freeze

Date: 2026-11-14
Dataset: OpenStax Calculus Volume 1, PDF pages 51–100
Chunk strategy: Strategy B (162 chunks)
Embedding: BAAI/bge-m3, 1024 dimensions
Vector store: PostgreSQL + pgvector

## DEV Retrieval Comparison

| Method | Hit@1 | Evidence Hit@3 |
|---|---:|---:|
| Dense | 12/15 | 14/15 |
| BM25 | 8/15 | 14/15 |
| Hybrid RRF | 11/15 | 15/15 |

Results use DEV evidence review and manual corrections.
Retrieval latency from the original benchmark is not
directly comparable because query embedding caching
affects the measured times.

## Context Experiments

C1: Full retrieved chunks, deduplicated, 3000-char budget.

C2-v1: Select up to 3 keyword-overlap sentences per chunk.
Finding: Q008 lost important evidence and abstained incorrectly.

C2-v2: Select the most relevant sentence and retain one
preceding sentence and two following sentences.
Finding: Q008 was corrected; Q002 still had an incomplete answer.

C1 Answer LLM tokens: 11222
C1 Answer API cost: USD 0.002265

C2-v2 Answer LLM tokens: 9935
C2-v2 Answer API cost: USD 0.002062

Costs above exclude the planner LLM and local retrieval.
These are DEV results only.

## DEV Error Analysis

Q008: Evidence lost during C2-v1 context selection.
C2-v2 restored the evidence and corrected the answer.

Q002: C2-v2 answer omitted the closing time, although
context contained opening time and total operating hours.

Q007: Citation did not support every example in the answer.
The evidence was split across adjacent chunks.

Q011: The answer included numerical periods not explicitly
recoverable from the stored context due to PDF extraction.

## Frozen Configuration

Retrieval: Hybrid RRF (RRF k=60)
Retrieved top-k: 3
Chunk strategy: B
Context selection: C1 baseline
Context budget: 3000 characters
Maximum retrieval calls: 1
LLM: OpenRouter, model fixed by OPENROUTER_MODEL

Selection criteria:
1. Evidence coverage and answer grounding
2. Correct abstention
3. Citation support
4. Token usage, API cost, and latency

Decision:
Prefer Hybrid + C1 for evidence preservation and grounding.
C2-v2 offers lower costs but has an answer-completeness issue.

No configurations will be changed based on TEST results.
TEST is reserved for final evaluation.


