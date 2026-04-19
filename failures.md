FAILURES.md — Honest Failure Analysis
This document identifies 4 cases where the system fails or underperforms, with specific root causes and concrete fixes.

Failure 1 — S4: Synthesis Query Misrouted as FACTUAL
Query

"Summarize the overall approach to AI governance across all the documents."

What Happened

Expected: SYNTHESIS
Actual: FACTUAL (confidence: 0.4601)
ROUGE-L: 0.2934 (lowest among all Factual/Synthesis queries)

The word "summarize" was not in the synthesis keyword list. The query retrieved a high-scoring chunk (0.46) from the EU AI Act that broadly discussed AI governance, pushing it above the FACTUAL threshold. The resulting answer only cited one document instead of synthesizing all four.
Root Cause
The synthesis keyword list missed summarize, overall, and across all as standalone triggers. The router's Stage 1 check returned no synthesis signal, so Stage 2's confidence score alone decided the route — and 0.46 crossed the FACTUAL threshold.
Fix

Add summarize, overall approach, across all, in general, collectively to the synthesis pattern list.
Lower the FACTUAL threshold slightly to 0.48 so borderline scores default to SYNTHESIS rather than FACTUAL.
Add a post-routing check: if the query contains "all documents" or "across", override to SYNTHESIS regardless of score.


Failure 2 — O4: Out-of-Scope Query Misrouted as SYNTHESIS
Query

"Give me a recipe for chocolate cake?"

What Happened

Expected: OUT_OF_SCOPE
Actual: SYNTHESIS (confidence: 0.3102)
Hallucination check: ✓ Passed (answer still correctly declined)

The word "recipe" is in the OOS pattern list, but "chocolate cake" matched a low-relevance chunk about "data ingredients" in an AI model training context, pushing the retrieval score to 0.31 — just above the synthesis threshold. The system routed to SYNTHESIS rather than OUT_OF_SCOPE.
Root Cause
The retrieval confidence thresholds were calibrated on regulation-domain queries. Coincidental vocabulary overlap (e.g., "data ingredients", "model training steps") can weakly match food-related queries, producing misleadingly non-zero similarity scores.
Fix

After retrieval, add a topic coherence check: if the top chunk's source text contains no AI/regulation keywords, treat the match as noise and downgrade to OUT_OF_SCOPE.
Add a minimum query-chunk overlap score (keyword intersection) as a secondary gate — pure semantic similarity is insufficient when vocabulary accidentally overlaps across domains.
Increase the SYNTHESIS threshold from 0.30 to 0.33 to reduce false synthesis triggers.


Failure 3 — Low ROUGE-L Scores on Synthesis Queries (avg 0.332)
What Happened
All 5 synthesis queries scored noticeably lower ROUGE-L than factual queries (avg 0.332 vs 0.399). The gap is especially visible in S3 (0.3088) and S4 (0.2934).
Root Cause
ROUGE-L measures n-gram overlap between generated text and a reference answer. For synthesis queries:

The reference answers were written as clean, concise summaries.
The generated answers were longer, structured with "Document A says... Document B says..." framing.
This structural difference naturally reduces n-gram overlap even when the semantic content is correct.

This is a metric mismatch problem, not a generation quality problem. Cosine similarity scores for the same queries (avg 0.612) reflect the semantic correctness more accurately.
Fix

Use BERTScore instead of or alongside ROUGE-L for synthesis queries — it measures semantic similarity rather than surface n-gram overlap.
Write reference answers in the same multi-source format as generated answers, so the metric comparison is fair.
Add a human-evaluation column in the results table for synthesis queries, since automated metrics alone are insufficient for cross-document answers.


Failure 4 — Contradictory Documents Produce Over-Hedged Answers
What Happened
The 4 documents intentionally contain partially contradictory information (as stated in the assignment). When contradictions were retrieved together (e.g., different definitions of "high-risk AI" across the EU AI Act and NIST RMF), the synthesis prompt produced answers like:

"Document A defines high-risk AI as X, while Document B uses a different categorization Y. These approaches differ and cannot be directly reconciled."

This is technically correct but unhelpful — the user learns that a contradiction exists but gets no guidance on which to trust.
Root Cause
The SYNTHESIS prompt instructs Claude to "acknowledge contradictions honestly" but provides no resolution strategy. Without metadata about document authority (legally binding vs. voluntary guideline), recency, or jurisdiction, the model defaults to presenting both sides equally and declining to resolve.
Fix

Tag documents with metadata at ingestion time: {"type": "legislation", "jurisdiction": "EU", "year": 2024, "binding": true} vs {"type": "guideline", "jurisdiction": "international", "binding": false}.
Pass metadata to the synthesis prompt: "Prefer the more specific or legally binding source. If jurisdictions differ, note that both may be correct in their own context."
Contradiction pre-detection: Before synthesis, run a lightweight check comparing key claims across retrieved chunks. If a contradiction is detected, flag it explicitly and apply the resolution heuristic above rather than presenting both sides as equally valid.
