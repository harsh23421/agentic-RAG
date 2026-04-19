"""
evaluate.py — Automated evaluation framework.
 
Metrics:
  1. Routing Accuracy   — did the router classify the query correctly?
  2. Retrieval Accuracy — did the right source documents get retrieved?
  3. Answer Quality     — ROUGE-L score vs expected answer (for F/S queries)
                          Cosine similarity score as a second metric.
 
Run with:
  python evaluate.py
 
Output:
  - Printed results table
  - results/evaluation_results.csv
"""
 
import os
import json
import pandas as pd
import numpy as np
from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
 
from ingestion import get_collection
from router import route_query, FACTUAL, SYNTHESIS, OUT_OF_SCOPE
from generator import generate_answer
 
load_dotenv()
 
os.makedirs("../results", exist_ok=True)
 
# ── Embedding model for cosine similarity ─────────────────────────────────────
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
ROUGE = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
 
 
# ── Test suite (15 questions, 5 per type) ─────────────────────────────────────
# Each entry: query, expected_type, expected_sources (list of doc name substrings),
#             expected_answer_keywords (for ROUGE reference)
 
TEST_QUESTIONS = [
    # ── FACTUAL (5) ───────────────────────────────────────────────────────────
    {
        "id":              "F1",
        "query":           "What does the EU AI Act define as a high-risk AI system?",
        "expected_type":   FACTUAL,
        "expected_sources": [],   # fill after downloading docs
        "reference_answer": (
            "The EU AI Act defines high-risk AI systems as those used in critical "
            "infrastructure, education, employment, essential services, law enforcement, "
            "migration, administration of justice, and democratic processes. "
            "These systems must meet strict requirements before being placed on the market."
        ),
    },
    {
        "id":              "F2",
        "query":           "What are the transparency obligations under the EU AI Act?",
        "expected_type":   FACTUAL,
        "expected_sources": [],
        "reference_answer": (
            "The EU AI Act requires providers of AI systems to ensure users know they "
            "are interacting with AI. Transparency obligations include clear labelling "
            "of AI-generated content and disclosure requirements for emotion recognition "
            "and biometric categorization systems."
        ),
    },
    {
        "id":              "F3",
        "query":           "What is the role of national competent authorities under the EU AI Act?",
        "expected_type":   FACTUAL,
        "expected_sources": [],
        "reference_answer": (
            "National competent authorities are responsible for supervising and enforcing "
            "the EU AI Act at the member state level. They conduct conformity assessments, "
            "investigate complaints, and can impose penalties for non-compliance."
        ),
    },
    {
        "id":              "F4",
        "query":           "What penalties can be imposed for violating the EU AI Act?",
        "expected_type":   FACTUAL,
        "expected_sources": [],
        "reference_answer": (
            "Violations of the EU AI Act can result in fines of up to 35 million euros "
            "or 7% of global annual turnover for prohibited AI practices, 15 million euros "
            "or 3% for other violations, and 7.5 million euros or 1.5% for providing "
            "incorrect information."
        ),
    },
    {
        "id":              "F5",
        "query":           "What is a conformity assessment in the context of the EU AI Act?",
        "expected_type":   FACTUAL,
        "expected_sources": [],
        "reference_answer": (
            "A conformity assessment is a procedure to verify that a high-risk AI system "
            "complies with the requirements of the EU AI Act before it is placed on the market. "
            "It can be conducted by the provider itself or by a third-party notified body."
        ),
    },
 
    # ── SYNTHESIS (5) ────────────────────────────────────────────────────────
    {
        "id":              "S1",
        "query":           "Compare how different documents approach the definition of AI risk.",
        "expected_type":   SYNTHESIS,
        "expected_sources": [],
        "reference_answer": (
            "Documents vary in how they categorize AI risk. The EU AI Act uses a tiered "
            "risk-based approach with prohibited, high-risk, and minimal-risk categories. "
            "Other frameworks focus on sector-specific risks or principle-based approaches "
            "that emphasize fairness and accountability rather than strict categorization."
        ),
    },
    {
        "id":              "S2",
        "query":           "What do all the documents say about transparency and explainability in AI?",
        "expected_type":   SYNTHESIS,
        "expected_sources": [],
        "reference_answer": (
            "Across documents, transparency and explainability are consistently identified "
            "as core AI governance principles. The documents agree that users should be "
            "informed when AI is involved in decisions, but differ on the technical "
            "requirements for achieving explainability."
        ),
    },
    {
        "id":              "S3",
        "query":           "How do the documents differ in their approach to AI accountability?",
        "expected_type":   SYNTHESIS,
        "expected_sources": [],
        "reference_answer": (
            "The documents show differing accountability frameworks. Some emphasize "
            "developer and deployer liability, while others focus on audit requirements "
            "and human oversight mechanisms. There is broad agreement that accountability "
            "is essential but disagreement on enforcement mechanisms."
        ),
    },
    {
        "id":              "S4",
        "query":           "Summarize the overall approach to AI governance across all the documents.",
        "expected_type":   SYNTHESIS,
        "expected_sources": [],
        "reference_answer": (
            "Collectively, the documents reflect a global convergence toward risk-based "
            "AI governance. Common themes include transparency, accountability, human "
            "oversight, and protection of fundamental rights. Differences appear in "
            "enforcement mechanisms, scope, and the degree of prescriptiveness."
        ),
    },
    {
        "id":              "S5",
        "query":           "How do the various documents address bias and fairness in AI systems?",
        "expected_type":   SYNTHESIS,
        "expected_sources": [],
        "reference_answer": (
            "All documents acknowledge that AI systems can perpetuate or amplify biases. "
            "They recommend testing for bias, ensuring representative training data, and "
            "regular auditing. Some documents provide specific technical requirements "
            "while others offer high-level principles."
        ),
    },
 
    # ── OUT OF SCOPE (5) ──────────────────────────────────────────────────────
    {
        "id":              "O1",
        "query":           "What is the best programming language to learn in 2024?",
        "expected_type":   OUT_OF_SCOPE,
        "expected_sources": [],
        "reference_answer": "NOT_IN_DOCS",
    },
    {
        "id":              "O2",
        "query":           "Who won the FIFA World Cup in 2022?",
        "expected_type":   OUT_OF_SCOPE,
        "expected_sources": [],
        "reference_answer": "NOT_IN_DOCS",
    },
    {
        "id":              "O3",
        "query":           "What is the current stock price of NVIDIA?",
        "expected_type":   OUT_OF_SCOPE,
        "expected_sources": [],
        "reference_answer": "NOT_IN_DOCS",
    },
    {
        "id":              "O4",
        "query":           "Can you give me a recipe for chocolate cake?",
        "expected_type":   OUT_OF_SCOPE,
        "expected_sources": [],
        "reference_answer": "NOT_IN_DOCS",
    },
    {
        "id":              "O5",
        "query":           "How will AI take over the world and destroy humanity?",
        "expected_type":   OUT_OF_SCOPE,
        "expected_sources": [],
        "reference_answer": "NOT_IN_DOCS",
    },
]
 
 
# ── Metric helpers ────────────────────────────────────────────────────────────
 
def rouge_l(hypothesis: str, reference: str) -> float:
    """ROUGE-L F1 score between hypothesis and reference."""
    if reference == "NOT_IN_DOCS":
        return None
    scores = ROUGE.score(reference, hypothesis)
    return round(scores["rougeL"].fmeasure, 4)
 
 
def cosine_sim(hypothesis: str, reference: str) -> float:
    """Cosine similarity between sentence embeddings."""
    if reference == "NOT_IN_DOCS":
        return None
    embs = EMBED_MODEL.encode([hypothesis, reference])
    sim = cosine_similarity([embs[0]], [embs[1]])[0][0]
    return round(float(sim), 4)
 
 
def check_routing(actual: str, expected: str) -> bool:
    return actual == expected
 
 
def check_retrieval(retrieved_sources: list[str], expected_sources: list[str]) -> bool:
    """
    Check if at least one expected source was retrieved.
    Returns True if expected_sources is empty (not specified for this question).
    """
    if not expected_sources:
        return True   # not evaluated for this question
    retrieved_lower = [s.lower() for s in retrieved_sources]
    for exp in expected_sources:
        if any(exp.lower() in r for r in retrieved_lower):
            return True
    return False
 
 
def oos_hallucination_check(answer: str) -> bool:
    """
    For OUT_OF_SCOPE queries, check the answer does NOT make factual claims.
    Returns True if answer correctly declines (no hallucination detected).
    Heuristic: answer should contain decline language and be relatively short.
    """
    decline_phrases = [
        "not available", "not found", "not in", "cannot find",
        "no information", "don't have", "do not have", "outside",
        "not covered", "not present", "unable to find", "not contained",
        "not addressed", "beyond the scope",
    ]
    answer_lower = answer.lower()
    has_decline = any(p in answer_lower for p in decline_phrases)
    # Also check it's not suspiciously long (long answers may be hallucinating)
    reasonable_length = len(answer.split()) < 200
    return has_decline and reasonable_length
 
 
# ── Main evaluation loop ──────────────────────────────────────────────────────
 
def run_evaluation():
    print("Loading ChromaDB collection...")
    collection = get_collection()
 
    if collection.count() == 0:
        raise RuntimeError(
            "ChromaDB collection is empty. "
            "Run: python ingestion.py ./data  first."
        )
 
    print(f"Collection has {collection.count()} chunks.\n")
    print("Running evaluation on 15 test questions...\n")
 
    rows = []
 
    for item in TEST_QUESTIONS:
        qid    = item["id"]
        query  = item["query"]
        e_type = item["expected_type"]
        e_srcs = item["expected_sources"]
        ref    = item["reference_answer"]
 
        print(f"[{qid}] {query[:70]}...")
 
        # Route
        decision = route_query(query, collection)
        # Generate
        result   = generate_answer(query, decision)
 
        answer       = result["answer"]
        actual_type  = result["query_type"]
        actual_srcs  = result["sources_used"]
        confidence   = result["routing_confidence"]
 
        # Metrics
        routing_correct   = check_routing(actual_type, e_type)
        retrieval_correct = check_retrieval(actual_srcs, e_srcs)
        rl_score          = rouge_l(answer, ref)
        cos_score         = cosine_sim(answer, ref)
 
        # Hallucination check for OOS
        no_hallucination = None
        if actual_type == OUT_OF_SCOPE or e_type == OUT_OF_SCOPE:
            no_hallucination = oos_hallucination_check(answer)
 
        rows.append({
            "id":                 qid,
            "query":              query,
            "expected_type":      e_type,
            "actual_type":        actual_type,
            "routing_correct":    routing_correct,
            "retrieval_correct":  retrieval_correct,
            "rouge_l":            rl_score,
            "cosine_sim":         cos_score,
            "no_hallucination":   no_hallucination,
            "confidence":         round(confidence, 4),
            "sources_used":       ", ".join(actual_srcs) if actual_srcs else "—",
            "answer_preview":     answer[:120].replace("\n", " "),
            "routing_reasoning":  result["routing_reasoning"],
        })
 
        status = "✓" if routing_correct else "✗"
        print(f"  Routing: {status} ({e_type} → got {actual_type})")
        if rl_score is not None:
            print(f"  ROUGE-L: {rl_score:.4f}  CosSim: {cos_score:.4f}")
        if no_hallucination is not None:
            h_status = "✓ no hallucination" if no_hallucination else "✗ possible hallucination"
            print(f"  OOS check: {h_status}")
        print()
 
    # ── Results table ─────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    csv_path = "../results/evaluation_results.csv"
    df.to_csv(csv_path, index=False)
 
    # ── Summary stats ─────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("EVALUATION SUMMARY")
    print("="*65)
 
    routing_acc = df["routing_correct"].mean()
    print(f"Overall Routing Accuracy : {routing_acc:.1%}")
 
    for qtype in [FACTUAL, SYNTHESIS, OUT_OF_SCOPE]:
        sub = df[df["expected_type"] == qtype]
        acc = sub["routing_correct"].mean()
        print(f"  {qtype:<14} Routing Acc: {acc:.1%}")
 
    factual_synth = df[df["expected_type"].isin([FACTUAL, SYNTHESIS])]
    avg_rouge = factual_synth["rouge_l"].dropna().mean()
    avg_cos   = factual_synth["cosine_sim"].dropna().mean()
    print(f"\nAvg ROUGE-L (F+S queries): {avg_rouge:.4f}")
    print(f"Avg CosSim  (F+S queries): {avg_cos:.4f}")
 
    oos_rows = df[df["expected_type"] == OUT_OF_SCOPE]
    oos_ok = oos_rows["no_hallucination"].sum()
    print(f"\nOOS Hallucination Prevention: {int(oos_ok)}/5 passed")
 
    print(f"\nFull results saved to: {csv_path}")
    print("="*65)
 
    # Print pretty table
    display_cols = [
        "id", "expected_type", "actual_type",
        "routing_correct", "rouge_l", "cosine_sim",
        "no_hallucination", "confidence",
    ]
    print("\nDetailed Results:")
    print(df[display_cols].to_string(index=False))
 
    return df
 
 
if __name__ == "__main__":
    run_evaluation()