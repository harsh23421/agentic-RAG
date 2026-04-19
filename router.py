"""
router.py — Explicit, inspectable query router.
 
Routing is a two-stage deterministic pipeline — NOT a black-box LLM call:
 
  Stage 1 — Keyword Heuristics
    Scan the query for strong linguistic signals:
      • Synthesis signals: "compare", "difference", "across", "both", "all",
        "relationship between", "how do X and Y", "contrast", etc.
      • Out-of-scope signals: topics clearly outside AI regulation
        (weather, sports, cooking, personal advice, etc.)
 
  Stage 2 — Retrieval Confidence Check
    Retrieve the top-5 chunks and inspect their cosine similarity scores:
      • If max_score >= FACTUAL_THRESH   → FACTUAL
      • If max_score >= SYNTHESIS_THRESH → SYNTHESIS (needs merging)
      • If max_score <  SYNTHESIS_THRESH → OUT_OF_SCOPE
 
  Final decision merges both stages with explicit precedence rules.
 
Why no black-box LLM routing?
  The assignment explicitly forbids "let the LLM decide." Every routing
  decision is explainable via the returned `reasoning` string.
"""
 
import re
from dataclasses import dataclass
from ingestion import retrieve
 
 
# ── Thresholds (tunable) ──────────────────────────────────────────────────────
FACTUAL_THRESH   = 0.45   # single chunk clearly answers the query
SYNTHESIS_THRESH = 0.30   # some relevant content exists but needs merging
# below SYNTHESIS_THRESH → OUT_OF_SCOPE
 
 
# ── Query type enum ───────────────────────────────────────────────────────────
FACTUAL      = "FACTUAL"
SYNTHESIS    = "SYNTHESIS"
OUT_OF_SCOPE = "OUT_OF_SCOPE"
 
 
# ── Keyword signal tables ─────────────────────────────────────────────────────
SYNTHESIS_PATTERNS = [
    r"\bcompare\b", r"\bcomparison\b", r"\bcontrast\b",
    r"\bdifference[s]?\b", r"\bsimilarit(y|ies)\b",
    r"\bacross\b.{0,30}\b(document|source|regulation|countri|approach)",
    r"\bboth\b.{0,40}\band\b",
    r"\ball (the |these )?(document|source|regulation|approach|framework)",
    r"\brelationship between\b",
    r"\bhow do .{0,40} (differ|compare|relate)",
    r"\bwhat are the (different|various|main) approach",
    r"\boverall\b", r"\bin general\b", r"\bbroadly\b",
    r"\bsummariz(e|ing)\b.{0,20}(all|multiple|each)",
    r"\bwhat do .{0,20} (say|mention|address) about\b",
    r"\bcombine\b", r"\bsynthesi[sz]",
    r"\bmultiple\b.{0,20}\b(source|document|regulation)",
]
 
OUT_OF_SCOPE_PATTERNS = [
    # clearly non-AI-regulation topics
    r"\bweather\b", r"\brecipe\b", r"\bcooking\b", r"\bsport[s]?\b",
    r"\bfootball\b", r"\bcricket\b", r"\bbaseball\b",
    r"\bstock price\b", r"\bstock market\b", r"\binvest(ment|ing)\b",
    r"\bpolitics\b(?!.*\bai\b)", r"\belection\b",
    r"\bhealth\b(?!.*\bai\b)", r"\bmedical\b(?!.*\bai\b)",
    r"\brelationship advice\b", r"\bpersonal\b.{0,20}\badvice\b",
    r"\bfilm\b", r"\bmovie\b", r"\bsong\b", r"\bmusic\b",
    r"\btravel\b", r"\bhotel\b", r"\bflight\b",
    # future/speculative that docs can't answer
    r"\bwill (ai|artificial intelligence) (take over|destroy|rule)\b",
    r"\bpredict.{0,20}future\b(?!.*\bregulat)",
]
 
# Topics that the documents DO cover (used to rescue borderline queries)
IN_SCOPE_ANCHORS = [
    r"\bregulat", r"\bai act\b", r"\bartificial intelligence\b", r"\bai\b",
    r"\beuropean union\b", r"\beu\b", r"\brisk\b.{0,20}\bai\b",
    r"\bgovernance\b", r"\bcompliance\b", r"\btransparency\b",
    r"\baccountabilit", r"\bliabilit", r"\bbias\b", r"\bfairness\b",
    r"\bdata protection\b", r"\bgdpr\b", r"\bprivacy\b",
    r"\bhigh.risk\b", r"\bfoundation model\b", r"\bllm\b",
    r"\bdeep fake\b", r"\bdeepfake\b", r"\bbiometric\b",
    r"\benforcemen", r"\bpenalt", r"\bfine[s]?\b",
]
 
 
@dataclass
class RoutingDecision:
    query_type: str          # FACTUAL | SYNTHESIS | OUT_OF_SCOPE
    confidence: float        # 0–1, based on top retrieval score
    top_chunks: list[dict]   # retrieved chunks (empty for OOS)
    reasoning: str           # human-readable explanation
 
 
# ── Helpers ───────────────────────────────────────────────────────────────────
def _match_any(text: str, patterns: list[str]) -> str | None:
    """Return first matching pattern label, or None."""
    text_lower = text.lower()
    for p in patterns:
        if re.search(p, text_lower):
            return p
    return None
 
 
def _has_in_scope_anchor(query: str) -> bool:
    return _match_any(query, IN_SCOPE_ANCHORS) is not None
 
 
# ── Main router ───────────────────────────────────────────────────────────────
def route_query(query: str, collection) -> RoutingDecision:
    """
    Classify a query and retrieve relevant chunks.
 
    Returns a RoutingDecision with full reasoning trace.
    """
    q = query.strip()
 
    # ── Stage 1: Keyword heuristics ──────────────────────────────────────────
    oos_match = _match_any(q, OUT_OF_SCOPE_PATTERNS)
    syn_match = _match_any(q, SYNTHESIS_PATTERNS)
 
    # If OOS keyword found AND no strong in-scope anchor → early exit
    if oos_match and not _has_in_scope_anchor(q):
        return RoutingDecision(
            query_type  = OUT_OF_SCOPE,
            confidence  = 0.0,
            top_chunks  = [],
            reasoning   = (
                f"Keyword heuristic matched out-of-scope pattern '{oos_match}' "
                f"and no AI-regulation anchor found in query. Skipping retrieval."
            ),
        )
 
    # ── Stage 2: Retrieve and check confidence ────────────────────────────────
    top_k    = 6 if syn_match else 5
    chunks   = retrieve(q, collection, top_k=top_k)
    max_score = chunks[0]["score"] if chunks else 0.0
    sources   = list({c["source"] for c in chunks})
 
    # ── Stage 3: Decision logic ───────────────────────────────────────────────
 
    # Explicit synthesis keyword + decent retrieval → SYNTHESIS
    if syn_match and max_score >= SYNTHESIS_THRESH:
        return RoutingDecision(
            query_type  = SYNTHESIS,
            confidence  = max_score,
            top_chunks  = chunks,
            reasoning   = (
                f"Synthesis keyword pattern '{syn_match}' detected. "
                f"Top retrieval score {max_score:.3f} >= threshold {SYNTHESIS_THRESH}. "
                f"Retrieved chunks span {len(sources)} source(s): {sources}."
            ),
        )
 
    # High retrieval confidence → FACTUAL
    if max_score >= FACTUAL_THRESH:
        return RoutingDecision(
            query_type  = FACTUAL,
            confidence  = max_score,
            top_chunks  = chunks[:3],   # fewer chunks for focused answer
            reasoning   = (
                f"Top retrieval score {max_score:.3f} >= factual threshold "
                f"{FACTUAL_THRESH}. Single-source answer likely. "
                f"Best chunk from: '{chunks[0]['source']}'."
            ),
        )
 
    # Medium confidence + synthesis signal → SYNTHESIS (weak evidence)
    if syn_match and max_score >= 0.20:
        return RoutingDecision(
            query_type  = SYNTHESIS,
            confidence  = max_score,
            top_chunks  = chunks,
            reasoning   = (
                f"Synthesis keyword '{syn_match}' with moderate score "
                f"{max_score:.3f}. Attempting cross-chunk synthesis."
            ),
        )
 
    # Low confidence → OUT_OF_SCOPE
    return RoutingDecision(
        query_type  = OUT_OF_SCOPE,
        confidence  = max_score,
        top_chunks  = chunks,
        reasoning   = (
            f"Max retrieval score {max_score:.3f} below synthesis threshold "
            f"{SYNTHESIS_THRESH}. No reliable evidence found in knowledge base."
        ),
    )
 
 
if __name__ == "__main__":
    # Quick smoke test
    from ingestion import get_collection
    col = get_collection()
    test_queries = [
        "What is the EU AI Act?",
        "Compare how different documents define high-risk AI systems.",
        "What is the best recipe for pasta carbonara?",
    ]
    for q in test_queries:
        d = route_query(q, col)
        print(f"\nQuery : {q}")
        print(f"Type  : {d.query_type}  (confidence={d.confidence:.3f})")
        print(f"Reason: {d.reasoning}")