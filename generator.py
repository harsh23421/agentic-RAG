"""
generator.py — Grounded answer generation using Claude (Anthropic API).
 
Three answer modes:
  FACTUAL      → concise, grounded answer from top chunks, with source citation
  SYNTHESIS    → structured answer combining multiple chunks/sources
  OUT_OF_SCOPE → polite decline with no hallucination
"""
 
import os
import anthropic
from dotenv import load_dotenv
from router import RoutingDecision, FACTUAL, SYNTHESIS, OUT_OF_SCOPE
 
load_dotenv()
 
MODEL = "claude-sonnet-4-20250514"
 
# ── Prompt templates ──────────────────────────────────────────────────────────
 
FACTUAL_SYSTEM = """You are a precise AI regulation research assistant.
Your job is to answer questions ONLY using the provided document excerpts.
 
Rules:
- Answer directly and concisely (2-4 sentences unless more detail is needed).
- Cite your source document(s) inline, e.g.: [Source: eu_ai_act].
- If the excerpts do not fully support the answer, say so explicitly.
- Do NOT add information beyond what the excerpts contain.
- Do NOT speculate or extrapolate."""
 
SYNTHESIS_SYSTEM = """You are an AI regulation research analyst.
Your job is to synthesize information from multiple document excerpts.
 
Rules:
- Identify agreements, disagreements, and complementary perspectives across sources.
- Structure your answer clearly (e.g., use "Document A says... Document B adds...").
- Cite each source inline, e.g.: [Source: oecd_principles].
- Acknowledge contradictions or gaps honestly.
- Do NOT add information beyond what the excerpts contain."""
 
OUT_OF_SCOPE_SYSTEM = """You are an AI regulation research assistant.
You only answer questions about AI regulation based on a specific document set."""
 
 
def _format_chunks(chunks: list[dict]) -> str:
    """Format retrieved chunks into a numbered context block."""
    parts = []
    for i, ch in enumerate(chunks, 1):
        parts.append(
            f"[Excerpt {i} | Source: {ch['source']} | Relevance: {ch['score']:.3f}]\n"
            f"{ch['text']}"
        )
    return "\n\n---\n\n".join(parts)
 
 
def _call_claude(system: str, user: str) -> str:
    """Single Claude API call, returns response text."""
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    message = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return message.content[0].text.strip()
 
 
# ── Answer generators ─────────────────────────────────────────────────────────
 
def generate_factual(query: str, chunks: list[dict]) -> str:
    context = _format_chunks(chunks)
    user_prompt = f"""Document excerpts:
{context}
 
Question: {query}
 
Answer (grounded in the excerpts above only):"""
    return _call_claude(FACTUAL_SYSTEM, user_prompt)
 
 
def generate_synthesis(query: str, chunks: list[dict]) -> str:
    context = _format_chunks(chunks)
    user_prompt = f"""Document excerpts from multiple sources:
{context}
 
Question requiring synthesis: {query}
 
Synthesized answer (comparing and combining the sources above):"""
    return _call_claude(SYNTHESIS_SYSTEM, user_prompt)
 
 
def generate_out_of_scope(query: str, reasoning: str) -> str:
    user_prompt = f"""The user asked: "{query}"
 
The knowledge base search returned: {reasoning}
 
Write a brief, honest response explaining that the information is not available
in the provided documents. Suggest what kind of source might have this information.
Do NOT attempt to answer the question."""
    return _call_claude(OUT_OF_SCOPE_SYSTEM, user_prompt)
 
 
# ── Main dispatcher ───────────────────────────────────────────────────────────
 
def generate_answer(query: str, decision: RoutingDecision) -> dict:
    """
    Generate a grounded answer based on routing decision.
 
    Returns dict with keys:
      query, query_type, answer, sources_used, routing_confidence, routing_reasoning
    """
    if decision.query_type == FACTUAL:
        answer = generate_factual(query, decision.top_chunks)
        sources = list({c["source"] for c in decision.top_chunks})
 
    elif decision.query_type == SYNTHESIS:
        answer = generate_synthesis(query, decision.top_chunks)
        sources = list({c["source"] for c in decision.top_chunks})
 
    else:  # OUT_OF_SCOPE
        answer = generate_out_of_scope(query, decision.reasoning)
        sources = []
 
    return {
        "query":               query,
        "query_type":          decision.query_type,
        "answer":              answer,
        "sources_used":        sources,
        "routing_confidence":  decision.confidence,
        "routing_reasoning":   decision.reasoning,
    }
 
 
if __name__ == "__main__":
    from ingestion import get_collection
    from router import route_query
 
    col = get_collection()
 
    test_cases = [
        "What penalties does the EU AI Act impose for violations?",
        "How do the documents differ in their approach to AI transparency?",
        "What is the capital of France?",
    ]
 
    for q in test_cases:
        print(f"\n{'='*60}")
        print(f"Query: {q}")
        decision = route_query(q, col)
        result   = generate_answer(q, decision)
        print(f"Type  : {result['query_type']}")
        print(f"Answer: {result['answer']}")
        print(f"Sources: {result['sources_used']}")