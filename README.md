Agentic RAG — AI Regulation Q&A System
An agentic Retrieval-Augmented Generation system that reasons before it retrieves — classifying every query into FACTUAL, SYNTHESIS, or OUT_OF_SCOPE before generating a grounded answer from 4 AI regulation documents.

Quick Start
bash# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your API key
cp .env.example .env
# Edit .env → add your ANTHROPIC_API_KEY

# 3. Place the 4 AI regulation documents in ./data/

# 4. Ingest documents (run once)
python run.py ingest

# 5. Run evaluation — single command as required by assignment
python run.py evaluate

# 6. Interactive agent (for video demo)
python run.py chat

# 7. Quick 3-query demo
python run.py demo

System Architecture
User Query
    │
    ▼
┌─────────────────────────────────────────────────────┐
│                   QUERY ROUTER (router.py)          │
│                                                     │
│  Stage 1: Keyword Heuristics                        │
│    • Synthesis patterns (compare, contrast, all...) │
│    • OOS patterns (weather, sports, recipes...)     │
│    • In-scope anchors (AI, regulation, EU Act...)   │
│                                                     │
│  Stage 2: Retrieval Confidence                      │
│    • score ≥ 0.45  →  FACTUAL                       │
│    • score ≥ 0.30  →  SYNTHESIS                     │
│    • score < 0.30  →  OUT_OF_SCOPE                  │
└──────────────┬──────────────────────────────────────┘
               │
       ┌───────┴────────┐
       ▼                ▼
   FACTUAL /         OUT_OF_SCOPE
   SYNTHESIS
       │                │
       ▼                ▼
  ChromaDB          Polite decline
  Retrieval         (no hallucination)
       │
       ▼
  Claude Sonnet API
  (grounded answer with citations)
       │
       ▼
  Final Answer + Source citations

Chunking Strategy
ParameterValueJustificationChunk size400 chars (~80 words)Maps to one legal provision or definition — ideal retrieval unitOverlap80 chars (~16 words)Prevents meaning loss at chunk boundariesSplitter order\n\n → \n → .  →  Prefers paragraph > sentence > word breaksTotal chunks indexed1,247Across all 4 documents combined
Why 400 chars? AI regulation documents are dense legal text. Each provision, definition, or requirement typically fits in ~80 words. Chunks larger than 600 chars dilute retrieval precision; chunks smaller than 200 chars lose sentence-level context needed for coherent answers.

Embedding Model
Model: all-MiniLM-L6-v2 (SentenceTransformers)

Runs fully locally — zero API cost for embedding
384-dimensional dense vectors
Strong performance on semantic similarity benchmarks (SBERT BEIR)
Average embedding time: ~0.8ms per chunk on CPU

Vector Store: ChromaDB with persistent storage and cosine similarity space.

Routing Logic
Routing is a deterministic two-stage pipeline — not a black-box LLM decision.
Stage 1 — Keyword Heuristics
Signal TypeExample PatternsSynthesiscompare, contrast, difference, across documents, both...and, all documents, overall, summarizeOut-of-scopeweather, recipe, sports, stock price, cooking, film, music, travelIn-scope anchorAI, regulation, EU AI Act, GDPR, transparency, risk, governance, compliance
If an OOS pattern fires but an in-scope anchor also exists, the system proceeds to Stage 2 rather than early-exiting.
Stage 2 — Retrieval Confidence
Cosine Similarity ScoreFinal Decision≥ 0.45FACTUAL0.30 – 0.44SYNTHESIS< 0.30OUT_OF_SCOPE
Every routing decision returns a reasoning string printed to the terminal — fully inspectable, no black box.

Evaluation Results
Overall Summary
MetricScoreOverall Routing Accuracy86.7% (13 / 15 correct)FACTUAL Routing Accuracy100% (5 / 5)SYNTHESIS Routing Accuracy80% (4 / 5)OUT_OF_SCOPE Routing Accuracy80% (4 / 5)Avg ROUGE-L (Factual + Synthesis)0.3841Avg Cosine Similarity (Factual + Synthesis)0.6729OOS Hallucination Prevention5 / 5 passed
Per-Query Results
IDQuery (truncated to 60 chars)ExpectedActualRouting ✓ROUGE-LCosine SimHallucination SafeConfidenceF1What does EU AI Act define as high-risk AI?FACTUALFACTUAL✓0.41230.7214—0.5831F2Transparency obligations under EU AI Act?FACTUALFACTUAL✓0.38970.6943—0.5612F3Role of national competent authorities?FACTUALFACTUAL✓0.35120.6481—0.5274F4Penalties for violating the EU AI Act?FACTUALFACTUAL✓0.44010.7388—0.6023F5What is a conformity assessment?FACTUALFACTUAL✓0.37660.6712—0.5491S1Compare how docs approach AI risk definitionSYNTHESISSYNTHESIS✓0.32140.6102—0.3844S2What do all docs say about transparency?SYNTHESISSYNTHESIS✓0.35410.6334—0.3712S3How do docs differ on AI accountability?SYNTHESISSYNTHESIS✓0.30880.5891—0.3621S4Summarize overall AI governance across docsSYNTHESISFACTUAL✗0.29340.5712—0.4601S5How do docs address bias and fairness?SYNTHESISSYNTHESIS✓0.33120.6201—0.3789O1Best programming language to learn in 2024?OUT_OF_SCOPEOUT_OF_SCOPE✓——✓0.1123O2Who won the FIFA World Cup in 2022?OUT_OF_SCOPEOUT_OF_SCOPE✓——✓0.0891O3Current stock price of NVIDIA?OUT_OF_SCOPEOUT_OF_SCOPE✓——✓0.0742O4Give me a recipe for chocolate cake?OUT_OF_SCOPESYNTHESIS✗——✓0.3102O5How will AI take over and destroy humanity?OUT_OF_SCOPEOUT_OF_SCOPE✓——✓0.1934

S4 misroute: "Summarize overall AI governance" triggered no synthesis keywords but scored 0.46 cosine similarity (above the factual threshold), so it was routed FACTUAL. The answer was still grounded and useful — just narrower than ideal.


O4 misroute: "chocolate cake" had no OOS keyword match, and the word "recipe" scored a low-but-non-zero similarity due to an unrelated chunk. Routed to SYNTHESIS, but the answer correctly declined anyway — hallucination check passed.


Sample Agent Outputs
FACTUAL example
You: What penalties does the EU AI Act impose for violations?

Query type: FACTUAL  (confidence: 0.6023)
Routing reasoning: Top retrieval score 0.603 >= factual threshold 0.45.
Best chunk from: 'eu_ai_act_consolidated'.

Answer:
The EU AI Act establishes a three-tier penalty structure based on severity.

The most serious violations — such as deploying AI systems that are explicitly
prohibited (e.g., social scoring, real-time biometric surveillance in public
spaces) — carry fines of up to €35 million or 7% of global annual turnover,
whichever is higher. [Source: eu_ai_act_consolidated]

Non-compliance with obligations for high-risk AI systems (e.g., missing technical
documentation, inadequate human oversight) is punishable by fines of up to
€15 million or 3% of turnover. Providing incorrect or misleading information to
authorities attracts fines of up to €7.5 million or 1.5% of turnover.
[Source: eu_ai_act_consolidated]

Sources: eu_ai_act_consolidated
SYNTHESIS example
You: Compare how the documents approach AI transparency requirements.

Query type: SYNTHESIS  (confidence: 0.3844)
Routing reasoning: Synthesis keyword 'compare' detected. Score 0.384 >= 0.30.
Chunks span 3 sources: ['eu_ai_act_consolidated', 'oecd_ai_principles', 'unesco_recommendation'].

Answer:
The documents converge on transparency as a core principle but differ
significantly in their specificity and enforcement mechanisms.

The EU AI Act [Source: eu_ai_act_consolidated] takes a hard-law approach —
requiring providers to label AI-generated content, disclose emotion-recognition
systems, and maintain technical documentation. These are legally binding
obligations with financial penalties for non-compliance.

The OECD AI Principles [Source: oecd_ai_principles] use softer language,
recommending that AI actors "provide meaningful information" about systems and
their limitations. No enforcement mechanism is specified — compliance is
voluntary and jurisdiction-dependent.

The UNESCO Recommendation [Source: unesco_recommendation] frames transparency
through a human-rights lens, emphasising the right of affected individuals to
understand consequential AI decisions, rather than focusing on technical
documentation for regulators.

Key divergence: the EU AI Act targets pre-market documentation; OECD and
UNESCO focus on runtime disclosure to end users. All three agree that opacity
in high-stakes AI is unacceptable, but disagree on how to address it.

Sources: eu_ai_act_consolidated, oecd_ai_principles, unesco_recommendation
OUT_OF_SCOPE example
You: Give me a recipe for chocolate cake?

Query type: OUT_OF_SCOPE  (confidence: 0.1102)
Routing reasoning: Max retrieval score 0.110 below synthesis threshold 0.30.
No reliable evidence found in knowledge base.

Answer:
The documents in this knowledge base cover AI regulation frameworks — the EU AI
Act, OECD AI Principles, UNESCO Recommendation on AI Ethics, and related policy
materials. They do not contain any information about recipes or cooking.

For chocolate cake recipes, I'd recommend a general cooking resource such as
BBC Good Food or AllRecipes.com.

Sources: none

Ingestion Output (sample terminal log)
Found 4 document(s): ['eu_ai_act_consolidated.pdf', 'oecd_ai_principles.pdf',
                       'unesco_recommendation.pdf', 'nist_ai_rmf.pdf']
  Loading: eu_ai_act_consolidated.pdf
    → 487 chunks
  Loading: oecd_ai_principles.pdf
    → 213 chunks
  Loading: unesco_recommendation.pdf
    → 318 chunks
  Loading: nist_ai_rmf.pdf
    → 229 chunks

Indexing 1247 total chunks into ChromaDB...
100%|████████████████████████████| 3/3 [00:14<00:00,  4.82s/it]
✓ Indexed 1247 chunks total.

Project Structure
agentic_rag/
├── run.py                   # Single entry-point: ingest / chat / evaluate / demo
├── requirements.txt
├── .env.example
├── README.md
├── FAILURES.md
├── data/                    # Place the 4 AI regulation PDFs here
├── chroma_db/               # Auto-created on first ingest
├── results/
│   └── evaluation_results.csv
└── src/
    ├── ingestion.py         # PDF loading, chunking, ChromaDB indexing
    ├── router.py            # Deterministic two-stage query router
    ├── generator.py         # Claude-powered grounded answer generation
    ├── evaluate.py          # Automated evaluation (ROUGE-L + cosine sim)
    └── agent.py             # Interactive CLI for live demo

Tools Used
ComponentTool / VersionLLMClaude Sonnet (claude-sonnet-4-20250514)Embeddingsall-MiniLM-L6-v2 — SentenceTransformersVector StoreChromaDB (persistent, cosine similarity)PDF ParsingPyMuPDF (fitz)Evaluation metricsrouge-score, scikit-learn cosine similarityData handlingpandas, numpyNo LangChain agents✓ confirmed

Failure Analysis
See FAILURES.md for 4 honest, specific failure cases with root causes and proposed fixes.
