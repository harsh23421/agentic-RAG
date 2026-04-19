#!/usr/bin/env python3
"""
run.py — Single entry-point for the entire pipeline.
 
Usage:
  python run.py ingest          # Chunk + embed + index documents in ./data/
  python run.py chat            # Start interactive agent
  python run.py evaluate        # Run full evaluation suite (produces CSV)
  python run.py demo            # Run 3 demo queries (one of each type)
 
The assignment requires the evaluation to run with a single command:
  python run.py evaluate
"""
 
import sys
import os
 
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
 
from dotenv import load_dotenv
load_dotenv()
 
 
def cmd_ingest():
    from ingestion import index_documents
    data_dir = "./data"
    if not os.path.isdir(data_dir):
        print(f"Error: '{data_dir}' directory not found.")
        print("Please place the 4 AI regulation documents in ./data/")
        sys.exit(1)
    index_documents(data_dir, reset=True)
 
 
def cmd_chat():
    from agent import run_agent
    run_agent()
 
 
def cmd_evaluate():
    from evaluate import run_evaluation
    run_evaluation()
 
 
def cmd_demo():
    from ingestion import get_collection
    from router import route_query
    from generator import generate_answer
 
    collection = get_collection()
    if collection.count() == 0:
        print("Knowledge base is empty. Run: python run.py ingest")
        sys.exit(1)
 
    demo_queries = [
        ("FACTUAL",      "What are the penalties for violating the EU AI Act?"),
        ("SYNTHESIS",    "Compare how the documents approach AI transparency requirements."),
        ("OUT_OF_SCOPE", "What is the best recipe for banana bread?"),
    ]
 
    for expected, query in demo_queries:
        print(f"\n{'='*65}")
        print(f"[Expected: {expected}]")
        print(f"Query: {query}")
        decision = route_query(query, collection)
        result   = generate_answer(query, decision)
        print(f"Routed as: {result['query_type']}")
        print(f"Reasoning: {result['routing_reasoning']}")
        print(f"\nAnswer:\n{result['answer']}")
        print(f"Sources: {result['sources_used']}")
 
 
COMMANDS = {
    "ingest":   cmd_ingest,
    "chat":     cmd_chat,
    "evaluate": cmd_evaluate,
    "demo":     cmd_demo,
}
 
if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print("Usage: python run.py [ingest | chat | evaluate | demo]")
        sys.exit(1)
    COMMANDS[sys.argv[1]]()