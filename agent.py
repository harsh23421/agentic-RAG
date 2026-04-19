"""
agent.py — Interactive CLI agent for live demo.
 
Usage:
  python agent.py
 
The agent:
  1. Loads the ChromaDB collection
  2. Accepts user queries interactively
  3. Routes, retrieves, and generates answers
  4. Prints routing decision + answer with full transparency
"""
 
import os
import sys
from dotenv import load_dotenv
 
load_dotenv()
 
from ingestion import get_collection
from router import route_query, FACTUAL, SYNTHESIS, OUT_OF_SCOPE
from generator import generate_answer
 
 
BANNER = """
╔══════════════════════════════════════════════════════════╗
║        Agentic RAG — AI Regulation Q&A System           ║
║        Documents: EU AI Act + related frameworks        ║
║        Type 'quit' to exit | 'debug' to toggle debug    ║
╚══════════════════════════════════════════════════════════╝
"""
 
TYPE_COLORS = {
    FACTUAL:      "\033[92m",   # green
    SYNTHESIS:    "\033[94m",   # blue
    OUT_OF_SCOPE: "\033[91m",   # red
}
RESET = "\033[0m"
BOLD  = "\033[1m"
 
 
def print_decision(decision, debug: bool):
    color = TYPE_COLORS.get(decision.query_type, "")
    print(f"\n{BOLD}Query type:{RESET} {color}{decision.query_type}{RESET}  "
          f"(confidence: {decision.confidence:.3f})")
    if debug:
        print(f"{BOLD}Routing reasoning:{RESET} {decision.reasoning}")
        if decision.top_chunks:
            print(f"\n{BOLD}Top retrieved chunks:{RESET}")
            for i, ch in enumerate(decision.top_chunks[:3], 1):
                print(f"  [{i}] source={ch['source']}  score={ch['score']:.3f}")
                print(f"      {ch['text'][:120]}...")
 
 
def print_answer(result: dict):
    print(f"\n{BOLD}Answer:{RESET}")
    print(result["answer"])
    if result["sources_used"]:
        print(f"\n{BOLD}Sources:{RESET} {', '.join(result['sources_used'])}")
 
 
def run_agent():
    print(BANNER)
 
    print("Loading knowledge base...")
    try:
        collection = get_collection()
        count = collection.count()
        if count == 0:
            print("⚠️  ChromaDB collection is empty!")
            print("   Run first: python ingestion.py ./data")
            sys.exit(1)
        print(f"✓ Loaded {count} chunks from ChromaDB.\n")
    except Exception as e:
        print(f"Error loading collection: {e}")
        sys.exit(1)
 
    debug = False
 
    while True:
        try:
            query = input(f"{BOLD}You:{RESET} ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break
 
        if not query:
            continue
        if query.lower() == "quit":
            print("Goodbye!")
            break
        if query.lower() == "debug":
            debug = not debug
            print(f"Debug mode: {'ON' if debug else 'OFF'}")
            continue
 
        print("\nThinking...", end="\r")
 
        try:
            decision = route_query(query, collection)
            print_decision(decision, debug)
 
            result = generate_answer(query, decision)
            print_answer(result)
 
        except Exception as e:
            print(f"Error: {e}")
 
        print("\n" + "─"*60)
 
 
if __name__ == "__main__":
    run_agent()
 