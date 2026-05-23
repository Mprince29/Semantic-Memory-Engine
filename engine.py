from semantic_memory.application.engine import SemanticMemoryEngine


def main() -> None:
    engine = SemanticMemoryEngine()
    sample_history = [
        "I am building an Ollama project in Python and need to submit it tomorrow.",
        "Please keep it local only and avoid cloud services.",
        "Earlier I asked about Docker setup and ChromaDB embeddings.",
    ]
    engine.ingest_turns(sample_history, session_id="demo")
    query = "How should I containerize this?"

    print("=== SPL ===")
    print(engine.build_spl(query, session_id="demo"))
    print()

    try:
        print("=== Ollama Response ===")
        print(engine.answer(query, session_id="demo"))
    except Exception as exc:
        print("Ollama call skipped:", exc)


if __name__ == "__main__":
    main()
