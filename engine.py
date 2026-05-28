from semantic_memory.application.engine import SemanticRuntimeEngine


def print_section(title: str) -> None:
    print(f"=== {title} ===")


def main() -> None:
    engine = SemanticRuntimeEngine()
    session_id = "demo-v2"
    user_id = "demo-user"
    turns = [
        "I need to deploy a Flask API tomorrow. Prefer nginx, avoid docker, and keep it local only.",
        "At work, I prefer Docker for this project.",
        "I'm using Flask, Gunicorn, and Nginx, and I keep getting a 500 on startup.",
    ]

    print_section("Ingest")
    reports = engine.ingest_turns(
        turns,
        session_id=session_id,
        user_id=user_id,
        visibility="team",
    )
    for index, (turn, report) in enumerate(zip(turns, reports, strict=False), start=1):
        print(f"Turn {index}: {turn}")
        print(report.summary())
        print()

    query = "How should I debug this 500 startup deployment?"
    schema_name = "coding"
    spl, complexity = engine.build_spl(
        query,
        session_id=session_id,
        schema_name=schema_name,
        user_id=user_id,
    )

    print_section("SPL")
    print(spl)
    print()

    print_section("Retrieval Tier")
    print(
        {
            "schema": schema_name,
            "tier": complexity.tier,
            "token_budget": complexity.token_budget,
            "signals": complexity.signals,
        }
    )
    print()

    print_section("Ollama Response")
    try:
        print(
            engine.answer(
                query,
                session_id=session_id,
                schema_name=schema_name,
                user_id=user_id,
            )
        )
    except Exception as exc:
        print("Ollama call skipped:", exc)


if __name__ == "__main__":
    main()
