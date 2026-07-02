from src.graph import graph


def main():
    """
    Entry point for the Healthcare Clinic Assistant.
    """

    print("=" * 60)
    print("      Healthcare Clinic Assistant")
    print("Type 'exit' or 'quit' to end the conversation.")
    print("=" * 60)

    # Thread ID is used by LangGraph to maintain conversation memory.
    config = {
        "configurable": {
            "thread_id": "patient_001"
        }
    }

    while True:

        question = input("\nYou: ").strip()

        if question.lower() in ["exit", "quit"]:
            print("\nThank you for using the Healthcare Clinic Assistant!")
            break

        if not question:
            print("Please enter a valid question.")
            continue

        try:
            result = graph.invoke(
                {
                    "question": question
                },
                config=config
            )

            print("\nAssistant:")
            print(result["answer"])

        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()