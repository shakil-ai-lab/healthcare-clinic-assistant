from langchain_core.tools import tool

from src.retriever import get_retriever

# Load retriever once
retriever = get_retriever()


@tool
def clinic_policy_search(query: str) -> str:
    """
    Search healthcare clinic policies about appointments,
    doctor schedules, consultation fees, lab tests,
    medicine refills, emergencies, and report collection.
    """

    retrieved_docs = retriever.invoke(query)

    if not retrieved_docs:
        return (
            "I could not find this information in the provided clinic documents."
        )

    context = []

    for doc in retrieved_docs:
        source = doc.metadata.get("source", "Unknown Source")
        context.append(
            f"Source: {source}\n{doc.page_content}"
        )
     
    return "\n\n".join(context)

if __name__ == "__main__":

    question = "How can I book an appointment?"

    result = clinic_policy_search.invoke(
        {"query": question}
    )

    print(result)