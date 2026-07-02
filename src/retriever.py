from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def get_retriever():
    """
    Load the saved FAISS vector store
    and return a retriever.
    """

    # Load the same embedding model that was used
    # while creating the vector database
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Load the saved FAISS vector store
    vector_store = FAISS.load_local(
        folder_path="vectorstore",
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )

    # Create retriever
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    return retriever


if __name__ == "__main__":

    retriever = get_retriever()

    query = "How can I book an appointment?"

    results = retriever.invoke(query)

    print(f"\nUser Question: {query}\n")

    print("Retrieved Documents:\n")

    for i, doc in enumerate(results, start=1):

        print(f"Document {i}")

        print(doc.page_content)

        print("-" * 50)