from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from document_loader import load_documents


def create_vector_store():
    """
    Load documents, split them into chunks,
    create embeddings, and save them in a FAISS vector store.
    """

    # Load documents
    documents = load_documents()

    print(f"Loaded {len(documents)} documents.")

    # Create text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    # Split documents into chunks
    chunks = text_splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks.")

    # Load embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create FAISS vector store
    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    # Save locally
    vector_store.save_local("vectorstore")

    print("Vector Store saved successfully!")

    # return vector_store


if __name__ == "__main__":
    create_vector_store()