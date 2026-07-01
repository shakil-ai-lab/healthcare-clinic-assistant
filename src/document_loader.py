from langchain_community.document_loaders import DirectoryLoader, TextLoader


def load_documents():
    """
    Load all .txt files from the docs folder.
    """

    loader = DirectoryLoader(
        "docs",
        glob="*.txt",
        loader_cls=TextLoader
    )

    documents = loader.load()

    return documents


if __name__ == "__main__":

    docs = load_documents()

    print(f"Total documents loaded: {len(docs)}\n")

    for i, doc in enumerate(docs, start=1):

        print(f"Document {i}")

        print(doc.metadata)

        print(doc.page_content)

        print("-" * 50)