from pathlib import Path
import shutil

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document


# -----------------------------
# Config
# -----------------------------
PERSIST_DIR = "my_chroma_db"
COLLECTION_NAME = "sample"
EMBEDDING_MODEL = "nomic-embed-text"

# For experiments, reset DB every run.
# Set this to False if you want to keep old data.
RESET_DB = True


def print_results(title, results):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    for item in results:
        # For similarity_search: item is Document
        if isinstance(item, Document):
            print(f"\nContent: {item.page_content}")
            print(f"Metadata: {item.metadata}")

        # For similarity_search_with_score: item is tuple(Document, score)
        elif isinstance(item, tuple):
            doc, score = item
            print(f"\nScore: {score}")
            print(f"Content: {doc.page_content}")
            print(f"Metadata: {doc.metadata}")


def main():
    # -----------------------------
    # Optional: reset local Chroma DB
    # -----------------------------
    if RESET_DB and Path(PERSIST_DIR).exists():
        shutil.rmtree(PERSIST_DIR)

    # -----------------------------
    # Create LangChain documents
    # -----------------------------
    doc1 = Document(
        page_content=(
            "Virat Kohli is one of the most successful and consistent batsmen in IPL history. "
            "Known for his aggressive batting style and fitness, he has led the Royal Challengers "
            "Bangalore in multiple seasons."
        ),
        metadata={"team": "Royal Challengers Bangalore", "player": "Virat Kohli"},
    )

    doc2 = Document(
        page_content=(
            "Rohit Sharma is the most successful captain in IPL history, leading Mumbai Indians "
            "to five titles. He's known for his calm demeanor and ability to play big innings "
            "under pressure."
        ),
        metadata={"team": "Mumbai Indians", "player": "Rohit Sharma"},
    )

    doc3 = Document(
        page_content=(
            "MS Dhoni, famously known as Captain Cool, has led Chennai Super Kings to multiple "
            "IPL titles. His finishing skills, wicketkeeping, and leadership are legendary."
        ),
        metadata={"team": "Chennai Super Kings", "player": "MS Dhoni"},
    )

    doc4 = Document(
        page_content=(
            "Jasprit Bumrah is considered one of the best fast bowlers in T20 cricket. "
            "Playing for Mumbai Indians, he is known for his yorkers and death-over expertise."
        ),
        metadata={"team": "Mumbai Indians", "player": "Jasprit Bumrah"},
    )

    doc5 = Document(
        page_content=(
            "Ravindra Jadeja is a dynamic all-rounder who contributes with both bat and ball. "
            "Representing Chennai Super Kings, his quick fielding and match-winning performances "
            "make him a key player."
        ),
        metadata={"team": "Chennai Super Kings", "player": "Ravindra Jadeja"},
    )

    docs = [doc1, doc2, doc3, doc4, doc5]

    # Important:
    # Use fixed IDs so update/delete becomes easy.
    ids = [
        "virat-kohli",
        "rohit-sharma",
        "ms-dhoni",
        "jasprit-bumrah",
        "ravindra-jadeja",
    ]

    # -----------------------------
    # Local Ollama embeddings
    # -----------------------------
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    # -----------------------------
    # Create Chroma vector store
    # -----------------------------
    vector_store = Chroma(
        collection_name=COLLECTION_NAME, 
            # Name of the Chroma collection.
        # A collection is like a table/folder inside ChromaDB where your documents,
        # embeddings, metadata, and IDs are stored together.
        # Example: COLLECTION_NAME = "sample"
        embedding_function=embeddings,  
        # which embedding model we use
        # The embedding model used to convert text into vectors.
        # Whenever you add documents, Chroma uses this model to create embeddings.
        # Whenever you search with a query, the same model converts the query into a vector.
        # Example: embeddings = OllamaEmbeddings(model="nomic-embed-text")
        persist_directory=PERSIST_DIR,   
        # where data is stored
        # Local folder where Chroma stores the vector database files.
        # This allows your data to stay saved even after the Python script stops.
        # Example: PERSIST_DIR = "my_chroma_db"
    )

    # -----------------------------
    # Add documents
    # -----------------------------
    vector_store.add_documents(documents=docs, ids=ids)

    # -----------------------------
    # View documents
    # -----------------------------
    all_docs = vector_store.get(include=["documents", "metadatas"])

    print("\nAll documents in Chroma DB:")
    print(all_docs)

    # -----------------------------
    # Similarity search
    # -----------------------------
    search_results = vector_store.similarity_search(
        query="Who among these are a bowler?",
        k=2,   # How many similar vector need to display in result
    )

    print_results("Similarity Search Results", search_results)

    # -----------------------------
    # Similarity search with score
    # Lower score usually means more similar in Chroma distance-based results
    # -----------------------------
    scored_results = vector_store.similarity_search_with_score(
        query="Who among these are a bowler?",
        k=2,
    )

    print_results("Similarity Search With Score Results", scored_results)

    # -----------------------------
    # Metadata filtering
    # -----------------------------
    csk_results = vector_store.similarity_search_with_score(
        query="IPL player",
        k=5,
        filter={"team": "Chennai Super Kings"},
    )

    print_results("Metadata Filter Results: Chennai Super Kings", csk_results)

    # -----------------------------
    # Update document
    # -----------------------------
    updated_doc1 = Document(
        page_content=(
            "Virat Kohli, the former captain of Royal Challengers Bangalore, is renowned for "
            "his aggressive leadership and consistent batting performances. He holds the record "
            "for the most runs in IPL history, including multiple centuries in a single season. "
            "Despite RCB not winning an IPL title under his captaincy, Kohli's passion and "
            "fitness set a benchmark for the league. His ability to chase targets and anchor "
            "innings has made him one of the most dependable players in T20 cricket."
        ),
        metadata={"team": "Royal Challengers Bangalore", "player": "Virat Kohli"},
    )

    vector_store.update_document(
        document_id="virat-kohli",
        document=updated_doc1,
    )

    after_update = vector_store.get(
        ids=["virat-kohli"],
        include=["documents", "metadatas"],
    )

    print("\nAfter updating Virat Kohli document:")
    print(after_update)

    # -----------------------------
    # Delete document
    # -----------------------------
    vector_store.delete(ids=["virat-kohli"])

    after_delete = vector_store.get(include=["documents", "metadatas"])

    print("\nAfter deleting Virat Kohli document:")
    print(after_delete)


if __name__ == "__main__":
    main()