import faiss
import pickle

from embedder import get_embeddings


# Load Expert Index
expert_index = faiss.read_index(
    "02_Data/processed/expert_index.faiss"
)

# Load NonExpert Index
nonexpert_index = faiss.read_index(
    "02_Data/processed/nonexpert_index.faiss"
)


# Load Expert Documents
with open(
    "02_Data/processed/expert_docs.pkl",
    "rb"
) as f:

    expert_docs = pickle.load(f)


# Load NonExpert Documents
with open(
    "02_Data/processed/nonexpert_docs.pkl",
    "rb"
) as f:

    nonexpert_docs = pickle.load(f)


def search(query, top_k=2):

    query_vector = get_embeddings([query])

    _, expert_ids = expert_index.search(
        query_vector,
        top_k
    )

    _, nonexpert_ids = nonexpert_index.search(
        query_vector,
        top_k
    )

    print("\n" + "=" * 60)
    print("EXPERT RETRIEVAL")
    print("=" * 60)

    for idx in expert_ids[0]:
        print(expert_docs[idx])
        print("-" * 60)

    print("\n" + "=" * 60)
    print("NON-EXPERT RETRIEVAL")
    print("=" * 60)

    for idx in nonexpert_ids[0]:
        print(nonexpert_docs[idx])
        print("-" * 60)


if __name__ == "__main__":

    while True:

        query = input(
            "\nAsk Medical Question (type 'exit' to quit): "
        )

        if query.lower() == "exit":
            break

        search(query)