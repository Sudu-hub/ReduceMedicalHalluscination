import pickle
import faiss
import numpy as np

from source_Code.retrieval.embedder import get_embeddings


class DualRetriever:

    def __init__(self):

        self.expert_index = faiss.read_index(
            "02_Data/processed/expert_index.faiss"
        )

        self.nonexpert_index = faiss.read_index(
            "02_Data/processed/nonexpert_index.faiss"
        )

        with open(
            "02_Data/processed/expert_docs.pkl",
            "rb"
        ) as f:

            self.expert_docs = pickle.load(f)

        with open(
            "02_Data/processed/nonexpert_docs.pkl",
            "rb"
        ) as f:

            self.nonexpert_docs = pickle.load(f)

    def retrieve(
        self,
        query,
        top_k=3
    ):

        # -------------------------------------------------
        # Query Embedding
        # -------------------------------------------------

        query_vector = get_embeddings([query])

        query_vector = np.asarray(
            query_vector,
            dtype=np.float32
        )

        # Normalize because FAISS index uses cosine similarity
        faiss.normalize_L2(query_vector)

        # -------------------------------------------------
        # Expert Search
        # -------------------------------------------------

        expert_scores, expert_ids = self.expert_index.search(
            query_vector,
            top_k
        )

        expert_results = []

        for score, idx in zip(
            expert_scores[0],
            expert_ids[0]
        ):

            doc = self.expert_docs[idx].copy()

            doc["score"] = float(score)

            expert_results.append(doc)

        # -------------------------------------------------
        # NonExpert Search
        # -------------------------------------------------

        nonexpert_scores, nonexpert_ids = self.nonexpert_index.search(
            query_vector,
            top_k
        )

        nonexpert_results = []

        for score, idx in zip(
            nonexpert_scores[0],
            nonexpert_ids[0]
        ):

            doc = self.nonexpert_docs[idx].copy()

            doc["score"] = float(score)

            nonexpert_results.append(doc)

        # -------------------------------------------------
        # Return
        # -------------------------------------------------

        return {

            "expert": expert_results,

            "nonexpert": nonexpert_results,

            "expert_scores": [

                doc["score"]

                for doc in expert_results

            ],

            "nonexpert_scores": [

                doc["score"]

                for doc in nonexpert_results

            ]

        }


if __name__ == "__main__":

    retriever = DualRetriever()

    result = retriever.retrieve(
        "Can aspirin help stroke patients?"
    )

    print("\n========== EXPERT ==========\n")

    for doc in result["expert"]:

        print(doc["question"])
        print("Score:", doc["score"])
        print("-" * 50)

    print("\n========== NONEXPERT ==========\n")

    for doc in result["nonexpert"]:

        print(doc["question"])
        print("Score:", doc["score"])
        print("-" * 50)

    print("\nExpert Scores:")
    print(result["expert_scores"])

    print("\nNonExpert Scores:")
    print(result["nonexpert_scores"])