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

        # Generate query embedding
        query_vector = get_embeddings([query])

        query_vector = np.asarray(
            query_vector,
            dtype=np.float32
        )

        # Normalize because index uses cosine similarity
        faiss.normalize_L2(query_vector)

        # Expert Search
        expert_scores, expert_ids = self.expert_index.search(
            query_vector,
            top_k
        )

        # NonExpert Search
        nonexpert_scores, nonexpert_ids = self.nonexpert_index.search(
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

        nonexpert_results = []

        for score, idx in zip(
            nonexpert_scores[0],
            nonexpert_ids[0]
        ):

            doc = self.nonexpert_docs[idx].copy()

            doc["score"] = float(score)

            nonexpert_results.append(doc)

        return {

            "expert": expert_results,

            "nonexpert": nonexpert_results

        }