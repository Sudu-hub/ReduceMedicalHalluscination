import json
import pickle
import os

import faiss
import numpy as np

from embedder import get_embeddings


PROCESSED_DIR = "02_Data/processed"

os.makedirs(PROCESSED_DIR, exist_ok=True)


def normalize(vectors):
    vectors = np.asarray(vectors, dtype=np.float32)
    faiss.normalize_L2(vectors)
    return vectors


def save_index(documents, embedding_texts, index_name, docs_name):

    vectors = get_embeddings(embedding_texts)
    vectors = normalize(vectors)

    dimension = vectors.shape[1]

    # Cosine Similarity
    index = faiss.IndexFlatIP(dimension)

    index.add(vectors)

    faiss.write_index(
        index,
        os.path.join(PROCESSED_DIR, index_name)
    )

    with open(
        os.path.join(PROCESSED_DIR, docs_name),
        "wb"
    ) as f:
        pickle.dump(documents, f)

    print(f"{index_name} saved successfully.")
    print(f"Total Documents : {len(documents)}")


def build_expert_index():

    with open(
        "02_Data/expert_corpus/SPLUS_pubmedqa.json",
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    documents = []
    embedding_texts = []

    for item in data:

        question = item.get("question", "").strip()
        abstract = item.get("abstract", "").strip()
        answer = item.get("long_answer", "").strip()

        embedding_text = f"{question}\n{abstract}"

        documents.append({

            "question": question,

            "abstract": abstract,

            "answer": answer,

            "embedding_text": embedding_text

        })

        embedding_texts.append(
            embedding_text
        )

    save_index(
        documents,
        embedding_texts,
        "expert_index.faiss",
        "expert_docs.pkl"
    )


def build_nonexpert_index():

    with open(
        "02_Data/non_expert_corpus/non_expert_500.json",
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    documents = []
    embedding_texts = []

    for item in data:

        question = item.get("question", "").strip()
        answer = item.get("long_answer", "").strip()

        embedding_text = question

        documents.append({

            "question": question,

            "answer": answer,

            "embedding_text": embedding_text

        })

        embedding_texts.append(
            embedding_text
        )

    save_index(
        documents,
        embedding_texts,
        "nonexpert_index.faiss",
        "nonexpert_docs.pkl"
    )


if __name__ == "__main__":

    print("=" * 60)
    print("Building Expert FAISS Index")
    print("=" * 60)

    build_expert_index()

    print()

    print("=" * 60)
    print("Building NonExpert FAISS Index")
    print("=" * 60)

    build_nonexpert_index()

    print()
    print("=" * 60)
    print("All Indexes Created Successfully")
    print("=" * 60)