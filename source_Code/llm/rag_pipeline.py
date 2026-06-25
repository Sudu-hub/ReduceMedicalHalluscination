import faiss
import pickle

from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

from source_Code.retrieval.embedder import get_embeddings

# Change model
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

expert_index = faiss.read_index(
    "02_Data/processed/expert_index.faiss"
)

with open(
    "02_Data/processed/expert_docs.pkl",
    "rb"
) as f:
    expert_docs = pickle.load(f)


def retrieve_context(query):
    q = get_embeddings([query])

    _, ids = expert_index.search(
        q,
        2
    )

    contexts = []

    for idx in ids[0]:
        contexts.append(expert_docs[idx])

    return "\n".join(contexts)


def answer_question(query):
    context = retrieve_context(query)

    prompt = f"""You are a helpful medical assistant.

Use only the provided context to answer the question.
If the answer is not present in the context, say you don't know.

Context:
{context}

Question:
{query}

Answer:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.2,
        do_sample=False
    )

    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return answer


if __name__ == "__main__":

    while True:

        q = input("\nAsk Medical Question: ")

        if q.lower() == "exit":
            break

        result = answer_question(q)

        print("\n")
        print(result)