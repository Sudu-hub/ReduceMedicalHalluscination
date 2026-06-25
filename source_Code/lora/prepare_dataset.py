import json

from datasets import Dataset


def load_pubmed_dataset():

    with open(
        "02_Data/expert_corpus/SPLUS_pubmedqa.json",
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    samples = []

    for item in data:

        prompt = f"""
Question:
{item["question"]}

Evidence:
{item["abstract"]}

Answer:
"""

        samples.append({

            "text":
                prompt +
                item["long_answer"]

        })

    return Dataset.from_list(
        samples
    )


if __name__ == "__main__":

    dataset = load_pubmed_dataset()

    print(dataset)

    print(dataset[0])