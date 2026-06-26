import json

from datasets import Dataset


def load_nonexpert_dataset():

    with open(
        "02_Data/non_expert_corpus/non_expert_500.json",
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    samples = []

    for item in data:

        prompt = f"""
Question:
{item["question"]}

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

    dataset = load_nonexpert_dataset()

    print(dataset)

    print(dataset[0])