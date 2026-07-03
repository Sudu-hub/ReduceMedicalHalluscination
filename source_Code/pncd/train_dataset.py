import json

from torch.utils.data import Dataset

from source_Code.pncd.config import *


class PredictorDataset(Dataset):

    def __init__(self):

        print("=" * 60)
        print("Loading Predictor Dataset...")
        print("=" * 60)

        with open(
            "02_Data/expert_corpus/SPLUS_pubmedqa.json",
            "r",
            encoding="utf-8"
        ) as f:

            self.data = json.load(f)

        print(f"Total Samples : {len(self.data)}")

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        sample = self.data[idx]

        question = sample["question"].strip()

        abstract = sample["abstract"].strip()

        answer = sample["long_answer"].strip()

        return {

            "question": question,

            "abstract": abstract,

            "answer": answer

        }


if __name__ == "__main__":

    dataset = PredictorDataset()

    print("=" * 60)
    print(dataset[0])
    print("=" * 60)