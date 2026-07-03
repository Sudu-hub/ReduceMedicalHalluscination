import numpy as np
import torch


class FeatureExtractor:

    @staticmethod
    def extract(

        expert_scores,

        nonexpert_scores

    ):

        expert_scores = np.array(expert_scores)

        nonexpert_scores = np.array(nonexpert_scores)

        features = [

            expert_scores.mean(),

            expert_scores.max(),

            expert_scores.std(),

            nonexpert_scores.mean(),

            nonexpert_scores.max(),

            nonexpert_scores.std(),

            expert_scores.mean() -

            nonexpert_scores.mean()

        ]

        features = np.array(

            features,

            dtype=np.float32

        )

        return torch.tensor(

            features

        ).unsqueeze(0)