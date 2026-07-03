import torch
import torch.nn as nn

from source_Code.pncd.config import (
    INPUT_DIM,
    HIDDEN_DIM_1,
    HIDDEN_DIM_2,
    OUTPUT_DIM
)


class AdaptivePredictor(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                INPUT_DIM,
                HIDDEN_DIM_1
            ),

            nn.ReLU(),

            nn.Linear(
                HIDDEN_DIM_1,
                HIDDEN_DIM_2
            ),

            nn.ReLU(),

            nn.Linear(
                HIDDEN_DIM_2,
                OUTPUT_DIM
            )

        )

    def forward(self, features):

        output = self.network(features)

        alpha = torch.sigmoid(
            output[:, 0]
        )

        beta = torch.sigmoid(
            output[:, 1]
        )

        return alpha, beta

    def predict(self, features):

        self.eval()

        with torch.no_grad():

            alpha, beta = self.forward(features)

        return alpha, beta