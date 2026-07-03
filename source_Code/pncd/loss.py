import torch
import torch.nn as nn


class PNCDLoss(nn.Module):

    def __init__(self):

        super().__init__()

        self.loss = nn.CrossEntropyLoss()

    def forward(

        self,

        logits,

        labels

    ):

        logits = logits[:, :-1, :].contiguous()

        labels = labels[:, 1:].contiguous()

        loss = self.loss(

            logits.view(

                -1,

                logits.size(-1)

            ),

            labels.view(-1)

        )

        return loss