import torch

from source_Code.pncd.model_manager import ModelManager


class LogitExtractor:

    def __init__(self):

        self.manager = ModelManager()

        self.tokenizer = self.manager.tokenizer

        self.base_model = self.manager.base_model

        self.expert_model = self.manager.expert_model

        self.nonexpert_model = self.manager.nonexpert_model

    def get_logits(
        self,
        prompt
    ):

        inputs = self.tokenizer(

            prompt,

            return_tensors="pt"

        )

        inputs = {

            k: v.to(self.base_model.device)

            for k, v in inputs.items()

        }

        with torch.no_grad():

            base_logits = self.base_model(
                **inputs
            ).logits

            expert_logits = self.expert_model(
                **inputs
            ).logits

            nonexpert_logits = self.nonexpert_model(
                **inputs
            ).logits

        return {

            "base": base_logits,

            "expert": expert_logits,

            "nonexpert": nonexpert_logits

        }