import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from peft import (
    PeftModel
)

from source_Code.pncd.config import *


class ModelManager:

    def __init__(self):

        self.base_model_name = BASE_MODEL

        print("Loading Tokenizer...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name
        )

        print("Loading Base Model...")

        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            device_map="auto"
        )

        print("Loading Expert LoRA...")

        self.expert_model = PeftModel.from_pretrained(
            self.base_model,
            EXPERT_LORA_PATH
        )

        print("Loading NonExpert LoRA...")

        base_copy = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            device_map="auto"
        )

        self.nonexpert_model = PeftModel.from_pretrained(
            base_copy,
            NONEXPERT_LORA_PATH
        )

        self.base_model.eval()
        self.expert_model.eval()
        self.nonexpert_model.eval()

        for model in [
            self.base_model,
            self.expert_model,
            self.nonexpert_model
        ]:
            for p in model.parameters():
                p.requires_grad = False

        print("Models Loaded Successfully")