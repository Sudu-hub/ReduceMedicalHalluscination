import torch

from source_Code.llm.model_loader import (
    tokenizer,
    model
)


class Generator:

    @staticmethod
    def generate(

        prompt,

        max_tokens=200

    ):

        inputs = tokenizer(

            prompt,

            return_tensors="pt"

        )

        with torch.no_grad():

            outputs = model.generate(

                **inputs,

                max_new_tokens=max_tokens,

                do_sample=False

            )
            

        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(

            generated_tokens,

            skip_special_tokens=True

        )

        return response