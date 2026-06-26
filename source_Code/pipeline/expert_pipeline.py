import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from peft import (
    PeftModel
)

from source_Code.retrieval.retriever import (
    DualRetriever
)

from source_Code.llm.prompt_builder import (
    PromptBuilder
)


BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

LORA_PATH = "07_Models/expert_lora"


print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL
)

print("Loading Base Model...")

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto"
)

print("Loading Expert LoRA...")

expert_model = PeftModel.from_pretrained(
    base_model,
    LORA_PATH
)

retriever = DualRetriever()


def generate(model, prompt):

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():

        outputs = model.generate(

            **inputs,

            max_new_tokens=200,

            do_sample=False,

            temperature=0.0

        )

    generated = outputs[0][inputs.input_ids.shape[1]:]

    return tokenizer.decode(
        generated,
        skip_special_tokens=True
    )


while True:

    question = input(
        "\nAsk Medical Question (exit to quit): "
    )

    if question.lower() == "exit":
        break

    retrieval = retriever.retrieve(
        question
    )

    prompt = PromptBuilder.build_prompt(

        question,

        retrieval["expert"]

    )

    print("\n")
    print("="*80)
    print("TOP RETRIEVED DOCUMENT")
    print("="*80)

    print(
        retrieval["expert"][0]["question"]
    )

    print()

    print("="*80)
    print("BASE MODEL")
    print("="*80)

    print(
        generate(
            base_model,
            prompt
        )
    )

    print()

    print("="*80)
    print("EXPERT LORA")
    print("="*80)

    print(
        generate(
            expert_model,
            prompt
        )
    )