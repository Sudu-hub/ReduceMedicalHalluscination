import json
import torch

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

from peft import (
    LoraConfig,
    get_peft_model
)

# =====================================================
# CONFIGURATION
# =====================================================

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

DATA_PATH = "/content/ReduceMedicalHalluscination/02_Data/non_expert_corpus/non_expert_500.json"

OUTPUT_DIR = "/content/ReduceMedicalHalluscination/07_Models/nonexpert_lora"

MAX_LENGTH = 1024

BATCH_SIZE = 1

GRADIENT_ACCUMULATION = 8

LEARNING_RATE = 2e-4

NUM_EPOCHS = 3

LORA_R = 8

LORA_ALPHA = 16

LORA_DROPOUT = 0.05


# =====================================================
# LOAD DATASET
# =====================================================

def load_nonexpert_dataset():

    with open(
        DATA_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    samples = []

    for item in data:

        prompt = f"""Question:
{item["question"]}

Answer:
"""

        samples.append({

            "text": prompt + item["long_answer"]

        })

    return Dataset.from_list(samples)


print("=" * 60)
print("Loading Dataset...")
print("=" * 60)

dataset = load_nonexpert_dataset()

print(dataset)

print(dataset[0])


# =====================================================
# TOKENIZER
# =====================================================

print("=" * 60)
print("Loading Tokenizer...")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

tokenizer.pad_token = tokenizer.eos_token


# =====================================================
# MODEL
# =====================================================

print("=" * 60)
print("Loading Model...")
print("=" * 60)

model = AutoModelForCausalLM.from_pretrained(

    MODEL_NAME,

    torch_dtype=torch.float16,

    device_map="auto"

)


# =====================================================
# APPLY LORA
# =====================================================

print("=" * 60)
print("Applying LoRA...")
print("=" * 60)

lora_config = LoraConfig(

    r=LORA_R,

    lora_alpha=LORA_ALPHA,

    lora_dropout=LORA_DROPOUT,

    bias="none",

    task_type="CAUSAL_LM",

    target_modules=[

        "q_proj",

        "k_proj",

        "v_proj",

        "o_proj"

    ]

)

model = get_peft_model(

    model,

    lora_config

)

model.print_trainable_parameters()


# =====================================================
# TOKENIZATION
# =====================================================

def tokenize(example):

    tokens = tokenizer(

        example["text"],

        truncation=True,

        padding="max_length",

        max_length=MAX_LENGTH

    )

    tokens["labels"] = tokens["input_ids"].copy()

    return tokens


dataset = dataset.map(
    tokenize,
    remove_columns=["text"]
)

print(dataset.column_names)


# =====================================================
# TRAINING ARGUMENTS
# =====================================================

training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,

    per_device_train_batch_size=BATCH_SIZE,

    gradient_accumulation_steps=GRADIENT_ACCUMULATION,

    learning_rate=LEARNING_RATE,

    num_train_epochs=NUM_EPOCHS,

    logging_steps=10,

    save_strategy="epoch",

    report_to="none",

    fp16=True,

    remove_unused_columns=False,

    optim="adamw_torch",

    lr_scheduler_type="cosine",

    warmup_steps=50

)


# =====================================================
# TRAINER
# =====================================================

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=dataset,

    processing_class=tokenizer,

    data_collator=DataCollatorForLanguageModeling(

        tokenizer=tokenizer,

        mlm=False

    )

)

# =====================================================
# TRAIN
# =====================================================

print("=" * 60)
print("Starting NonExpert LoRA Training...")
print("=" * 60)

trainer.train()


# =====================================================
# SAVE MODEL
# =====================================================

print("=" * 60)
print("Saving NonExpert LoRA...")
print("=" * 60)

model.save_pretrained(
    OUTPUT_DIR
)

tokenizer.save_pretrained(
    OUTPUT_DIR
)

print("=" * 60)
print("Training Completed Successfully!")
print("=" * 60)