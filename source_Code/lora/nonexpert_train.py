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

from source_Code.lora.prepare_nonexpert_dataset import (
    load_nonexpert_dataset
)

from source_Code.lora.train_config import *


print("Loading Dataset...")

dataset = load_nonexpert_dataset()

print("Loading Tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

tokenizer.pad_token = tokenizer.eos_token

print("Loading Model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
)

print("Applying LoRA...")

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

print(model.print_trainable_parameters())


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
    tokenize
)

training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,

    per_device_train_batch_size=BATCH_SIZE,

    gradient_accumulation_steps=GRADIENT_ACCUMULATION,

    learning_rate=LEARNING_RATE,

    num_train_epochs=NUM_EPOCHS,

    logging_steps=10,

    save_strategy="epoch",

    report_to="none"
)

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=dataset,

    data_collator=DataCollatorForLanguageModeling(
        tokenizer,
        mlm=False
    )
)

print("Starting Expert LoRA Training...")

trainer.train()

print("Saving LoRA...")

model.save_pretrained(
    OUTPUT_DIR
)

tokenizer.save_pretrained(
    OUTPUT_DIR
)

print("Training Complete!")