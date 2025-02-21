import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

import logging
logging.basicConfig(
filename="step.log",
encoding="utf-8",
filemode="a",
format="{asctime} - {levelname} - {message}",
style="{",
datefmt="%Y-%m-%d %H:%M",
)

logging.warning("import done")

# Load DeepSeek V3 model & tokenizer
MODEL_NAME = "deepseek-ai/deepseek-llm-7b-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

logging.warning("model loaded")

# Load dataset (customize this part)
# dataset = load_dataset("your_dataset")  # Example: JSON format {"input": "wrong sentence", "output": "corrected sentence"}
# Example dataset
dataset = {
    "input": ["Ths is a smple txt.", "I hav a drem."],
    "output": ["This is a simple text.", "I have a dream."]
}

# Tokenization function
def tokenize_function(examples):
    inputs = [f"Fix the spelling: {text}" for text in examples["input"]]
    labels = [text for text in examples["output"]]
    model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=512)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(labels, padding="max_length", truncation=True, max_length=512)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Tokenize dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)


# LoRA Configuration
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # Tune attention layers
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# Training Arguments
training_args = TrainingArguments(
    output_dir="./deepseek-spell-checker",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_steps=100,
    save_total_limit=2,
    fp16=True,  # Enable mixed precision for speed
    deepspeed="ds_config.json",  # Use DeepSpeed config
    report_to="none"
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
)

# Train the model
trainer.train()

logging.warning("training ended")

# Save fine-tuned model
model.save_pretrained("./deepseek-spell-checker")
tokenizer.save_pretrained("./deepseek-spell-checker")

logging.warning("model saved")


logging.warning("testing started")

# Test the model
def correct_text(text):
    inputs = tokenizer(f"Fix the spelling: {text}", return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_length=512)
    return tokenizer.decode(output[0], skip_special_tokens=True)

print(correct_text("i red a book"))
