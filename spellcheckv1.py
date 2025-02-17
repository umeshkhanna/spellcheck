import logging
logging.basicConfig(
filename="step.log",
encoding="utf-8",
filemode="a",
format="{asctime} - {levelname} - {message}",
style="{",
datefmt="%Y-%m-%d %H:%M",
)

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset

logging.warning("import done")

model_name = "deepseek-ai/deepseek-llm-7b-base"  # Replace with the actual model name
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

logging.warning("model loaded")

# Example dataset
data = {
    "input": ["Ths is a smple txt.", "I hav a drem."],
    "output": ["This is a simple text.", "I have a dream."]
}

dataset = Dataset.from_dict(data)

def tokenize_function(examples):
    inputs = tokenizer(examples["input"], padding="max_length", truncation=True, return_tensors="pt")
    labels = tokenizer(examples["output"], padding="max_length", truncation=True, return_tensors="pt")
    return {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"], "labels": labels["input_ids"]}

tokenized_dataset = dataset.map(tokenize_function, batched=True)

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=500,
)

logging.warning("training started")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,  # Use a separate validation set if available
)

trainer.train()

logging.warning("training ended")

model.save_pretrained("./test-deepseek-v3")
tokenizer.save_pretrained("./test-deepseek-v3")

logging.warning("testing started")

input_text = "i red a book."
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs)
corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(corrected_text)