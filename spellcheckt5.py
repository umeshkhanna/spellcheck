import pandas as pd
from datasets import Dataset
import logging
from transformers import T5ForConditionalGeneration, T5Tokenizer, TrainingArguments, Trainer

logging.basicConfig(
filename="step.log",
encoding="utf-8",
filemode="a",
format="{asctime} - {levelname} - {message}",
style="{",
datefmt="%Y-%m-%d %H:%M",
)

logging.warning("import done")

# Load model and tokenizer
MODEL_NAME = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to("cuda")

logging.warning("model loaded")

# Load dataset
df = pd.read_csv("csv_file.csv")
# Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(df)

logging.warning("dataset loaded")

def tokenize_function(examples):
    # Format input as "Fix spelling: {incorrect text}"
    inputs = [f"Fix spelling: {text}" for text in examples["incorr"]]
    targets = [text for text in examples["corr"]]

    # Tokenize inputs & outputs
    model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=128)
    labels = tokenizer(targets, padding="max_length", truncation=True, max_length=128)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Apply tokenization
tokenized_dataset = dataset.map(tokenize_function, batched=True)

logging.warning("dataset tokenized")

train_test_split = tokenized_dataset.train_test_split(test_size=0.1, seed=42)

train_dataset = train_test_split["train"]
test_dataset = train_test_split["test"]

logging.warning("dataset train test split")

training_args = TrainingArguments(
    output_dir="./t5-spell-checker",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    save_steps=1000,
    save_total_limit=2,
    eval_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

logging.warning("training started")

trainer.train()

logging.warning("training ended")

tokenizer.save_pretrained('t5_um_spellchecker_model')

logging.warning("tokenizer saved")

trainer.save_model('t5_um_spellchecker_model')

logging.warning("model saved")

# Load model and tokenizer
MODEL_NAME = "t5_um_spellchecker_model"
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to("cuda")

def correct_spelling(text):
    prompt = f"Fix spelling: {text}"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    outputs = model.generate(**inputs, max_length=100)
    corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected_text

# Example
text = "I am goin to the park."
print(correct_spelling(text))
logging.warning("testing done")

text = "I wnt to the stor to buy some breaad."
print(correct_spelling(text))
logging.warning("testing done")

text = "I red a book."
print(correct_spelling(text))
logging.warning("testing done")

text = "I wnt to red a book."
print(correct_spelling(text))
logging.warning("testing done")