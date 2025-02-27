from datasets import load_dataset
import logging
from transformers import T5Tokenizer
from transformers import T5ForConditionalGeneration
from torch.utils.data import DataLoader
import torch
from torch.optim import AdamW
from tqdm import tqdm

logging.basicConfig(
filename="step.log",
encoding="utf-8",
filemode="a",
format="{asctime} - {levelname} - {message}",
style="{",
datefmt="%Y-%m-%d %H:%M",
)

logging.warning("import done")

# Load the CSV dataset
dataset = load_dataset("csv", data_files="csv_file.csv")

logging.warning("dataset loaded")

# Load the T5 tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-base")
# Load the T5 model
model = T5ForConditionalGeneration.from_pretrained("t5-base")

logging.warning("model loaded")

def tokenize_function(examples):
    # Format the input as "correct: <misspelled sentence>"
    inputs = ["correct: " + text for text in examples["incorr"]]
    targets = examples["corr"]

    # Tokenize inputs and targets
    model_inputs = tokenizer(inputs, max_length=64, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=64, truncation=True, padding="max_length").input_ids

    # Replace padding token IDs with -100 (ignored by the loss function)
    labels = [
        [(label if label != tokenizer.pad_token_id else -100) for label in labels_example]
        for labels_example in labels
    ]

    model_inputs["labels"] = labels
    return model_inputs

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["incorr", "corr"])


logging.warning("dataset tokenized")

train_dataloader = DataLoader(tokenized_dataset["train"], batch_size=8, shuffle=True)


logging.warning("training started")
# Set up the optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training parameters
epochs = 3

# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")

    for step, batch in enumerate(progress_bar):
        # Move batch to GPU
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=total_loss / (step + 1))

# Save the fine-tuned model
model.save_pretrained("./fine-tuned-t5-spelling-correction")
tokenizer.save_pretrained("./fine-tuned-t5-spelling-correction")

# Save the fine-tuned model

logging.warning("model saved")

model.save_pretrained("./fine-tuned-t5-spelling-correction")
tokenizer.save_pretrained("./fine-tuned-t5-spelling-correction")

logging.warning("saved model loaded for testing")

def correct_spelling(text):
    # Format the input as "correct: <misspelled sentence>"
    input_text = "correct: " + text

    # Tokenize the input
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    # Generate the corrected text
    outputs = model.generate(**inputs, max_length=64)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
misspelled_text = "I wnt to the stor to buy some breaad."
corrected_text = correct_spelling(misspelled_text)
print(f"Original: {misspelled_text}")
print(f"Corrected: {corrected_text}")

logging.warning("testing done")



# Example usage
misspelled_text = "I red a book."
corrected_text = correct_spelling(misspelled_text)
print(f"Original: {misspelled_text}")
print(f"Corrected: {corrected_text}")

logging.warning("testing done")


# Example usage
misspelled_text = "I wnt to red a book."
corrected_text = correct_spelling(misspelled_text)
print(f"Original: {misspelled_text}")
print(f"Corrected: {corrected_text}")

logging.warning("testing done")