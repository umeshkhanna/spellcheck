from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the tokenizer and model
model_name = "deepseek-ai/deepseek-llm-7b-base"  # Replace with the actual model name
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


model.gradient_checkpointing_enable()

from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()  # For gradient scaling in mixed precision


from datasets import Dataset

# Example dataset of misspelled and corrected text pairs
data = {
    "misspelled": [
        "I wnt to the stor to buy some breaad.",
        "She sed she wuld come over tonite.",
        "The whether is nice todai."
    ],
    "corrected": [
        "I went to the store to buy some bread.",
        "She said she would come over tonight.",
        "The weather is nice today."
    ]
}

# Convert to Hugging Face Dataset
dataset = Dataset.from_dict(data)

# Tokenize the dataset
def tokenize_function(examples):
    inputs = tokenizer(examples["misspelled"], truncation=True, padding="max_length", max_length=64)
    labels = tokenizer(examples["corrected"], truncation=True, padding="max_length", max_length=64).input_ids
    inputs["labels"] = labels
    return inputs

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["misspelled", "corrected"])


from torch.utils.data import DataLoader

train_dataloader = DataLoader(tokenized_dataset, batch_size=2, shuffle=True)  # Adjust batch size based on GPU memory


from torch.utils.data import DataLoader

train_dataloader = DataLoader(tokenized_dataset, batch_size=2, shuffle=True)  # Adjust batch size based on GPU memory


import torch
from tqdm import tqdm

# Training parameters
epochs = 3
gradient_accumulation_steps = 4  # Accumulate gradients over multiple steps
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

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

        # Forward pass with mixed precision
        with autocast():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / gradient_accumulation_steps  # Scale loss for gradient accumulation

        # Backward pass
        scaler.scale(loss).backward()

        # Gradient accumulation
        if (step + 1) % gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=total_loss / (step + 1))


model.save_pretrained("./fine-tuned-spelling-correction")
tokenizer.save_pretrained("./fine-tuned-spelling-correction")



# Load the fine-tuned model and tokenizer
model = AutoModelForCausalLM.from_pretrained("./fine-tuned-spelling-correction")
tokenizer = AutoTokenizer.from_pretrained("./fine-tuned-spelling-correction")

# Test the model
def correct_spelling(text):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_length=64)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
misspelled_sentence = "I wnt to the stor to buy some breaad."
corrected_sentence = correct_spelling(misspelled_sentence)
print(f"Original: {misspelled_sentence}")
print(f"Corrected: {corrected_sentence}")


misspelled_sentence = "I red a book."
corrected_sentence = correct_spelling(misspelled_sentence)
print(f"Original: {misspelled_sentence}")
print(f"Corrected: {corrected_sentence}")