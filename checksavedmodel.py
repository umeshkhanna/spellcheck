import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import logging
logging.basicConfig(
filename="step2.log",
encoding="utf-8",
filemode="a",
format="{asctime} - {levelname} - {message}",
style="{",
datefmt="%Y-%m-%d %H:%M",
)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = 'fine-tuned-spelling-correction'
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir="./model_cache")
tokenizer = AutoTokenizer.from_pretrained(model_name)


# Load the fine-tuned model and tokenizer
model = AutoModelForCausalLM.from_pretrained("./fine-tuned-spelling-correction")
tokenizer = AutoTokenizer.from_pretrained("./fine-tuned-spelling-correction")

logging.warning("model loaded")

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

logging.warning(corrected_sentence)
logging.warning("orig: {misspelled_sentence}")
logging.warning("orig: {corrected_sentence}")