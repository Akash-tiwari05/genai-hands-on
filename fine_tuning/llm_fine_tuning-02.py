# Import os module to manage environment variables
import os

# Import tokenizer class for converting text into tokens
from transformers import AutoTokenizer

# Import causal language model loader
from transformers import AutoModelForCausalLM

# Import PyTorch library
import torch


# Set Hugging Face access token as environment variable
os.environ["HF_TOKEN"] = "your_huggingface_token"


# Define the model name from Hugging Face Hub
model_name = "google/gemma-3-1b-it"

# Load tokenizer for the selected model
tokenizer = AutoTokenizer.from_pretrained(model_name)

input_prompt = [
    "The capital of India is"
]

tokenized = tokenizer(input_prompt, return_tensors="pt", padding=True)

tokenized["input_ids"]

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype = torch.bfloat16
)

gen_result = model.generate(
    tokenized["input_ids"],
    max_new_tokens=25,
    max_length=None
)

print(gen_result)

output = tokenizer.batch_decode(gen_result)
print(output)