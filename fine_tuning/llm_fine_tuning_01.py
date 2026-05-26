# Import os module to manage environment variables
import os

# Import tokenizer class for converting text into tokens
from transformers import AutoTokenizer

# Import causal language model loader
from transformers import AutoModelForCausalLM

# Import PyTorch library
import torch

# Import Hugging Face pipeline utility
from transformers import pipeline


# Set Hugging Face access token as environment variable
os.environ["HF_TOKEN"] = "your_huggingface_token"


# Define the model name from Hugging Face Hub
model_name = "google/gemma-3-1b-it"


# Load tokenizer for the selected model
tokenizer = AutoTokenizer.from_pretrained(model_name)


# Tokenize sample text and print tokenizer output
print(tokenizer("Hello, how are you?"))


# Print complete vocabulary of tokenizer
print(tokenizer.get_vocab())


# Convert text into input token IDs only
input_tokens = tokenizer("Hello, how are you?")["input_ids"]

# Print token IDs
print(input_tokens)


# Load pretrained causal language model
model = AutoModelForCausalLM.from_pretrained(
    model_name,

    # Use bfloat16 datatype for optimized memory usage
    dtype=torch.bfloat16
)


# Create text generation pipeline
gen_pipeline = pipeline(
    "text-generation",   # Specify pipeline task
    model=model,         # Loaded language model
    tokenizer=tokenizer  # Corresponding tokenizer
)


# Generate text response from prompt
gen_pipeline(
    "Hey there",         # Input prompt

    # Number of new tokens to generate
    max_new_tokens=25,

    # No fixed maximum total sequence length
    max_length=None
)