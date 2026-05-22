# Import the pipeline function from the Hugging Face transformers library
from transformers import pipeline

# Initialize a text-generation pipeline using the pre-trained GPT-2 model
# This downloads the model weights and configures it for generating text
generator = pipeline("text-generation", model="gpt2")

# Generate text starting with the prompt "Artificial Intelligence is"
# The 'max_new_tokens' parameter restricts the output to a maximum of 30 new words/tokens
result = generator("Artificial Intelligence is", max_new_tokens=30)

# Extract and print the completed text from the generated output dictionary
# 'result[0]' accesses the first generated response sequence
print(result[0]["generated_text"])