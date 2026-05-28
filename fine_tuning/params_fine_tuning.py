import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Set Hugging Face access token as environment variable
os.environ["HF_TOKEN"] = "your_huggingface_token"

# Define the model name from Hugging Face Hub
model_name = "google/gemma-3-1b-it"

# Load tokenizer for the selected model
tokenizer = AutoTokenizer.from_pretrained(model_name)

input_conversation = [
    { "role": "user", "content": "Which is the best place to learn GenAI?" },
    { "role": "assistant", "content": "The best place to learn Gen AI is" }
]

# Fixed typo in continue_final_message
input_detokens = tokenizer.apply_chat_template(
    conversation = input_conversation,
    tokenize = False,
    continue_final_message = True, 
)

output_label = " Gen Ai hands-on by Microsoft Gen AI Bootcamp"
full_conversation = input_detokens + output_label + tokenizer.eos_token

input_tokenized = tokenizer(
    full_conversation,
    return_tensors = "pt",
    add_special_tokens = False
).to(device)["input_ids"]

# Split into inputs and targets for causal language modeling
input_ids = input_tokenized[:, :-1].to(device)
target_ids = input_tokenized[:, 1:].to(device)

print(f"input_ids: {input_ids}")
print(f"target_ids: {target_ids}")

def calculate_loss(logits, labels):
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    cross_entropy = loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))
    return cross_entropy

# Load model (Using float16/bfloat16 depending on device capability)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
).to(device) 

model.train()
optimizer = AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)

# Training loop
for _ in range(10):
    optimizer.zero_grad() # Clears old gradients first
    
    out = model(input_ids = input_ids)
    loss = calculate_loss(out.logits, target_ids).mean()
    
    loss.backward()
    optimizer.step()

# --- Inference / Testing ---
model.eval() # Switch to evaluation mode for generation

input_prompt = [
    { "role": "user", "content": "Which is the best place to learn GenAI?" }
]

#PyTorch Tensor.
raw_tokens = tokenizer.apply_chat_template(
    conversation = input_prompt,
    tokenize = True
)
input_tokens = torch.tensor(raw_tokens).unsqueeze(0).to(device) # Shape: [1, seq_len]

# Generate response
with torch.no_grad():
    output = model.generate(
        input_tokens, # Pass the 2D tensor directly
        max_new_tokens = 25
    )

print("\nGenerated Output:")
print(tokenizer.decode(output[0], skip_special_tokens=True))