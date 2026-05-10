#Introductions of Tokens

import tiktoken

# Load tokenizer for the GPT-4o model
encoder = tiktoken.encoding_for_model('gpt-4o')

# Print the vocabulary size of the tokenizer
print("Vocab Size: ", encoder.n_vocab) #2,00,019 tokens in gpt-4o

text = "Python Is A High Level Programming Language"
tokens = encoder.encode(text)

print("Tokens: ",tokens) #Tokens:  [60502, 2763, 355, 6597, 16541, 65103, 20333]

my_tokens = [60502, 2763, 355, 6597, 16541, 65103, 20333]
decoded = encoder.decode(my_tokens)
print("Decode: ",decoded) #Python Is A High Level Programming Language

