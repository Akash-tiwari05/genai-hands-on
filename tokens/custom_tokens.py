import string

class Tokenization:

    def __init__(self):
        # Create mapping for uppercase letters A-Z mapped to indices 0–25
        self.tokens = {letter: index for index, letter in enumerate(string.ascii_uppercase)}

        # Add mapping for lowercase letters a-z mapped to indices 26–51
        self.tokens.update({letter: index + 26 for index, letter in enumerate(string.ascii_lowercase)})

        # Create reverse mapping (index → character) for decoding tokens back to text
        self.reverse_tokens = {index: letter for letter, index in self.tokens.items()}

    def encoder(self, key):
        # Encode the first character of the input string into its corresponding token
        if key and key[0] in self.tokens:
            return self.tokens[key[0]]
        else:
            raise ValueError("Invalid Key...")

    def decoder(self, value):
        # Decode numeric token back into its corresponding character
        if value in self.reverse_tokens:
            return self.reverse_tokens.get(value)
        else:
            raise ValueError("Value is Invalid! Please re-check your input.")


# Create Tokenization object
tokens = Tokenization()

# Encode character 'a' into token
result = tokens.encoder("a")
print(result)

# Decode token back into character
getKey = tokens.decoder(28)
print(getKey)