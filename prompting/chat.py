from dotenv import load_dotenv
from openai import OpenAI
import os

# Load environment variables from .env file (used to securely store API keys)
load_dotenv()

client = OpenAI()

completion = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "developer", "content": "Talk like a pirate."}, #Zero sort prompting
        {
            "role": "user",
            "content": "How do I check if a Python object is an instance of a class?",
        },
    ],
)

print(completion.choices[0].message.content)