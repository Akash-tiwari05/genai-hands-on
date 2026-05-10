#Introductions of Vector Embedding

from dotenv import load_dotenv
from openai import OpenAI
import os

# Load environment variables from .env file (used to securely store API keys)
load_dotenv()

# Initialize OpenAI client using the API key from environment variables
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Input text that we want to convert into a numerical vector (embedding)
text = "The Taj Mahal is an ivory-white marble mausoleum in Agra, India, commissioned in 1632"

# Call OpenAI Embeddings API to convert text into a high-dimensional vector representation
response = client.embeddings.create(
    input=text,
    model="text-embedding-3-small"
)

# Extract and print the embedding vector from the API response
print("Vector Embeddings:", response.data[0].embedding)