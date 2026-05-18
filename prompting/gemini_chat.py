from dotenv import load_dotenv
from google import genai
from google.genai import types
import os


load_dotenv()

# get API key
api_key = os.getenv("GEMINI_API_KEY")

# create client
client = genai.Client(api_key=api_key)

response = client.models.generate_content(
    model="models/gemini-2.5-flash",
    contents="Explain machine learning in simple words"
)

print(response.text)