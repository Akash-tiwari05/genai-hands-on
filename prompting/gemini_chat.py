from dotenv import load_dotenv
from google import genai
from google.genai import types
import os


load_dotenv()

# get API key
api_key = os.getenv("GEMINI_API_KEY")

# create client
client = genai.Client(api_key=api_key)

#Few shot prompting
system_prompt = """

You are an Ai Assistant who specialized in maths.You should not answer any query that is not related to maths.

For a given query help user to solve that along with explanation.

Example:
Input: 2 + 2
Output: 2 + 2 is 4 which is calculated by adding 2 + 2

Input: 3 * 10
Output: 3 * 10 is 30 which is calculated by multiplying by 3 by 10.

Input: Explain machine learning in simple words?
Output: Is it maths query?
"""

response = client.models.generate_content(
    model="models/gemini-2.5-flash",
    contents="What is genai",
    config=types.GenerateContentConfig(
        system_instruction=system_prompt,
    )
)

print(response.text)