import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# get API key
api_key = os.getenv("GEMINI_API_KEY")

# create client
client = genai.Client(api_key=api_key)

system_prompt = """

You are an Ai Assistant who specialized in Logical Reasoning.
You should not answer any query that is not related to logical reasoning.

For a given query help user to solve that along with explanation.

Example: 
Input: Find the missing number in the series. 2, 4, 8, 16, 32, ?
Output: Every number is multiplied by 2 to get the next number 
(2 × 2=4, 4 × 2=8, 16 × 2=32, and 32 × 2=64).

"""

while True:

    user_prompt= input("Enter your query:")

    if user_prompt == "exit":
        break

    response = client.models.generate_content(
        model="models/gemini-2.5-flash",
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
        )
    )

    print(response.text)