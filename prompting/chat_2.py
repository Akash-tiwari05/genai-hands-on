import os
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables
load_dotenv()

# Client automatically uses os.getenv("GEMINI_API_KEY")
client = genai.Client()

# Define the schema for a single reasoning step
class ReasoningStep(BaseModel):
    step: str = Field(description="The current phase: 'analyse', 'think', 'output', 'validate', or 'result'")
    content: str = Field(description="The detailed content/thought process for this specific step.")

# Define the final container schema that holds all steps
class ProblemResolution(BaseModel):
    steps: list[ReasoningStep]

# Optimized system prompt focusing purely on the logic (the schema handles the format)
system_prompt = """
You are an AI assistant who is an expert in breaking down complex problems.

For the given user input, analyze the input and break down the problem step-by-step. 
You must think deeply, iterating through your thoughts before arriving at the final result.

You must follow these logical phases in sequence:
1. "analyse" - Understand what the user is asking.
2. "think" - Formulate the method, logic, or steps to solve it.
3. "output" - Generate the raw draft answer.
4. "validate" - Double-check your logic and draft answer for correctness.
5. "result" - Provide the final polished answer and explanation to the user.
"""

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="What is 3 + 4 * 5",
    config=types.GenerateContentConfig(
        system_instruction=system_prompt,
        # Enforce strict JSON output matching our Pydantic structure
        response_mime_type="application/json",
        response_schema=ProblemResolution,
        temperature=0.2 # Lower temperature makes reasoning more structured and deterministic
    )
)

# The output is guaranteed to be valid JSON matching your schema
print(response.text)