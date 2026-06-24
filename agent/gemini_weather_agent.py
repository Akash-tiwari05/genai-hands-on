from dotenv import load_dotenv
from google import genai
from langsmith import wrappers
from langsmith import traceable
import json
import os

# Load environment variables from .env
load_dotenv()

# Instantiate the regular Google Gen AI client
raw_client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY")
)

# Wrap the client to intercept operations and log them to LangSmith
client = wrappers.wrap_gemini(
    raw_client,
    tracing_extra={
        "tags": ["gemini", "python"],
        "metadata": {
            "integration": "google-genai",
        },
    },
)



@traceable
def get_weather(city: str):
    # TODO: Replace with actual weather API
    return "31 degree celsius"


system_prompt = """
You are a helpful AI Assistant specialized in resolving user queries.
You work in start, plan, action, observe mode.

For the given user query and available tools:
- Plan the step-by-step execution
- Select the relevant tool
- Perform an action
- Wait for observation
- Resolve the user query

Rules:
- Follow the Output JSON Format
- Always perform one step at a time
- Carefully analyze the user query

Output JSON Format:
{
    "step": "string",
    "content": "string",
    "function": "function name if step is action",
    "input": "input for function"
}

Available Tools:
- get_weather(city)

Example:
User Query: What is the weather of New York?

Output:
{ "step": "plan", "content": "The user is interested in weather data of New York" }

Output:
{ "step": "plan", "content": "I should call get_weather" }

Output:
{ "step": "action", "function": "get_weather", "input": "New York" }

Output:
{ "step": "observe", "output": "12 Degree Cel" }

Output:
{ "step": "output", "content": "The weather for New York seems to be 12 degrees." }
"""

user_query = "What is the current weather of Agra?"

# Generate response from Gemini
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=f"{system_prompt}\n\nUser Query: {user_query}",
)

# Print Gemini response
print(response.text)