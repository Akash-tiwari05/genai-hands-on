import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
import json
from pydantic import BaseModel, Field

load_dotenv()

# 1. Initialize Client
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found")

client = genai.Client(api_key=api_key)

# 2. Define the Strict Chain-of-Thought Schema
class MedicalCoT(BaseModel):
    analyse: str = Field(description="Deconstruct the core clinical question.")
    think: str = Field(description="Evaluate underlying pathophysiology, drug mechanisms, or medical guidelines.")
    output: str = Field(description="Draft the direct medical answer.")
    validate: str = Field(description="Double-check the answer against established clinical guidelines and safety guardrails.")
    result: str = Field(description="Provide the final explanation combined with the mandatory medical disclaimer.")

# 3. Prompts Setup
system_prompt = """
You are a highly capable, evidence-based Medical AI Assistant assisting medical professionals.
Base answers on current, peer-reviewed medical consensus. Provide sources when possible. 
Guardrails: Never diagnose or prescribe. Always include a clinical verification disclaimer.

You must solve the user's query by executing a strict Chain-of-Thought reasoning loop. 
Fill out every single field required by the response schema sequentially.

### IMPORTANT NOTES ON STYLE:
- Keep all explanations short, punchy, and direct.
- Avoid unnecessary medical jargon where simple terms work just as well. 
- Use brief bullet points or short sentences instead of long paragraphs.
- Ensure the final "result" is highly scannable for a busy medical professional.
"""

# 4. Core LLM Function
def query_medical_agent(user_query: str):
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=user_query,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            # Force the model to strictly follow the CoT structure
            response_mime_type="application/json",
            response_schema=MedicalCoT,
            temperature=0.2, 
        )
    )
    
    # The response.text is guaranteed to match the MedicalCoT schema
    return json.loads(response.text)

# 5. Main Execution Loop
if __name__ == "__main__":
    user_prompt = input("Ask a medical question: -> ")
    
    print("\nProcessing clinical reasoning...\n")
    result = query_medical_agent(user_prompt)
    
    # Print out the sequential steps beautifully
    print(f"🧠 [1. Analyse]: {result.get('analyse')}\n")
    print(f"🤖 [2. Think]: {result.get('think')}\n")
    print(f"📝 [3. Output]: {result.get('output')}\n")
    print(f"🛡️ [4. Validate]: {result.get('validate')}\n")
    print(f"✨ [5. Result]:\n{result.get('result')}")