import os
from mem0 import Memory
from google import genai

# Set your Google AI Studio API key
os.environ["GOOGLE_API_KEY"] = "your_gemini_api_key_here"

# 1. Configure Mem0 to use Gemini for both LLM processing and embeddings
config = {
    "llm": {
        "provider": "google_ai",
        "config": {
            "model": "gemini-2.0-flash-001",
            "temperature": 0.1
        }
    },
    "embedder": {
        "provider": "google_ai",
        "config": {
            "model": "text-embedding-004"
        }
    }
}

# Initialize the Mem0 Memory instance
memory = Memory.from_config(config)

# 2. Add interactions to memory for a specific user
user_id = "user_alex_99"
chat_interaction = "Hi! I am Alex. I am a vegan and I am currently learning Python backend development."

# Mem0 extracts salient facts and stores them automatically
memory.add(chat_interaction, user_id=user_id)

# 3. Simulate a completely new session / retrieve context later
query = "What should I cook for dinner tonight that won't distract me from studying?"

# Retrieve only the facts relevant to the query
relevant_memories = memory.search(query, user_id=user_id)

# Flatten retrieved memories into a clean string context
context = "\n".join([mem["memory"] for mem in relevant_memories])

# 4. Generate a personalized response using Google Gemini
client = genai.Client()
prompt = f"""
You are a helpful AI assistant. 
Use the following historical context about the user if relevant to answer their query.

[User Context]
{context}

[User Query]
{query}
"""

response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents=prompt,
)

print(response.text)