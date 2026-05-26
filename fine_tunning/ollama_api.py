# Import FastAPI framework
from fastapi import FastAPI, Body

# Import Ollama client to interact with local LLM
from ollama import Client

# Create FastAPI application instance
app = FastAPI()

# Initialize Ollama client with local server URL
client = Client(
    host='http://localhost:11434'
)

# Create POST endpoint "/chat"
@app.post("/chat")
def chat(message: str = Body(..., description="Chat Message")):
    
    # Send user message to Ollama model
    response = client.chat(
        model="phi3:mini",   # Specify the model name
        messages=[
            {
                "role": "user",      # Role of sender
                "content": message   # Actual user message
            }
        ]
    )

    # Return only the generated response content
    return response['message']['content']