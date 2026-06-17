import os
from dotenv import load_dotenv
from mem0 import Memory
from google import genai
from google.genai import types


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)

QDRANT_HOST = "localhost"

NEO4J_URL = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "reform-william-center-vibrate-press-5829"

config = {
    "version": "v1.1",
    "embedder": {
        "provider": "gemini",
        "config": {
            "api_key": api_key,
            "model": "gemini-embedding-001",
            "output_dimensionality": 1536,
        },
    },
    "llm": {
        "provider": "gemini",
        "config": {
            "api_key": api_key,
            "model": "gemini-2.5-flash",
        },
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "host": QDRANT_HOST,
            "port": 6333,
            "collection_name": "mem0_gemini_1536",
        },
    },
    "graph_store": {
        "provider": "neo4j",
        "config": {
            "url": NEO4J_URL,
            "username": NEO4J_USERNAME,
            "password": NEO4J_PASSWORD,
        },
    },
}

mem_client = Memory.from_config(config)


def chat(message):
    memories = mem_client.search(
        query=message,
        filters={"user_id": "user1"}
    )

    memory_context = "\n".join(
        [m["memory"] for m in memories.get("results", [])]
    )

    prompt = f"""
    Relevant memories:
    {memory_context}

    User:
    {message}
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    mem_client.add(
        message,
        user_id="user1"
    )

    return response.text


while True:
    message = input(">> ")
    print("BOT: ", chat(message=message))
