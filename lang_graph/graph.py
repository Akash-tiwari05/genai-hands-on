import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from typing import Literal
from pydantic import BaseModel


load_dotenv()

# 1. Initialize Client
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found")

client = genai.Client(api_key=api_key)

#Schema
class DetectCallResponse(BaseModel):
    is_coding_question: bool

class CodingAIResponse(BaseModel):
    answer: str

class State(TypedDict) :
    user_message: str
    ai_message: str
    is_coding_question: bool

def detect_query(state: State):
    user_message = state.get("user_message") 

    system_prompt = """
    You are an AI assistant. Your job is to detect if the user's query is related to coding question or not.

    Return the response in specified JSON boolean only.
    """

    #gemini call 
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=user_message,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
            response_schema=DetectCallResponse,
        ),
    )
    

    print(response.parsed)



    state["is_coding_question"] = response.parsed.is_coding_question
    return state

def route_edge(state: State) -> Literal["solve_coding_quetion","solve_simple_quetion"]:

    is_coding_question = state["is_coding_question"]

    if is_coding_question:
        return "solve_coding_quetion"
    else:
        return "solve_simple_quetion"

def solve_coding_quetion(state: State):
    user_message = state["user_message"]

    #gemini call( coding Question 3.0 pro)
    system_prompt = """
    You are an AI assistant. Your job is to resolve the user query based on the coding problem he facing.
    """

    #gemini call 
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=user_message,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
            response_schema=CodingAIResponse,
        ),
    )
    

    print(response.text)
    state["ai_message"] = response.parsed.answer 
    return state


def solve_simple_quetion(state: State):
    user_message = state["user_message"]

    #gemini call( coding Question 2.5 flash)
    system_prompt = """
    You are an AI assistant. Your job is to resolve the user query which are not coding problems.
    """

    #gemini call 
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=user_message,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
            response_schema=CodingAIResponse,
        ),
    )
    

    print(response.text)
    state["ai_message"] = response.parsed.answer 
    return state


graph_builder = StateGraph(State)

#Creating Nodes(Vertexes)
graph_builder.add_node("detect_query",detect_query)
graph_builder.add_node("solve_coding_quetion",solve_coding_quetion)
graph_builder.add_node("solve_simple_quetion",solve_simple_quetion)

#Conecting nodes with edges
graph_builder.add_edge(START, "detect_query")
graph_builder.add_conditional_edges("detect_query",route_edge)

graph_builder.add_edge("solve_coding_quetion",END)
graph_builder.add_edge("solve_simple_quetion",END)

graph = graph_builder.compile()

#use graph
def call_graph():
    state = {
        "user_message":"Can you explain pydentic in python?",
        "ai_message": "",
        "is_coding_question": False
    }
    result = graph.invoke(state)
    print("Final result", result)


call_graph()