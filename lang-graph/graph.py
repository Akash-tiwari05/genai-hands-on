import os
from dotenv import load_dotenv
from google import genai
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from typing import Literal


load_dotenv()

# 1. Initialize Client
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found")

client = genai.Client(api_key=api_key)

class State(TypedDict) :
    user_message: str
    ai_message: str
    is_coding_question: bool

def detect_query(state: State):
    user_message = state.get("user_message") 

    #gemini call 
    state["is_coding_question"] = True
    return state

def route_edge(state: State) -> Literal["solve_coding_quetion","solve_simple_quetion"]:

    is_coding_question = state.get("is_coding_question")

    if is_coding_question:
        return "solve_coding_quetion"
    else:
        return "solve_simple_quetion"

def solve_coding_quetion(state: State):
    user_message = state.get("user_message")

    #gemini call( coding Question 3.0 pro)
    state["ai_message"] = "Here is your coding question answer" 
    return state


def solve_simple_quetion(state: State):
    user_message = state.get("user_message")

    #gemini call( coding Question 2.5 flash)
    state["ai_message"] = "Please ask some coding related question" 
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
        "user_message":"Hey there! How are you?",
        "ai_message": "",
        "is_coding_question": False
    }
    result = graph.invoke(state)
    print("Final result", result)


call_graph()