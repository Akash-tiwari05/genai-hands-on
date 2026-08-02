from typing import Annotated
from typing_extensions import TypedDict

from dotenv import load_dotenv

from langchain_core.messages import AnyMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt

load_dotenv()


@tool
def human_assistance_tool(query: str):
    """Request assistance from a human."""

    human_response = interrupt(
        {
            "query": query
        }
    )

    return human_response["data"]


model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
)

model = model.bind_tools([human_assistance_tool])


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def chatbot(state: State):
    response = model.invoke(state["messages"])

    return {
        "messages": [response]
    }


builder = StateGraph(State)

builder.add_node("chatbot", chatbot)
builder.add_node("tools", ToolNode([human_assistance_tool]))

builder.add_edge(START, "chatbot")

builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)

builder.add_edge("tools", "chatbot")

graph = builder.compile()


def create_chat_pointer(checkpointer):
    return builder.compile(checkpointer=checkpointer)