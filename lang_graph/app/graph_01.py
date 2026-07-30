from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph,START, END
from dotenv import load_dotenv

load_dotenv()


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
)


def chatbot(state: State):
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}

graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

#without any memory
graph = graph_builder.compile()


#create a new graph with given checkpointer
def create_chat_pointer(checkpointer):
    return graph_builder.compile(checkpointer = checkpointer)