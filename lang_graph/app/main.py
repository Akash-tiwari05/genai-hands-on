from graph_01 import create_chat_pointer 
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from pymongo import MongoClient
from langgraph.checkpoint.mongodb import MongoDBSaver


load_dotenv()

# 1. Establish your MongoDB Connection
client = MongoClient("mongodb://root:root@localhost:27017")

def main():

    # 2. Initialize the Checkpointer
    # It will manage 'checkpoints' and 'writes' collections in your database
    checkpointer = MongoDBSaver(client, db_name="langgraph_memory")

    # 3. FIX: Create the graph instance with MongoDB memory attached
    graph = create_chat_pointer(checkpointer)

    # 3. Define the configuration with a unique thread identifier
    # Every time you use this thread_id, MongoDB will load the previous messages.
    config = {"configurable": {"thread_id": "3"}}
    
    while True:
        user_input = input("> ")

        if user_input.lower() in ["exit", "quit"]:
            break

        state = None

        for state in graph.stream(
            {
                "messages": [HumanMessage(content=user_input)],
            },
            config=config,
            stream_mode="values",
        ):
            pass  # Wait until streaming completes to print the final block
        
        # 6. Clean print execution layout
        if state and "messages" in state:
            print("\nAI:", state["messages"][-1].content)
            print()



if __name__ == "__main__":
    main()
