from dotenv import load_dotenv
from pymongo import MongoClient

from graph_01 import create_chat_pointer
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.types import Command

load_dotenv()

client = MongoClient("mongodb://root:root@localhost:27017")

config = {
    "configurable": {
        "thread_id": "3"
    }
}


def main():
    checkpointer = MongoDBSaver(
        client,
        db_name="langgraph_memory"
    )

    graph = create_chat_pointer(checkpointer)

    state = graph.get_state(config)

    if not state.values:
        print("No checkpoint found.")
        return

    messages = state.values.get("messages", [])

    if not messages:
        print("No messages found.")
        return

    print("=" * 80)
    print("Conversation History")
    print("=" * 80)

    for msg in messages:
        msg.pretty_print()

    print("=" * 80)

    last_message = messages[-1]

    print("\nLast Message:")
    last_message.pretty_print()

    resolution = input("\nResolution > ")

    # Use a dict if interrupt() expects structured data
    resume_command = Command(
        resume={"data": resolution}
    )

    print("\nResuming graph...\n")

    for event in graph.stream(
        resume_command,
        config=config,
        stream_mode="values"
    ):
        if "messages" in event:
            event["messages"][-1].pretty_print()
        else:
            print(event)


if __name__ == "__main__":
    main()