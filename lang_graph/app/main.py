from graph_01 import graph
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

def main():
    while True:
        user_input = input("> ")

        if user_input.lower() in ["exit", "quit"]:
            break


        for state in graph.stream(
            {
                "messages": [HumanMessage(content=user_input)],
            },
            stream_mode="values",
        ):
            print("\nAI:", state["messages"][-1].content)
            print()




if __name__ == "__main__":
    main()
