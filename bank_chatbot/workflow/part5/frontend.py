from langchain_core.messages import HumanMessage

from backend import graph


def run_agent(question: str, thread_id: str):
    """Helper function to run the agent and print the conversation flow."""
    print(f"\n--- Running Agent for: '{question}' ---\n")
    inputs = {"messages": [HumanMessage(content=question)]}

    result = graph.invoke(inputs, {"configurable":{"thread_id": thread_id}, "recursion_limit": 10})
    return result["messages"][-1].content

    # The stream() method lets us see the output of each node as it's executed
    # for output in graph.stream(inputs, {"configurable":{"thread_id": thread_id}, "recursion_limit": 10}):
    #     for key, value in output.items():
    #         print(f"Output from node '{key}':")
    #         print("---")
    #         print(value)
    #     print("\n---\n")

# run_agent("Hi, my name is Bob. Can you tell me my account balance?", "aaa")
# run_agent("What is the latest transaction in my account?", "aaa")
# run_agent("How much mortgage can I get if i want to buy an apartment for 3000000 ILS?", "aaa")
# run_agent("Can you tell me a fun fact about space?", "aaa")