from typing import Annotated, Literal

from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict

from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from langgraph.checkpoint.memory import MemorySaver

from dotenv import load_dotenv


load_dotenv()


@tool
def get_account_data():
    """Provides account data"""
    pass
    # return "will need to get your account data first"


@tool
def get_general_info():
    """Provides answers to general questions"""
    pass


class State(TypedDict):
    messages: Annotated[list, add_messages]
    account_id: str


graph_builder = StateGraph(State)


tools = [get_account_data, get_general_info]
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_with_intention_tools = llm.bind_tools(tools)

memory = MemorySaver()

system_intention_template = """
    You are a friendly customer assistant at Bank Hapoalim.
    You are going to great a customer and assist him with his questions
"""
prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_intention_template),
    MessagesPlaceholder(variable_name="messages_placeholder"),
])

intention_chain = prompt_template | llm_with_intention_tools

#######################

db = SQLDatabase.from_uri("sqlite:///../../docs/demo.db")
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
db_tools = toolkit.get_tools()
llm_with_account_actions_tools = llm.bind_tools(db_tools)
system_account_actions_template = """
    You are a friendly customer assistant at Bank Hapoalim.
    You are answering queries about account_id {account_id}.
"""
account_actions_prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_account_actions_template),
    MessagesPlaceholder(variable_name="messages_placeholder"),
])

account_actions_chain = account_actions_prompt_template | llm_with_account_actions_tools


def intention_detector(state: State):
    result = intention_chain.invoke({"messages_placeholder": state["messages"]})
    # result = llm.invoke(state["messages"])
    return {"messages": [result]}


def chatbot(state: State):
    return {"messages": [llm_with_intention_tools.invoke(state["messages"])]}


def account_actions(state: State):
    return {"messages": [account_actions_chain.invoke({
        "messages_placeholder": state["messages"],
        "account_id": state["account_id"]
    })]}


def identify_account(state: State):
    account_id = state.get("account_id")
    if not account_id:
        account_id = input("Insert your account: ")
        secret_code = input("Insert secret code: ")
    return {"account_id": account_id}


def intent_condition(state: State) -> Literal["account_data", "general_data", "__end__"]:
    if isinstance(state, list):
        ai_message = state[-1]
    elif isinstance(state, dict) and (messages := state.get("messages", [])):
        ai_message = messages[-1]
    elif messages := getattr(state, "messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "account_data"
    elif len(ai_message.tool_calls) == 2:
        return "general_data"
    return "__end__"


graph_builder.add_node("intention_detector", intention_detector)
graph_builder.add_node("account_data", ToolNode([get_account_data]))
graph_builder.add_node("general_data", ToolNode([get_general_info]))
graph_builder.add_node("identify_account", identify_account)
graph_builder.add_node("account_actions", account_actions)
graph_builder.add_node("tools", ToolNode(tools=db_tools))
graph_builder.add_edge(START, "intention_detector")
graph_builder.add_conditional_edges(
    "intention_detector",
    intent_condition,
)
graph_builder.add_edge("account_data", "identify_account")
graph_builder.add_edge("identify_account", "account_actions")
graph_builder.add_conditional_edges(
    "account_actions",
    tools_condition,
    path_map={"tools": "tools", END: END}
)
graph_builder.add_edge("tools", "account_actions")
graph_builder.add_edge("account_actions", END)
graph_builder.add_edge("general_data", END)


graph = graph_builder.compile(checkpointer=memory)


def stream_graph_updates(user_input: str, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    events = graph.stream(
        {"messages": [("user", user_input)]}, config, stream_mode="values"
    )
    for event in events:
        event["messages"][-1].pretty_print()


def save_graph_png():
    graph.get_graph().draw_mermaid_png(output_file_path="graph.png")


if __name__ == '__main__':
    save_graph_png()
