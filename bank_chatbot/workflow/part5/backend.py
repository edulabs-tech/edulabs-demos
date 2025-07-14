from typing import Literal

from dotenv import load_dotenv
from langchain import hub
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.prebuilt.chat_agent_executor import AgentState
from langchain_community.agent_toolkits import create_sql_agent, SQLDatabaseToolkit

from vectore_store import retriever

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver


checkpointer = MemorySaver()


class State(AgentState):
    intent: str


llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")


# Nodes
def intent_detector_node(state: State):
    """
    Classifies the user's intent as 'ACCOUNT', 'MORTGAGE', or 'OTHER'.
    This node updates the 'intent' field in the state.
    """

    # We use the last message in the state for classification.
    user_question = state["messages"][-1].content

    system_intent_template = """
        You are a friendly customer assistant at Bank Hapoalim.
        Your task is to detect the intent of the customer's question - one of 3 options:
        1. Account-related questions (ACCOUNT)
        2. Mortgage-related questions (MORTGAGE)
        3. Other questions (OTHER)
        You should return ONLY one of : ACCOUNT, MORTGAGE, OTHER
        """
    intent_prompt = ChatPromptTemplate.from_messages([
        ("system", system_intent_template),
        ("user", "{question}")
    ])

    intent_chain = intent_prompt | llm | StrOutputParser()
    intent = intent_chain.invoke({"question": user_question}).strip()
    print(f"Detected Intent: {intent}")
    return {"intent": intent}


# DB Agent Creation: Create a full SQL agent that can reason about and query the database.
# This agent is a self-contained unit that we will call from a single node.
db = SQLDatabase.from_uri("sqlite:///../../docs/demo.db")
sql_toolkit = SQLDatabaseToolkit(db=db, llm=llm)
db_agent = create_sql_agent(
    llm=llm,
    toolkit=sql_toolkit,
    verbose=True,
    agent_type="openai-tools",
    handle_parsing_errors=True,
)
def account_node(state: State):
    """
    Invokes the pre-built SQL Agent to answer database-related questions.
    """

    user_question = state["messages"][-1].content
    # The agent's response is a dictionary, we extract the 'output'
    response = db_agent.invoke({"input": user_question})
    return {"messages": [AIMessage(content=response["output"])]}


# Retriever agent


def mortgage_node(state: State):
    """
    Performs Retrieval-Augmented Generation (RAG) for mortgage questions.
    """
    user_question = state["messages"][-1].content
    rag_prompt = hub.pull("rlm/rag-prompt")

    def format_docs(original_docs):
        return "\n\n".join(doc.page_content for doc in original_docs)

    # The RAG chain retrieves documents, formats them, and passes them to the LLM.
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    response = rag_chain.invoke(user_question)
    return {"messages": [AIMessage(content=response)]}

# Edges
def intent_selection_edge(state: State) -> Literal["account_node", "mortgage_node", "__end__"]:
    if state["intent"] == "ACCOUNT":
        return "account_node"
    elif state["intent"] == "MORTGAGE":
        return "mortgage_node"
    else:
        return "__end__"

# Graph
graph_builder = StateGraph(State)

graph_builder.add_node("intent_detector_node", intent_detector_node)
graph_builder.add_node("account_node", account_node)
graph_builder.add_node("mortgage_node", mortgage_node)

graph_builder.add_edge(START, "intent_detector_node")
graph_builder.add_conditional_edges("intent_detector_node", intent_selection_edge)
graph_builder.add_edge("account_node", END)
graph_builder.add_edge("mortgage_node", END)

graph = graph_builder.compile(checkpointer=checkpointer)

def save_graph_png():
    graph.get_graph().draw_mermaid_png(output_file_path="graph.png")


if __name__ == '__main__':
    save_graph_png()