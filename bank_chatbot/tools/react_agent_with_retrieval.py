from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.tools import tool, create_retriever_tool
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from bank_chatbot.rag.self_retriever_chroma import retriever

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# wrap retriever as a tool
retriever_tool = create_retriever_tool(
    retriever,
    name="search_movies",
    description="Search for informaiton about movies",
)
tools = [
    retriever_tool,
]

memory = MemorySaver()
agent_executor = create_react_agent(
    llm,
    tools,
    checkpointer=memory
)