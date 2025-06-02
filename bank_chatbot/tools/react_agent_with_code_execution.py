from dotenv import load_dotenv
from langchain_experimental.tools import PythonAstREPLTool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

tool = PythonAstREPLTool()

tools = [
    tool,
]

memory = MemorySaver()
agent_executor = create_react_agent(
    llm,
    tools,
    checkpointer=memory
)