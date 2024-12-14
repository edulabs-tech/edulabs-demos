from pprint import pprint

from dotenv import load_dotenv
from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.messages import SystemMessage, HumanMessage

from bank_chatbot.tools.income_tax_tool import calculate_income_tax

load_dotenv()

# Add memory to the process
memory = MemorySaver()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# Construct retriever
# loader = PyPDFLoader("../docs/59321_booklet_guide_mashknta_A4_Pages_03.pdf",)
# docs = loader.load()
#
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# splits = text_splitter.split_documents(docs)
# vectorstore = InMemoryVectorStore.from_documents(
#     documents=splits, embedding=OpenAIEmbeddings()
# )
# retriever = vectorstore.as_retriever()


# Build tool
# retriever_tool = create_retriever_tool(
#     retriever,
#     "Mortgage-Booklet-Retriever",
#     "Searches and returns excerpts from the mortgage guide booklet",
# )


# db = SQLDatabase.from_uri("sqlite:///../docs/demo.db")
# print(db.dialect)
# print(db.get_usable_table_names())


# toolkit = SQLDatabaseToolkit(db=db, llm=llm)
# db_tools = toolkit.get_tools()
# pprint(db_tools)


SQL_PREFIX = """You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct SQLite query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 5 results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
Only use the below tools. Only use the information returned by the below tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

To start you should ALWAYS look at the tables in the database to see what you can query.
Do NOT skip this step.
Then you should query the schema of the most relevant tables."""

system_message = SystemMessage(content=SQL_PREFIX)

tools = [
    calculate_income_tax,
    # retriever_tool,
    # *db_tools
]
agent_executor = create_react_agent(
    llm,
    tools,
    # checkpointer=memory,
    messages_modifier=system_message
)




# More tools:
# https://python.langchain.com/docs/integrations/tools/
