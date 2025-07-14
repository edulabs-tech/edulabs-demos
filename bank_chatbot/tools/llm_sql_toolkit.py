from pprint import pprint

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI
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


db = SQLDatabase.from_uri("sqlite:///../docs/demo.db")
# print(db.dialect)
# print(db.get_usable_table_names())


toolkit = SQLDatabaseToolkit(db=db, llm=llm)
db_tools = toolkit.get_tools()
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


llm_with_tools = llm.bind_tools(db_tools)

response = llm_with_tools.invoke([
        ("system", SQL_PREFIX),
        ("user", "What's my balance?")
])

print(response)