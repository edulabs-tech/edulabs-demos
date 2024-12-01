from dotenv import load_dotenv
from langchain_community.tools import TavilySearchResults
import pprint

load_dotenv()

tool = TavilySearchResults(max_results=2)
tools = [tool]
pprint.pprint(tool.invoke("What's a 'node' in LangGraph?"))