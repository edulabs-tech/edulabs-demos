from pprint import pprint

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

open_ai_model = ChatOpenAI(model="gpt-4o-mini")
response = open_ai_model.invoke("Hi")
pprint(response.dict())