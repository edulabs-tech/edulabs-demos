from pprint import pprint

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

load_dotenv()

# model = ChatOpenAI(model="gpt-4o-mini")
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")


response = model.invoke("Hi")
response.pretty_print()