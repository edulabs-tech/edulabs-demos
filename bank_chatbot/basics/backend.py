from typing import List

from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.messages import trim_messages, BaseMessage


# Load environment variables from .env file
load_dotenv()

system_template = """
    You are a friendly customer assistant at Bank Hapoalim.
    Your name is {assistant_name}.
    All your answers should be in {language}.
    Be friendly, greet the customer, present yourself, and answer their questions related to bank operations.
"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_template),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{text}")
])

open_ai_model = ChatOpenAI(model="gpt-4o-mini")
gemini_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
# llama

parser = StrOutputParser()

open_ai_chain = prompt_template | open_ai_model | parser
gemini_chain = prompt_template | gemini_model | parser

gemini_trimmer = trim_messages(
    max_tokens=100,
    strategy="last",
    token_counter=gemini_model,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

open_ai_trimmer = trim_messages(
    max_tokens=100,
    strategy="last",
    token_counter=open_ai_model,
    include_system=True,
    allow_partial=False,
    start_on="human",
)


def pretty_print_messages(messages: List[BaseMessage]):
    for msg in messages:
        msg.pretty_print()


def invoke_with_trim(text, history, language, model_type="Gemini"):
    chain = open_ai_chain if model_type == "Open AI" else gemini_chain

    print(f"Messages BEFORE trimming ({len(history)}):", history, sep='\n')
    trimmer = open_ai_trimmer if model_type == "Open AI" else gemini_trimmer
    history = trimmer.invoke(history)
    print(f"Messages AFTER trimming ({len(history)}):", history, sep='\n')

    return chain.invoke({"assistant_name": "David", "language": language, "text": text, "history": history})


def invoke_llm(text, history, language, model_type="Gemini"):
    chain = open_ai_chain if model_type == "Open AI" else gemini_chain
    return chain.invoke({"assistant_name": "David",
                         "language": language,
                         "text": text,
                         "history": history})


def stream_llm(text, history, language, model_type="Gemini"):
    chain = open_ai_chain if model_type == "Open AI" else gemini_chain
    full_msg = ""
    for chunk in chain.stream(
            {"assistant_name": "David", "language": language, "text": text, "history": history},
            stream_mode="messages"):
        full_msg = full_msg + chunk
        yield full_msg




