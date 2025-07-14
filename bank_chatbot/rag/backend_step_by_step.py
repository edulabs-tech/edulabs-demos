from pprint import pprint
import numpy as np
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# Load environment variables from .env file
load_dotenv()

# INDEXING: LOAD
# A Document is an object with some page_content (str) and metadata (dict)

# There are 160+ integrations to choose from
# https://python.langchain.com/docs/integrations/document_loaders/

loader = PyPDFLoader("../docs/59321_booklet_guide_mashknta_A4_Pages_03.pdf",)
docs = loader.load()
#
print(f"Total docs: {len(docs)}")
print(f"Example doc metadata: {docs[0].metadata}")
print(f"Example snippet of doc content: {docs[5].page_content[:200]}")
print(f'Total characters in all docs: {sum([len(doc.page_content) for doc in docs])}')


# INDEXING: SPLIT
# Other document transformers:
# https://python.langchain.com/docs/integrations/document_transformers/

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000, chunk_overlap=200, add_start_index=True
# )
# all_splits = text_splitter.split_documents(docs)
# print(f"Splits number: {len(all_splits)}")
# print(f"Example split content: {all_splits[27].page_content}")
# print(f"Example split metadata: {all_splits[27].metadata}")

# Embedding model
# embedding_model = OpenAIEmbeddings()
# embedding_model=GoogleGenerativeAIEmbeddings(model="models/embedding-001")


# INDEXING: STORE
# vectorstore = Chroma.from_documents(
#     documents=all_splits,
#     embedding=embedding_model
# )


# Cosine Distance (used by Chroma):
# distance=1−cosine_similarity
# 0 = identical
# Closer to 1 = less similar

# example_text = "How can I contact the bank?"
# example_text = "Where do I buy flowers?"
# results = vectorstore.similarity_search_with_score(example_text, k=4)
# pprint(results)


# RETRIEVAL AND GENERATION: RETRIEVAL
# Create a simple application that takes a user question,
# searches for documents relevant to that question,
# passes the retrieved documents and initial question to a model, and returns an answer

# The most common type of Retriever is the VectorStoreRetriever,
# which uses the similarity search capabilities of a vector store to facilitate retrieval.
# limit the number of documents k returned by the retriever to 6

# retriever = vectorstore.as_retriever(
#     search_type="similarity", search_kwargs={"k": 6})
# retrieved_docs = retriever.invoke(example_text)

# pprint(retrieved_docs)

# RETRIEVAL AND GENERATION: GENERATE
# Let’s put it all together into a chain that takes a question,
# retrieves relevant documents, constructs a prompt,
# passes it into a model, and parses the output.
# open_ai_model = ChatOpenAI(model="gpt-4o-mini")

# Using prompt from the prompt hub:
# https://smith.langchain.com/hub/rlm/rag-prompt


# prompt = hub.pull("rlm/rag-prompt")
# #
# def format_docs(original_docs):
#     return "\n\n".join(doc.page_content for doc in original_docs)
#
# rag_chain = (
#     {"context": retriever | format_docs, "question": RunnablePassthrough()}
#     | prompt
#     | open_ai_model
#     | StrOutputParser()
# )
# #
# for chunk in rag_chain.stream(example_text):
#     print(chunk, end="", flush=True)
