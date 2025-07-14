from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings


loader = PyPDFLoader("../../docs/59321_booklet_guide_mashknta_A4_Pages_03.pdf",)
docs = loader.load()
print(f"docs loaded: {len(docs)}")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

# INDEXING: STORE
# vectorstore = Chroma.from_documents(
#     documents=all_splits,
#     # embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
#     embedding=OpenAIEmbeddings(),
#     persist_directory="./chroma_db",
#     collection_name="mortgage_docs"
#
# )

vectorstore = Chroma(
    embedding_function=OpenAIEmbeddings(),
    persist_directory="./chroma_db",
    collection_name="mortgage_docs"
)

# RETRIEVAL AND GENERATION: RETRIEVAL
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})


