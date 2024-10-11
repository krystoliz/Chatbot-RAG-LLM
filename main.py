import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.vectorstores import Pinecone as PineconeStore

load_dotenv()

loader = PyPDFDirectoryLoader("health_knowledge_base.txt")
document =loader.load()

text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=100)
texts = text_splitter.split_documents(document)
print(f"created {len(texts)} chunks ")

embeddings = OpenAIEmbeddings(openai_api_type=os.environ.get("OPENAI_API_KEY"))
PineconeStore.from_documents(texts, embeddings, index_name=os.environ.get("INDEX_NAME"))

