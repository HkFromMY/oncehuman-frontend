from langchain.prompts import PromptTemplate 
from dotenv import load_dotenv, find_dotenv 
import os 

load_dotenv(find_dotenv())

LLM_MODEL = 'llama-3.1-70b-versatile'

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

PINECONE_INDEX_NAME = os.environ.get('PINECONE_INDEX_NAME')

EMBEDDING_MODEL = 'sentence-transformers/all-mpnet-base-v2'

HISTORY_AWARE_PROMPT = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

RAG_SYSTEM_PROMPT = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\

Context: {context}"""
