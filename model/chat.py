from utils.constant import (
    LLM_MODEL,
    HISTORY_AWARE_PROMPT,
    RAG_SYSTEM_PROMPT,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    EMBEDDING_MODEL,
)
from langchain_groq import ChatGroq 
from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface.embeddings.huggingface_endpoint import HuggingFaceEndpointEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

import time 
import streamlit as st

def get_session_history(session_id):
    store = st.session_state.chat_history
    if session_id not in store:
        store[session_id] = ChatMessageHistory()

    return store[session_id]

@st.cache_resource
def load_model():
    model = ChatGroq(
        model=LLM_MODEL,
        temperature=0.5,
        max_tokens=4096,
        timeout=None, 
        max_retries=3,
    )

    return model

@st.cache_resource
def create_pinecone_index():
    if not PINECONE_API_KEY:
        raise ValueError("Please set the PINECONE_API_KEY environment variable")
    
    pc = Pinecone(api_key=PINECONE_API_KEY)

    existing_indexes = [index_info['name'] for index_info in pc.list_indexes()]

    # must ensure that the index name given exists, and have documents loaded
    if PINECONE_INDEX_NAME not in existing_indexes:
        raise ValueError("Please create the Pinecone index")
    
    index = pc.Index(PINECONE_INDEX_NAME)

    return index 

@st.cache_resource
def create_embedding():
    return HuggingFaceEndpointEmbeddings(
        model=EMBEDDING_MODEL,
    )

@st.cache_resource
def create_retriever():
    index = create_pinecone_index()
    embedding = create_embedding()

    vector_store = PineconeVectorStore(index=index, embedding=embedding)
    retriever = vector_store.as_retriever(
        search_type='mmr',
        search_kwargs={
            'k': 5,
            'fetch_k': 25,
            'lambda_mult': 0.5
        }
    )

    return retriever 

@st.cache_resource
def create_rag_chain():
    llm = load_model()
    retriever = create_retriever()

    history_aware_template = ChatPromptTemplate.from_messages([
        ('system', HISTORY_AWARE_PROMPT),
        MessagesPlaceholder('chat_history'),
        ('human', '{input}')
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm=llm, 
        retriever=retriever,
        prompt=history_aware_template,
    )

    rag_prompt_template = ChatPromptTemplate.from_messages([
        ('system', RAG_SYSTEM_PROMPT),
        MessagesPlaceholder('chat_history'),
        ('human', '{input}')
    ])

    qa_chain = create_stuff_documents_chain(llm=llm, prompt=rag_prompt_template)
    rag_chain = create_retrieval_chain(
        retriever=history_aware_retriever,
        combine_docs_chain=qa_chain
    )

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key='input',
        history_messages_key='chat_history',
        output_messages_key='answer',
    )

    return conversational_rag_chain

def send_message(message):
    conversational_rag_chain = create_rag_chain()
    response = conversational_rag_chain.invoke(
        { 'input': message },
        { 'configurable': { 'session_id': '1' }}
    )
    message = response['answer']

    for chunk in message.split():
        yield chunk + " "
        time.sleep(0.05)
