import streamlit as st 
from model.chat import send_message

st.title("Once Human Chatbot")

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = {}

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

prompt = st.chat_input('Have a chat with Once Human Chatbot')
if prompt:
    with st.chat_message('user'):
        st.markdown(prompt)

    st.session_state.messages.append({ 'role': 'user', 'content': prompt })

    with st.chat_message('assistant'):
        try:
            message_generator = send_message(prompt)
            response = st.write_stream(message_generator)

        except:
            response = 'Model currently is too busy. Please try again later.'
            st.markdown(response)

    st.session_state.messages.append({ 'role': 'assistant', 'content': response })