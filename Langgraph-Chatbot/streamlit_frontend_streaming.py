from typing import TypedDict
import streamlit as st
from langchain_core.messages import HumanMessage
from langgraph_backend import basic_chatbot_workflow

# Define config type properly
class ChatConfig(TypedDict):
    configurable: dict[str, str]

config: ChatConfig = {"configurable": {"thread_id": '1'}}

# st.session_state -> dict -> in this dict we can store the conversation and it does not get lost when the page is refreshed
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []


# we are loading the conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input("Type here")

if user_input:
    
    # first add the message to message_history
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message("user"):
        st.text(user_input)
    
    with st.chat_message("assistant"):
        ai_message = st.write_stream(
            message_chunk.content for message_chunk, metadata in basic_chatbot_workflow.stream(
                {'messages': [HumanMessage(content=user_input)]},
                config = {"configurable": {"thread_id": '1'}},
                stream_mode = 'messages'
            )
            
        )
    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})    


        
