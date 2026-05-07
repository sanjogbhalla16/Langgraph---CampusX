from typing import TypedDict
import streamlit as st
from langchain_core.messages import HumanMessage
from langgraph_tool_backend import chatbot
import uuid # used to generate unique thread ids for each conversation


# **********************************************Utility Functions**********************************************

def generate_thread_id():
    return str(uuid.uuid4())

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(st.session_state['thread_id'])
    st.session_state['message_history'] = []

def add_thread(thread_id, title='New Chat'): # to store both the uuid and the chat title
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'][thread_id] = title
        
def load_conversation(thread_id):
    # This function can be expanded to load conversation history from a database or file based on the thread_id
    state = chatbot.get_state(config={"configurable": {"thread_id": str(thread_id)}})
    return state.values.get('messages', [])

# *********************************************Session State Management*********************************************

# st.session_state -> dict -> in this dict we can store the conversation and it does not get lost when the page is refreshed
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

# Adding the unique thread id to the session state to manage conversations separately
if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()
    
# if 'chat_threads' not in st.session_state: # change it to dict where both uuid and the chat title comes
#     st.session_state['chat_threads'] = {}
    
if 'chat_threads' not in st.session_state or not isinstance(st.session_state['chat_threads'], dict):
    st.session_state['chat_threads'] = {}

add_thread(st.session_state['thread_id']) # Add the current thread id to the list of chat threads   
    
# ***********************************************Config Type Definition**********************************************

# Define config type properly
class ChatConfig(TypedDict):
    configurable: dict[str, str]

config: ChatConfig = {"configurable": {"thread_id": str(st.session_state['thread_id'])}}

# **********************************************SideBar UI **********************************************

st.sidebar.title("Langgraph Chatbot")

if st.sidebar.button("New Chat"):
    reset_chat()

st.sidebar.header('My Conversations')

# We need to show the title on each chat button
for thread_id, title in reversed(list(st.session_state['chat_threads'].items())): # Display threads in reverse order (most recent first)
     if st.sidebar.button(title, key=f"chat_{thread_id}"):
        st.session_state['thread_id'] = thread_id
        messages = load_conversation(thread_id)
        
        temp_messages = []
        
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = 'user'
            else:
                role = 'assistant'
            temp_messages.append({'role': role, 'content': msg.content})
            
        st.session_state['message_history'] = temp_messages

# **********************************************Loading Conversation History and Main UI*********************************************
# we are loading the conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input("Type here")

if user_input:
    
    # first we need to see that the input that we are getting is for a New Chat
    if st.session_state['chat_threads'].get(st.session_state['thread_id'], 'New Chat') == "New Chat":
        st.session_state['chat_threads'][st.session_state['thread_id']] = user_input[:40] # take that thread id and assign the user_input as the title upto 40 chars only
        
    # first add the message to message_history
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message("user"):
        st.text(user_input)
    
    with st.chat_message("assistant"):
        ai_message = st.write_stream(
            message_chunk.content for message_chunk, metadata in chatbot.stream(
                {'messages': [HumanMessage(content=user_input)]},
                config = config,
                stream_mode = 'messages'
            )
            
        )
    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})    


        
