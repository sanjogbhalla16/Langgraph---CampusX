import streamlit as st

with st.chat_message("user"):
    st.text("Hello, how are you?")
    
with st.chat_message("assistant"):
    st.text("I am good, thank you! How can I assist you today?")
    

user_input = st.chat_input("Type here")

if user_input:
    with st.chat_message("user"):
        st.text(user_input)
        