from logging import config
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages # reducer function
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

load_dotenv()

llm = ChatOpenAI()

# user message - the human ask the question
# AI message - the reply from llm
# system message - the roles that you provide to your llm

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages] # we are using reducer function
    

def chat_node(state: ChatState):
    
    # take the user query from state
    message = state['messages']
    # send it to llm
    response = llm.invoke(message)
    # response store the state
    return {'messages': [response]}

conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)
# Checkpointer
checkpointer = SqliteSaver(conn=conn)

# Create our graph
graph = StateGraph(ChatState)
# add nodes to your graph
graph.add_node('chat_node', chat_node)
# add edges to your graph
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

# compile your graph 
basic_chatbot_workflow = graph.compile(checkpointer=checkpointer)


# test

# Define config type properly
class ChatConfig(TypedDict):
    configurable: dict[str, str]

config: ChatConfig = {"configurable": {"thread_id": 'thread-1'}}

response = basic_chatbot_workflow.invoke(
    {'messages': [HumanMessage(content='What is my name?')]},
                config = config
)
print(response)
