from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages # reducer function
from langgraph.checkpoint.memory import MemorySaver

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

# Checkpointer
checkpointer = MemorySaver()

# Create our graph
graph = StateGraph(ChatState)
# add nodes to your graph
graph.add_node('chat_node', chat_node)
# add edges to your graph
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

# compile your graph 
basic_chatbot_workflow = graph.compile(checkpointer=checkpointer)