# backend.py

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Literal 
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from dotenv import load_dotenv
import sqlite3
import requests
import random

load_dotenv()


# ----------------
# 1. LLM
# ----------------
llm = ChatOpenAI()


# ----------------
# 2. Tools
# ----------------
search_tool = DuckDuckGoSearchRun()

@tool
def calculator(first_num: float, second_num: float, operation: str) -> str:
    """
    Perform arithmetic on two numbers.

    Args:
        first_num: The first number.
        second_num: The second number.
        operation: One of 'add', 'sub', 'mul', or 'div'.

    Returns:
        A string describing the result.
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return "Error: Division by zero is not allowed."
            result = first_num / second_num
        else:
            return f"Error: Unsupported operation '{operation}'"

        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"
    
@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=API_KEY"
    r = requests.get(url)
    return r.json()

# ----------------
# Make tool list
# ----------------

tools = [get_stock_price, search_tool, calculator]

# Make the LLM tool-aware
llm_with_tools = llm.bind_tools(tools)


# ----------------
# 3. State
# ----------------
# state
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# ----------------
# 4. Nodes
# ----------------
# graph nodes
def chat_node(state: ChatState):
    """ LLM node that may answer or request a tool call. """
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools) # Execute tool calls

# ----------------
# 5. Checkpointer
# ----------------
conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)


# ----------------
# 6. Graph
# ----------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")

# If the LLM asked for a tool, go to ToolNode; else finish
graph.add_conditional_edges("chat_node", tools_condition)

# we will loop back to chat_node after tool execution to allow multiple tool calls in a conversation
graph.add_edge("tools", "chat_node")

# compile your graph 
chatbot = graph.compile(checkpointer=checkpointer)
