from langgraph.graph import StateGraph, START
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from typing import TypedDict,Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
import sys
import asyncio

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

load_dotenv()  # Load environment variables from .env file

llm = ChatOpenAI(model="gpt-5")

# MCP client for local FastMCP server
client = MultiServerMCPClient(
    {
        "arith": {
            "transport": "stdio",
            "command": "C:/Users/Sanjog Bhalla/Desktop/Langgraph - CampusX/MCP-Client-Langgraph/venv/Scripts/python.exe",
            "args": ["C:/Users/Sanjog Bhalla/Desktop/Langgraph - CampusX/MCP-Client-Langgraph/main.py"],
        }
    }
)


# state
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    

async def build_graph():
    
    tools = await client.get_tools()  # Dynamically fetch tools from the MCP server
    
    print(tools)

    llm_with_tools = llm.bind_tools(tools)
    
    # nodes
    async def chat_node(state: ChatState):

        messages = state["messages"]
        response = await llm_with_tools.ainvoke(messages)
        return {'messages': [response]}

    tool_node = ToolNode(tools) # This node is internally asynchronous

    # defining graph and nodes
    graph = StateGraph(ChatState)

    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", tool_node)

    # defining graph connections
    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")

    chatbot = graph.compile()
    return chatbot

async def main():
    
    chatbot = await build_graph()
    
    # running the graph
    result = await chatbot.ainvoke({"messages": [HumanMessage(content="Find the modulus of 132354 and 23 and give answer like a cricket commentator.")]})

    print(result['messages'][-1].content)
    
if __name__ == "__main__": # Here, main() only runs when you run this file directly; importing it just loads the function but does not execute it
    asyncio.run(main())
