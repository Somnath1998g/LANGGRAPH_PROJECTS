from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv

load_dotenv()
llm= ChatGroq(model="llama-3.3-70b-versatile")

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    # take user query from state
    messages = state['messages']
    response = llm.invoke(messages)
    return {'messages': [response]}

checkpoint_saver = InMemorySaver()
graph=StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START,"chat_node")
graph.add_edge("chat_node",END)
chatbot=graph.compile(checkpointer=checkpoint_saver)

