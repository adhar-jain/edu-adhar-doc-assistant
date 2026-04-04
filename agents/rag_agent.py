"""
rag_agent.py
LangGraph agent for RAG reasoning workflow
LangGraph-based RAG agent with conversation memory.
"""

from typing import TypedDict, List
from langgraph.graph import StateGraph, END


class AgentState(TypedDict):
    question: str
    rewritten_question: str
    context: str
    answer: str
    sources: list
    chat_history: list


def rewrite_query(state, llm):

    question = state["question"]

    prompt = f"""
    Rewrite the user question to improve document retrieval.

    Question:
    {question}
    """

    response = llm.invoke(prompt)

    return {"rewritten_question": response.content}


def retrieve(state, retriever):

    query = state["rewritten_question"]

    docs = retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in docs])

    sources = list(set([
        doc.metadata.get("source", "unknown")
        for doc in docs
    ]))

    return {
        "context": context,
        "sources": sources
    }


def reason(state, llm):

    prompt = f"""
    You are an AI assistant helping analyze enterprise documents.

    Context:
    {state['context']}

    Question:
    {state['question']}

    Think step-by-step before answering.
    """

    response = llm.invoke(prompt)

    return {"answer": response.content}


def build_rag_agent(llm, retriever):

    graph = StateGraph(AgentState)

    graph.add_node("rewrite", lambda s: rewrite_query(s, llm))
    graph.add_node("retrieve", lambda s: retrieve(s, retriever))
    graph.add_node("reason", lambda s: reason(s, llm))

    graph.set_entry_point("rewrite")

    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "reason")
    graph.add_edge("reason", END)

    return graph.compile()