"""LCEL generation chains and LangGraph CRAG state machine."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

STUFF_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant. Answer the question using only the provided context. "
        "If the context does not contain enough information to answer, say so clearly.",
    ),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])

CITATION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a precise assistant. Answer the question using only the provided context. "
        "For each factual claim, cite the source snippet using [1], [2], etc.",
    ),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])

QUERY_REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Rewrite the user's question to be more specific and retrieval-friendly. "
        "Return only the rewritten question with no explanation.",
    ),
    ("human", "{question}"),
])

HYDE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Write a short hypothetical document (2–4 sentences) that would directly answer this "
        "question. This hypothetical document will be used to improve retrieval.",
    ),
    ("human", "{question}"),
])

CRAG_GRADE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a relevance grader. Is the following document relevant to the question? "
        "Reply with only 'yes' or 'no'.",
    ),
    ("human", "Question: {question}\n\nDocument: {document}"),
])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_docs(docs: list[Document]) -> str:
    """Format a list of Documents into a numbered context block."""
    return "\n\n---\n\n".join(f"[{i + 1}] {doc.page_content}" for i, doc in enumerate(docs))


# ---------------------------------------------------------------------------
# Generation helpers (context already retrieved)
# ---------------------------------------------------------------------------

def answer_with_stuff(llm, question: str, docs: list[Document]) -> str:
    """Generate an answer using the 'stuff all context' strategy.

    Args:
        llm: LangChain chat model.
        question: User query.
        docs: Retrieved documents.

    Returns:
        Generated answer string.
    """
    chain = STUFF_PROMPT | llm | StrOutputParser()
    return chain.invoke({"context": format_docs(docs), "question": question})


def answer_with_citation(llm, question: str, docs: list[Document]) -> str:
    """Generate a citation-grounded answer referencing context snippets by number.

    Args:
        llm: LangChain chat model.
        question: User query.
        docs: Retrieved documents.

    Returns:
        Generated answer with inline citations.
    """
    chain = CITATION_PROMPT | llm | StrOutputParser()
    return chain.invoke({"context": format_docs(docs), "question": question})


def rewrite_query(llm, question: str) -> str:
    """Rewrite a query to be more retrieval-friendly.

    Config key: query_transform = "rewrite"
    """
    chain = QUERY_REWRITE_PROMPT | llm | StrOutputParser()
    return chain.invoke({"question": question})


def generate_hyde_query(llm, question: str) -> str:
    """Generate a hypothetical document for HyDE retrieval.

    Config key: query_transform = "hyde"
    """
    chain = HYDE_PROMPT | llm | StrOutputParser()
    return chain.invoke({"question": question})


# ---------------------------------------------------------------------------
# LangGraph CRAG state machine
# ---------------------------------------------------------------------------

def build_crag_graph(llm, retriever):
    """Build and compile a CRAG self-correction graph using LangGraph.

    The graph: retrieve → grade → (generate | rewrite → retrieve → grade → generate).
    One retry is allowed. If no relevant docs are found after retry, generates anyway.

    Config key: advanced = "crag"

    Args:
        llm: LangChain chat model used for grading, rewriting, and generating.
        retriever: LangChain retriever.

    Returns:
        Compiled LangGraph CompiledGraph.
    """
    from typing import TypedDict

    from langgraph.graph import END, StateGraph

    class CRAGState(TypedDict):
        """Mutable state passed between CRAG nodes."""

        question: str
        documents: list[Document]
        generation: str
        retry_count: int

    grade_chain = CRAG_GRADE_PROMPT | llm | StrOutputParser()
    rewrite_chain = QUERY_REWRITE_PROMPT | llm | StrOutputParser()

    def retrieve_node(state: CRAGState) -> CRAGState:
        """Retrieve documents for the current question."""
        docs = retriever.invoke(state["question"])
        return {**state, "documents": docs}

    def grade_node(state: CRAGState) -> CRAGState:
        """Filter retrieved documents to only those relevant to the question."""
        relevant = []
        for doc in state["documents"]:
            verdict = grade_chain.invoke(
                {"question": state["question"], "document": doc.page_content}
            )
            if "yes" in verdict.lower():
                relevant.append(doc)
        return {**state, "documents": relevant}

    def generate_node(state: CRAGState) -> CRAGState:
        """Generate an answer from the (filtered) documents."""
        answer = answer_with_stuff(llm, state["question"], state["documents"])
        return {**state, "generation": answer}

    def rewrite_node(state: CRAGState) -> CRAGState:
        """Rewrite the query and increment retry counter."""
        rewritten = rewrite_chain.invoke({"question": state["question"]})
        return {**state, "question": rewritten, "retry_count": state.get("retry_count", 0) + 1}

    def decide_action(state: CRAGState) -> str:
        """Route: generate if docs exist or max retries reached, else rewrite."""
        if state["documents"] or state.get("retry_count", 0) >= 1:
            return "generate"
        return "rewrite"

    graph = StateGraph(CRAGState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_node)
    graph.add_node("generate", generate_node)
    graph.add_node("rewrite", rewrite_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges(
        "grade", decide_action, {"generate": "generate", "rewrite": "rewrite"}
    )
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("generate", END)

    return graph.compile()
