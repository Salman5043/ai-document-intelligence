from langgraph.graph import END, START, StateGraph

from app.agent.qa_nodes import (
    generate_grounded_answer,
    prepare_qa_context,
    retrieve_document_chunks,
)
from app.agent.qa_state import DocumentQAState


def build_qa_graph():
    workflow = StateGraph(DocumentQAState)

    workflow.add_node(
        "retrieve_chunks",
        retrieve_document_chunks,
    )

    workflow.add_node(
        "prepare_context",
        prepare_qa_context,
    )

    workflow.add_node(
        "generate_answer",
        generate_grounded_answer,
    )

    workflow.add_edge(START, "retrieve_chunks")
    workflow.add_edge("retrieve_chunks", "prepare_context")
    workflow.add_edge("prepare_context", "generate_answer")
    workflow.add_edge("generate_answer", END)

    return workflow.compile()


document_qa_graph = build_qa_graph()