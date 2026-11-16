
from langgraph.graph import END, START, StateGraph

from app.agent.nodes import (
    extract_invoice,
    prepare_document,
)
from app.agent.state import InvoiceExtractionState
from app.agent.validation_node import validate_extracted_invoice


def build_extraction_graph():
    workflow = StateGraph(InvoiceExtractionState)

    workflow.add_node("prepare_document", prepare_document)
    workflow.add_node("extract_invoice", extract_invoice)
    workflow.add_node("validate_invoice", validate_extracted_invoice)

    workflow.add_edge(START, "prepare_document")
    workflow.add_edge("prepare_document", "extract_invoice")
    workflow.add_edge("extract_invoice", "validate_invoice")
    workflow.add_edge("validate_invoice", END)

    return workflow.compile()


invoice_extraction_graph = build_extraction_graph()
