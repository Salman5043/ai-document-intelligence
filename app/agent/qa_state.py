from typing import Any, TypedDict


class DocumentQAState(TypedDict):
    document_id: int
    question: str
    retrieval_limit: int

    retrieved_chunks: list[dict[str, Any]]
    context: str

    answer: str | None
    sources: list[dict[str, Any]]
    insufficient_evidence: bool

    errors: list[str]
    warnings: list[str]