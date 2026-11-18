
import logging
from typing import Any

from app.agent.state import InvoiceExtractionState
from app.extraction.llm import get_llm
from app.extraction.schemas import InvoiceExtraction

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """
You are an expert invoice data extraction system.

Extract invoice information from the supplied document.

STRICT RULES:
1. Extract only information explicitly supported by the document.
2. Never invent, estimate, or assume missing values.
3. Return null for missing or unreadable values.
4. Preserve invoice numbers as written.
5. Convert dates to YYYY-MM-DD only when unambiguous.
6. Extract all identifiable invoice line items.
7. Do not confuse subtotal, tax, discount, and total.
8. Identify currency only when supported by the document.
9. Add important absent fields to missing_fields.
10. Add unclear or conflicting fields to uncertain_fields.
11. Put relevant explanations in extraction_notes.
12. Do not calculate missing invoice values.
13. Do not treat purchase order numbers as invoice numbers.
14. Do not interpret OCR-uncertain text as confirmed information.
15. Return only information supported by the provided source.

For confidence_scores:
- Use a score between 0 and 1.
- Assign high confidence only when the source clearly supports the value.
- Assign lower confidence when text is ambiguous or OCR quality is poor.
- Do not assign confidence to missing values.

For field_evidence:
- Include short source excerpts supporting extracted fields.
- Do not create evidence that is absent from the source.

Return data using the requested structured schema.
"""


def prepare_document(state: InvoiceExtractionState):
    """Prepare extracted PDF, image, spreadsheet, or CSV content."""

    document = state.get("document", {})
    text_parts: list[str] = []
    warnings = list(state.get("warnings", []))

    # Extracted PDF/image page content
    for page in document.get("pages", []):
        page_text = page.get("text", "")

        if not page_text:
            continue

        page_number = page.get("page_number", "Unknown")
        text_parts.append(
            f"PAGE {page_number}\n{page_text}"
        )

        # Preserve OCR quality information
        average_confidence = page.get("average_confidence")

        if average_confidence is not None:
            try:
                confidence = float(average_confidence)

                if confidence < 0.70:
                    warnings.append(
                        f"Page {page_number} has low OCR confidence "
                        f"({confidence:.2f})."
                    )
            except (TypeError, ValueError):
                warnings.append(
                    f"Page {page_number} has invalid OCR confidence metadata."
                )

        for warning in page.get("warnings", []):
            warnings.append(
                f"Page {page_number}: {warning}"
            )

    # Extracted spreadsheet/CSV content
    for table in document.get("tables", []):
        sheet_name = table.get("sheet_name", "Table")
        records = table.get("records", [])

        if records:
            text_parts.append(
                f"TABLE: {sheet_name}\n{records}"
            )

    # Fallback for documents containing only extracted_text
    if not text_parts:
        extracted_text = document.get("extracted_text", "")

        if extracted_text:
            text_parts.append(extracted_text)

    document_text = "\n\n".join(text_parts)

    if not document_text.strip():
        warnings.append("No usable document text was found.")

    return {
        "document_text": document_text,
        "warnings": list(dict.fromkeys(warnings)),
    }


def _build_field_evidence(
    extraction: InvoiceExtraction,
    document_text: str,
) -> dict[str, str]:
    """
    Verify that returned evidence excerpts actually occur
    in the source document.

    This does not prove that the extracted value is correct;
    it only prevents unsupported evidence from being retained.
    """

    verified_evidence = {}

    for field_name, excerpt in extraction.field_evidence.items():
        if not isinstance(excerpt, str) or not excerpt.strip():
            continue

        if excerpt.casefold() in document_text.casefold():
            verified_evidence[field_name] = excerpt

    return verified_evidence



def extract_invoice(state: InvoiceExtractionState):
    """Extract structured invoice information using Groq."""

    document_text = state.get("document_text", "")

    if not document_text.strip():
        return {
            "extraction": None,
            "errors": ["Document contains no extractable text."],
        }

    llm = get_llm()

    structured_llm = llm.with_structured_output(
        InvoiceExtraction
    )

    result = structured_llm.invoke([
        ("system", EXTRACTION_PROMPT),
        (
            "human",
            "Extract invoice information from this document:\n\n"
            + document_text,
        ),
    ])

    if isinstance(result, dict):
        result = InvoiceExtraction.model_validate(result)

    if not isinstance(result, InvoiceExtraction):
        raise TypeError(
            "LLM returned an unexpected invoice extraction result."
        )

    verified_evidence = _build_field_evidence(
        result,
        document_text,
    )

    result.field_evidence = verified_evidence

    return {
        "extraction": result,
        "errors": [],
    }
