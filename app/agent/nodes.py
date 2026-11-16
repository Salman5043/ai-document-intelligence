
from app.agent.state import InvoiceExtractionState
from app.extraction.llm import get_llm
from app.extraction.schemas import InvoiceExtraction

EXTRACTION_PROMPT = """
You are an expert invoice data extraction system.

Extract invoice information from the supplied document.

Rules:
1. Extract only information explicitly supported by the document.
2. Never invent, estimate, or assume missing values.
3. Return null for missing or unreadable values.
4. Preserve invoice numbers as written.
5. Convert dates to YYYY-MM-DD only when the date is unambiguous.
6. Extract all identifiable invoice line items.
7. Do not confuse subtotal, tax, and total.
8. Identify the currency only when supported by the document.
9. Add important absent fields to missing_fields.
10. Add unclear or conflicting fields to uncertain_fields.
11. Put relevant explanations in extraction_notes.

The document may contain OCR mistakes. Do not silently guess
when an OCR error makes a value uncertain.

Return data using the requested structured schema.
"""


def prepare_document(state: InvoiceExtractionState):
    document = state.get("document", {})

    text_parts = []

    # Extracted text from PDF pages or images
    for page in document.get("pages", []):
        page_text = page.get("text", "")
        if page_text:
            page_number = page.get("page_number", "")
            text_parts.append(
                f"PAGE {page_number}\n{page_text}"
            )

    # Extracted spreadsheet or CSV tables
    for table in document.get("tables", []):
        sheet_name = table.get("sheet_name", "Table")
        records = table.get("records", [])

        text_parts.append(
            f"TABLE: {sheet_name}\n{records}"
        )

    # Fallback for documents containing only extracted_text
    if not text_parts:
        extracted_text = document.get("extracted_text", "")
        if extracted_text:
            text_parts.append(extracted_text)

    document_text = "\n\n".join(text_parts)

    warnings = list(state.get("warnings", []))

    if not document_text.strip():
        warnings.append("No usable document text was found.")

    return {
        "document_text": document_text,
        "warnings": warnings,
    }



def extract_invoice(state: InvoiceExtractionState):
    document_text = state.get("document_text", "")

    if not document_text.strip():
        return {
            "extraction": None,
            "errors": ["Document contains no extractable text."],
        }

    try:
        llm = get_llm()

        structured_llm = llm.with_structured_output(
            InvoiceExtraction
        )

        result = structured_llm.invoke([
            ("system", EXTRACTION_PROMPT),
            (
                "human",
                "Extract invoice information from this document:\n\n"
                + document_text
            ),
        ])

        return {
            "extraction": result,
            "errors": [],
        }

    except Exception as exc: # noqa: BLE001 
        return {
            "extraction": None,
            "errors": [
                f"Invoice extraction failed: {exc!s}"
            ],
        }
