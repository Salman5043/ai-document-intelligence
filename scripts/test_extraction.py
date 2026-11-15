import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from app.agent.graph import invoice_extraction_graph


sample_document = {
    "filename": "sample_invoice.pdf",
    "pages": [
        {
            "page_number": 1,
            "text": """
            ABC Technologies Ltd.
            INVOICE

            Invoice Number: INV-2026-001
            Invoice Date: 2026-10-15

            Supplier: ABC Technologies Ltd.
            Customer: XYZ Trading Ltd.

            Description       Quantity    Unit Price    Amount
            Software License      2          500.00     1000.00
            Support Service       1          200.00      200.00

            Subtotal: 1200.00
            Tax: 180.00
            Total: 1380.00
            Currency: USD
            """
        }
    ],
    "tables": [],
    "extracted_text": "",
}

result = invoice_extraction_graph.invoke({
    "document": sample_document,
    "document_text": "",
    "extraction": None,
    "existing_invoices": [],
    "validation": None,
    "errors": [],
    "warnings": [],
})

print("\n========== EXTRACTION RESULT ==========")

if result.get("extraction"):
    print(
        result["extraction"].model_dump_json(indent=2)
    )

print("\n========== ERRORS ==========")
print(result.get("errors", []))

print("\n========== WARNINGS ==========")
print(result.get("warnings", []))

print("\n========== VALIDATION RESULT ==========")
print(result.get("validation"))