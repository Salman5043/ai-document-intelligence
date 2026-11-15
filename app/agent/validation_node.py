
from app.agent.state import InvoiceExtractionState
from app.validation.validator import validate_invoice


def validate_extracted_invoice(
    state: InvoiceExtractionState,
):
    invoice = state.get("extraction")

    if invoice is None:
        return {
            "validation": {
                "valid": False,
                "review_required": True,
                "errors": ["No invoice extraction result available."],
                "warnings": [],
                "checks": {},
            }
        }

    validation_result = validate_invoice(
        invoice=invoice,
        existing_invoices=state.get("existing_invoices", []),
    )

    return {
        "validation": validation_result,
    }
