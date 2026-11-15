
from datetime import datetime
from decimal import Decimal, InvalidOperation

from app.extraction.schemas import InvoiceExtraction


TOLERANCE = Decimal("0.02")

REQUIRED_FIELDS = [
    "invoice_number",
    "invoice_date",
    "supplier_name",
    "total",
]


def to_decimal(value):
    if value is None:
        return None

    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None


def amounts_match(first, second):
    if first is None or second is None:
        return False

    return abs(first - second) <= TOLERANCE


def normalize_text(value):
    return " ".join(str(value or "").strip().lower().split())


def validate_invoice(
    invoice: InvoiceExtraction,
    existing_invoices: list[dict] | None = None,
) -> dict:
    errors = []
    warnings = []
    checks = {}

    # 1. Required fields
    missing_fields = []

    for field in REQUIRED_FIELDS:
        value = getattr(invoice, field, None)

        if value is None or not str(value).strip():
            missing_fields.append(field)

    checks["required_fields"] = not missing_fields

    if missing_fields:
        errors.append(
            f"Missing required fields: {', '.join(missing_fields)}"
        )

    # 2. Invoice date validation
    if invoice.invoice_date:
        try:
            datetime.strptime(invoice.invoice_date, "%Y-%m-%d")
            checks["invoice_date_format"] = True
        except ValueError:
            checks["invoice_date_format"] = False
            errors.append(
                "Invoice date must use YYYY-MM-DD format."
            )
    else:
        checks["invoice_date_format"] = False

    # 3. Line-item validation
    line_item_errors = []
    line_item_amounts = []

    for index, item in enumerate(invoice.line_items):
        quantity = to_decimal(item.quantity)
        unit_price = to_decimal(item.unit_price)
        amount = to_decimal(item.amount)

        if quantity is not None and unit_price is not None:
            expected_amount = quantity * unit_price

            if amount is None:
                line_item_errors.append(
                    f"Line item {index + 1}: amount is missing."
                )
            elif not amounts_match(expected_amount, amount):
                line_item_errors.append(
                    f"Line item {index + 1}: "
                    f"quantity × unit price does not match amount."
                )

        if amount is not None:
            line_item_amounts.append(amount)

    checks["line_items"] = not line_item_errors
    errors.extend(line_item_errors)

    # 4. Subtotal validation
    subtotal = to_decimal(invoice.subtotal)

    if invoice.line_items and len(line_item_amounts) == len(invoice.line_items):
        calculated_subtotal = sum(
            line_item_amounts, Decimal("0")
        )

        subtotal_valid = amounts_match(
            calculated_subtotal,
            subtotal,
        )

        checks["subtotal"] = subtotal_valid

        if not subtotal_valid:
            errors.append(
                "Sum of line-item amounts does not match subtotal."
            )
    else:
        checks["subtotal"] = None
        warnings.append(
            "Subtotal could not be fully verified from line items."
        )

    # 5. Total validation
    tax = to_decimal(invoice.tax)
    total = to_decimal(invoice.total)

    if subtotal is not None and tax is not None and total is not None:
        calculated_total = subtotal + tax

        total_valid = amounts_match(calculated_total, total)
        checks["total"] = total_valid

        if not total_valid:
            errors.append(
                "Subtotal plus tax does not match invoice total."
            )
    else:
        checks["total"] = None
        warnings.append(
            "Total could not be fully verified because "
            "subtotal, tax, or total is missing."
        )

    # 6. Duplicate detection
    duplicate = False

    invoice_number = normalize_text(invoice.invoice_number)
    supplier_name = normalize_text(invoice.supplier_name)

    for existing in existing_invoices or []:
        existing_number = normalize_text(
            existing.get("invoice_number")
        )
        existing_supplier = normalize_text(
            existing.get("supplier_name")
        )

        if (
            invoice_number
            and supplier_name
            and invoice_number == existing_number
            and supplier_name == existing_supplier
        ):
            duplicate = True
            break

    checks["duplicate"] = not duplicate

    if duplicate:
        errors.append(
            "Potential duplicate invoice detected."
        )

    # 7. Extraction uncertainty
    if invoice.uncertain_fields:
        warnings.append(
            "Uncertain fields require human verification: "
            + ", ".join(invoice.uncertain_fields)
        )

    if invoice.missing_fields:
        warnings.append(
            "The extraction model reported missing fields: "
            + ", ".join(invoice.missing_fields)
        )

    # 8. Final decision
    review_required = bool(
        errors
        or warnings
        or invoice.uncertain_fields
        or invoice.missing_fields
    )

    return {
        "valid": not errors,
        "review_required": review_required,
        "errors": errors,
        "warnings": warnings,
        "checks": checks,
    }
