
import re

from datetime import date
from decimal import Decimal, InvalidOperation

from app.extraction.schemas import InvoiceExtraction


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

TOLERANCE = Decimal("0.02")

MIN_FIELD_CONFIDENCE = 0.70

REQUIRED_FIELDS = [
    "invoice_number",
    "invoice_date",
    "supplier_name",
    "total",
]


# ---------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------

def to_decimal(value) -> Decimal | None:
    """
    Safely convert a value to Decimal.

    Returns None for missing, invalid, or non-finite values.
    """

    if value is None or isinstance(value, bool):
        return None

    try:
        result = Decimal(str(value).strip())

        if not result.is_finite():
            return None

        return result

    except (InvalidOperation, ValueError, TypeError):
        return None


def is_invalid_numeric(value) -> bool:
    """
    Identify values that were supplied but cannot be safely
    converted into finite Decimal amounts.
    """

    return value is not None and to_decimal(value) is None


def amounts_match(
    first: Decimal | None,
    second: Decimal | None,
) -> bool:
    if first is None or second is None:
        return False

    return abs(first - second) <= TOLERANCE


def normalize_text(value) -> str:
    """
    Normalize text for comparison.
    """

    return " ".join(
        str(value or "").casefold().strip().split()
    )


def normalize_invoice_number(value) -> str:
    """
    Normalize invoice numbers for duplicate detection.

    Examples:
        INV-001 -> inv001
        INV 001 -> inv001
    """

    normalized = normalize_text(value)

    return re.sub(r"[^a-z0-9]", "", normalized)


def is_valid_iso_date(value) -> bool:
    """
    Validate strict YYYY-MM-DD dates.
    """

    if not value:
        return False

    try:
        parsed = date.fromisoformat(value)

        return parsed.isoformat() == value

    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------

def detect_duplicate(
    invoice: InvoiceExtraction,
    existing_invoices: list[dict] | None = None,
) -> dict:
    """
    Detect exact and probable duplicate invoices.

    Exact duplicate:
        Same normalized supplier and invoice number.

    Probable duplicate:
        Same normalized supplier, invoice date, and total,
        with compatible currency information.

    A probable duplicate is flagged for review rather than
    automatically treated as a confirmed duplicate.
    """

    invoice_number = normalize_invoice_number(
        invoice.invoice_number
    )

    supplier_name = normalize_text(
        invoice.supplier_name
    )

    invoice_date = invoice.invoice_date

    invoice_total = to_decimal(invoice.total)

    currency = normalize_text(invoice.currency)

    for existing in existing_invoices or []:

        existing_number = normalize_invoice_number(
            existing.get("invoice_number")
        )

        existing_supplier = normalize_text(
            existing.get("supplier_name")
        )

        existing_date = existing.get("invoice_date")

        existing_total = to_decimal(
            existing.get("total")
        )

        existing_currency = normalize_text(
            existing.get("currency")
        )

        same_supplier = (
            supplier_name
            and supplier_name == existing_supplier
        )

        # Exact identity match.
        if (
            same_supplier
            and invoice_number
            and invoice_number == existing_number
        ):
            return {
                "detected": True,
                "type": "exact",
                "message": "Potential duplicate invoice detected.",
                "matching_invoice": {
                    "invoice_number": existing.get("invoice_number"),
                    "supplier_name": existing.get("supplier_name"),
                },
            }

        # Probable duplicate based on financial and date details.
        currencies_compatible = (
            not currency
            or not existing_currency
            or currency == existing_currency
        )

        same_financial_details = (
            same_supplier
            and invoice_date
            and invoice_date == existing_date
            and amounts_match(
                invoice_total,
                existing_total,
            )
            and currencies_compatible
        )

        if same_financial_details:
            return {
                "detected": True,
                "type": "probable",
                "message": (
                    "Possible duplicate invoice detected based on "
                    "supplier, invoice date, and total amount."
                ),
                "matching_invoice": {
                    "invoice_number": existing.get("invoice_number"),
                    "supplier_name": existing.get("supplier_name"),
                },
            }

    return {
        "detected": False,
        "type": None,
        "message": None,
        "matching_invoice": None,
    }


# ---------------------------------------------------------------------
# Invoice validation
# ---------------------------------------------------------------------

def validate_invoice(
    invoice: InvoiceExtraction,
    existing_invoices: list[dict] | None = None,
) -> dict:
    """
    Validate extracted invoice data.

    Validation includes:
        - Required fields.
        - Date formats.
        - Numeric values.
        - Line-item calculations.
        - Subtotal reconciliation.
        - Tax, discount, and total reconciliation.
        - Duplicate detection.
        - Extraction confidence.
        - Manual-review requirements.

    Returns:
        A dictionary containing validation results.
    """

    errors: list[str] = []
    warnings: list[str] = []
    checks: dict = {}

    review_reasons: list[str] = []

    # -------------------------------------------------------------
    # 1. Required fields
    # -------------------------------------------------------------

    missing_fields = []

    for field in REQUIRED_FIELDS:

        value = getattr(invoice, field, None)

        if value is None or not str(value).strip():
            missing_fields.append(field)

    checks["required_fields"] = not missing_fields

    if missing_fields:
        errors.append(
            "Missing required fields: "
            + ", ".join(missing_fields)
        )

        review_reasons.append(
            "One or more required invoice fields are missing."
        )

    # -------------------------------------------------------------
    # 2. Date validation
    # -------------------------------------------------------------

    if invoice.invoice_date:

        invoice_date_valid = is_valid_iso_date(
            invoice.invoice_date
        )

        checks["invoice_date_format"] = invoice_date_valid

        if not invoice_date_valid:
            errors.append(
                "Invoice date must use YYYY-MM-DD format."
            )

    else:
        checks["invoice_date_format"] = False

    if invoice.due_date:

        due_date_valid = is_valid_iso_date(
            invoice.due_date
        )

        checks["due_date_format"] = due_date_valid

        if not due_date_valid:
            errors.append(
                "Due date must use YYYY-MM-DD format."
            )

        elif (
            invoice.invoice_date
            and is_valid_iso_date(invoice.invoice_date)
            and date.fromisoformat(invoice.due_date)
            < date.fromisoformat(invoice.invoice_date)
        ):
            errors.append(
                "Due date cannot be earlier than invoice date."
            )

    else:
        checks["due_date_format"] = None

    # -------------------------------------------------------------
    # 3. Numeric field validation
    # -------------------------------------------------------------

    numeric_fields = [
        "subtotal",
        "tax",
        "discount",
        "total",
    ]

    invalid_numeric_fields = []

    for field in numeric_fields:

        value = getattr(invoice, field, None)

        if is_invalid_numeric(value):
            invalid_numeric_fields.append(field)

    checks["numeric_fields"] = not invalid_numeric_fields

    if invalid_numeric_fields:
        errors.append(
            "Invalid numeric values in fields: "
            + ", ".join(invalid_numeric_fields)
        )

        review_reasons.append(
            "One or more financial fields contain invalid amounts."
        )

    # -------------------------------------------------------------
    # 4. Line-item validation
    # -------------------------------------------------------------

    line_item_errors = []
    line_item_amounts: list[Decimal] = []

    all_line_amounts_available = bool(invoice.line_items)

    for index, item in enumerate(invoice.line_items):

        quantity = to_decimal(item.quantity)
        unit_price = to_decimal(item.unit_price)
        amount = to_decimal(item.amount)

        if (
            is_invalid_numeric(item.quantity)
            or is_invalid_numeric(item.unit_price)
            or is_invalid_numeric(item.amount)
        ):
            line_item_errors.append(
                f"Line item {index + 1}: contains an invalid numeric value."
            )

        if quantity is not None and unit_price is not None:

            expected_amount = quantity * unit_price

            if amount is None:

                line_item_errors.append(
                    f"Line item {index + 1}: amount is missing."
                )

                all_line_amounts_available = False

            elif not amounts_match(expected_amount, amount):

                line_item_errors.append(
                    f"Line item {index + 1}: "
                    "quantity multiplied by unit price "
                    "does not match amount."
                )

        if amount is not None:
            line_item_amounts.append(amount)

        else:
            all_line_amounts_available = False

    checks["line_items"] = not line_item_errors

    errors.extend(line_item_errors)

    if line_item_errors:
        review_reasons.append(
            "One or more line items failed validation."
        )

    # -------------------------------------------------------------
    # 5. Subtotal validation
    # -------------------------------------------------------------

    subtotal = to_decimal(invoice.subtotal)

    if (
        invoice.line_items
        and all_line_amounts_available
        and len(line_item_amounts) == len(invoice.line_items)
        and subtotal is not None
    ):

        calculated_subtotal = sum(
            line_item_amounts,
            Decimal("0"),
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

            review_reasons.append(
                "Invoice subtotal does not match its line items."
            )

    else:
        checks["subtotal"] = None

        warnings.append(
            "Subtotal could not be fully verified from line items."
        )

        review_reasons.append(
            "Subtotal requires manual verification."
        )

    # -------------------------------------------------------------
    # 6. Financial reconciliation
    #
    # Formula:
    # subtotal + tax - discount = total
    #
    # An absent discount is treated as zero for reconciliation.
    # A missing tax is not assumed to be zero.
    # -------------------------------------------------------------

    tax = to_decimal(invoice.tax)
    discount = to_decimal(invoice.discount)
    total = to_decimal(invoice.total)

    if (
        subtotal is not None
        and tax is not None
        and total is not None
    ):

        effective_discount = (
            discount
            if discount is not None
            else Decimal("0")
        )

        calculated_total = (
            subtotal + tax - effective_discount
        )

        total_valid = amounts_match(
            calculated_total,
            total,
        )

        checks["total"] = total_valid

        if not total_valid:
            errors.append(
                "Subtotal plus tax minus discount "
                "does not match invoice total."
            )

            review_reasons.append(
                "Financial reconciliation failed."
            )

    else:
        checks["total"] = None

        warnings.append(
            "Total could not be fully verified because "
            "subtotal, tax, or total is missing or invalid."
        )

        review_reasons.append(
            "Financial reconciliation requires manual verification."
        )

    # -------------------------------------------------------------
    # 7. Duplicate detection
    # -------------------------------------------------------------

    duplicate_result = detect_duplicate(
        invoice=invoice,
        existing_invoices=existing_invoices,
    )

    duplicate_detected = duplicate_result["detected"]

    checks["duplicate"] = not duplicate_detected

    if duplicate_result["type"] == "exact":

        errors.append(
            duplicate_result["message"]
        )

        review_reasons.append(
            "An exact invoice identity match was found."
        )

    elif duplicate_result["type"] == "probable":

        warnings.append(
            duplicate_result["message"]
        )

        review_reasons.append(
            "A probable duplicate requires manual verification."
        )

    # -------------------------------------------------------------
    # 8. Extraction uncertainty
    # -------------------------------------------------------------

    if invoice.uncertain_fields:

        warnings.append(
            "Uncertain fields require human verification: "
            + ", ".join(invoice.uncertain_fields)
        )

        review_reasons.append(
            "One or more extracted fields are uncertain."
        )

    if invoice.missing_fields:

        warnings.append(
            "The extraction model reported missing fields: "
            + ", ".join(invoice.missing_fields)
        )

        review_reasons.append(
            "The extraction model reported missing fields."
        )

    # -------------------------------------------------------------
    # 9. Field-level confidence
    # -------------------------------------------------------------

    low_confidence_fields = []
    invalid_confidence_fields = []

    for field, confidence in invoice.confidence_scores.items():

        confidence_decimal = to_decimal(confidence)

        if (
            confidence_decimal is None
            or confidence_decimal < Decimal("0")
            or confidence_decimal > Decimal("1")
        ):

            invalid_confidence_fields.append(field)
            continue

        if confidence_decimal < Decimal(
            str(MIN_FIELD_CONFIDENCE)
        ):

            low_confidence_fields.append(field)

    checks["confidence_scores"] = (
        not invalid_confidence_fields
    )

    if invalid_confidence_fields:

        errors.append(
            "Invalid confidence scores for fields: "
            + ", ".join(invalid_confidence_fields)
        )

        review_reasons.append(
            "One or more extraction confidence scores are invalid."
        )

    if low_confidence_fields:

        warnings.append(
            "Low-confidence extraction fields require verification: "
            + ", ".join(low_confidence_fields)
        )

        review_reasons.append(
            "One or more fields have low extraction confidence."
        )

    # -------------------------------------------------------------
    # 10. Final decision
    # -------------------------------------------------------------

    review_required = bool(
        errors
        or warnings
        or review_reasons
    )

    return {
        "valid": not errors,
        "review_required": review_required,
        "errors": errors,
        "warnings": warnings,
        "checks": checks,
        "duplicate_details": duplicate_result,
        "low_confidence_fields": low_confidence_fields,
        "review_reasons": list(dict.fromkeys(review_reasons)),
    }
