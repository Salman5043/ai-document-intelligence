
import { useCallback, useEffect, useState } from "react";
import {
  correctInvoice,
  getDocumentResult,
} from "../services/api";

interface Props {
  documentId: number | string;
}

interface Invoice {
  id: number;
  invoice_number?: string | null;
  invoice_date?: string | null;
  supplier_name?: string | null;
  customer_name?: string | null;
  subtotal?: number | string | null;
  tax?: number | string | null;
  total?: number | string | null;
  currency?: string | null;
  line_items?: Array<{
    description?: string;
    quantity?: number;
    unit_price?: number;
    amount?: number;
  }>;
  valid?: boolean;
  review_required?: boolean;
  validation_errors?: string[];
  validation_warnings?: string[];
}

const editableFields = [
  "invoice_number",
  "invoice_date",
  "supplier_name",
  "customer_name",
  "subtotal",
  "tax",
  "total",
  "currency",
] as const;

export default function InvoiceResult({ documentId }: Props) {
  const [invoice, setInvoice] = useState<Invoice | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");

  const loadResult = useCallback(async () => {
    setLoading(true);
    setError("");

    try {
      const data = await getDocumentResult(documentId);

      // The API returns "invoices", not "invoice".
      const extractedInvoice = data.invoices?.[0];

      if (!extractedInvoice) {
        setInvoice(null);
        setError("No invoice record was found for this document.");
        return;
      }

      setInvoice(extractedInvoice);
    } catch (err: any) {
      setError(
        err.response?.data?.detail ||
          "Unable to load the extracted invoice."
      );
    } finally {
      setLoading(false);
    }
  }, [documentId]);

  useEffect(() => {
    loadResult();
  }, [loadResult]);

  function updateField(field: string, value: string) {
    setInvoice((current) => {
      if (!current) return current;

      return {
        ...current,
        [field]: value,
      };
    });

    setMessage("");
  }

  async function saveChanges() {
    if (!invoice) return;

    setSaving(true);
    setError("");
    setMessage("");

    try {
      const correction = {
        invoice_number: invoice.invoice_number || null,
        invoice_date: invoice.invoice_date || null,
        supplier_name: invoice.supplier_name || null,
        customer_name: invoice.customer_name || null,
        subtotal:
          invoice.subtotal === null || invoice.subtotal === undefined ||
          invoice.subtotal === ""
            ? null
            : Number(invoice.subtotal),
        tax:
          invoice.tax === null || invoice.tax === undefined ||
          invoice.tax === ""
            ? null
            : Number(invoice.tax),
        total:
          invoice.total === null || invoice.total === undefined ||
          invoice.total === ""
            ? null
            : Number(invoice.total),
        currency: invoice.currency || null,
      };

      // Important: use invoice ID (3), not document ID (9).
      await correctInvoice(invoice.id, correction);

      setMessage("Invoice corrections saved successfully.");

      // Reload the latest server-side values and validation result.
      await loadResult();
    } catch (err: any) {
      const detail = err.response?.data?.detail;

      setError(
        typeof detail === "string"
          ? detail
          : JSON.stringify(detail || "Unable to save invoice corrections.")
      );
    } finally {
      setSaving(false);
    }
  }

  if (loading) {
    return (
      <section className="panel">
        <p>Loading extracted invoice...</p>
      </section>
    );
  }

  if (!invoice) {
    return (
      <section className="panel">
        <h2>Extracted invoice</h2>
        <p className="error">{error || "No invoice found."}</p>
      </section>
    );
  }

  const fields = [
    { name: "invoice_number", label: "Invoice number", type: "text" },
    { name: "invoice_date", label: "Invoice date", type: "text" },
    { name: "supplier_name", label: "Supplier name", type: "text" },
    { name: "customer_name", label: "Customer name", type: "text" },
    { name: "subtotal", label: "Subtotal", type: "number" },
    { name: "tax", label: "VAT / Tax", type: "number" },
    { name: "total", label: "Total", type: "number" },
    { name: "currency", label: "Currency", type: "text" },
  ];

  return (
    <section className="panel">
      <h2>Extracted invoice</h2>

      <p>
        Invoice ID: <strong>{invoice.id}</strong>
      </p>

      <p>
        Validation:{" "}
        <strong>
          {invoice.valid ? "Valid" : "Review required"}
        </strong>
      </p>

      <div className="invoice-fields">
        {fields.map((field) => (
          <label className="field" key={field.name}>
            <span>{field.label}</span>

            <input
              type={field.type}
              step={field.type === "number" ? "0.01" : undefined}
              value={(invoice as any)[field.name] ?? ""}
              onChange={(event) =>
                updateField(field.name, event.target.value)
              }
            />
          </label>
        ))}
      </div>

      <button onClick={saveChanges} disabled={saving}>
        {saving ? "Saving..." : "Save corrections"}
      </button>

      {message && <p>{message}</p>}
      {error && <p className="error">{error}</p>}

      <h3>Invoice line items</h3>

      {invoice.line_items?.length ? (
        <div className="table-wrapper">
          <table>
            <thead>
              <tr>
                <th>Description</th>
                <th>Quantity</th>
                <th>Unit price</th>
                <th>Amount</th>
              </tr>
            </thead>
            <tbody>
              {invoice.line_items.map((item, index) => (
                <tr key={index}>
                  <td>{item.description}</td>
                  <td>{item.quantity}</td>
                  <td>{item.unit_price}</td>
                  <td>{item.amount}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p>No line items available.</p>
      )}

      {invoice.validation_errors?.length ? (
        <>
          <h3>Validation errors</h3>
          <ul>
            {invoice.validation_errors.map((item, index) => (
              <li key={index}>{item}</li>
            ))}
          </ul>
        </>
      ) : null}

      {invoice.validation_warnings?.length ? (
        <>
          <h3>Validation warnings</h3>
          <ul>
            {invoice.validation_warnings.map((item, index) => (
              <li key={index}>{item}</li>
            ))}
          </ul>
        </>
      ) : null}
    </section>
  );
}
