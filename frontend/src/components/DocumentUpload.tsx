
import { useState } from "react";
import { uploadDocument } from "../services/api";

interface Props {
  onUploaded: (document: any) => void;
}

export default function DocumentUpload({ onUploaded }: Props) {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function handleUpload() {
    if (!file) {
      setError("Please select a document first.");
      return;
    }

    setLoading(true);
    setError("");

    try {
      const result = await uploadDocument(file);
      onUploaded(result);
      setFile(null);
    } catch (err: any) {
      setError(
        err.response?.data?.detail || "Document upload failed."
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="panel">
      <h2>Upload a document</h2>
      <p>Supported formats: PDF, JPG, PNG, TIFF, XLSX, XLS and CSV.</p>

      <input
        type="file"
        accept=".pdf,.jpg,.jpeg,.png,.tif,.tiff,.xlsx,.xls,.csv"
        onChange={(event) =>
          setFile(event.target.files?.[0] ?? null)
        }
      />

      {file && <p>Selected: {file.name}</p>}

      <button onClick={handleUpload} disabled={loading || !file}>
        {loading ? "Uploading..." : "Upload document"}
      </button>

      {error && <p className="error">{error}</p>}
    </section>
  );
}
