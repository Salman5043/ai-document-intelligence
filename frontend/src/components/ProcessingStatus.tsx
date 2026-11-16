
import { useEffect, useState } from "react";
import { getDocumentStatus } from "../services/api";

interface Props {
  documentId: number | string;
  onCompleted: () => void;
}

export default function ProcessingStatus({
  documentId,
  onCompleted,
}: Props) {
  const [status, setStatus] = useState("uploaded");
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;
    let timer: number | undefined;

    async function checkStatus() {
      try {
        const result = await getDocumentStatus(documentId);
        if (!active) return;

        setStatus(result.status);

        if (result.status === "processed") {
          onCompleted();
          return;
        }

        if (result.status === "failed") {
          setError("Document processing failed. Check the backend logs.");
          return;
        }

        timer = window.setTimeout(checkStatus, 2000);
      } catch {
        if (!active) return;
        setError("Unable to retrieve processing status.");
      }
    }

    checkStatus();

    return () => {
      active = false;
      if (timer) window.clearTimeout(timer);
    };
  }, [documentId, onCompleted]);

  return (
    <section className="panel">
      <h3>Processing status</h3>
      <p>Status: <strong>{status}</strong></p>
      {status !== "processed" && status !== "failed" && (
        <p>Checking again in a moment...</p>
      )}
      {error && <p className="error">{error}</p>}
    </section>
  );
}
