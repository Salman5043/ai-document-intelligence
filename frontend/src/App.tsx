
import { useCallback, useState } from "react";
import DocumentUpload from "./components/DocumentUpload";
import ProcessingStatus from "./components/ProcessingStatus";
import InvoiceResult from "./components/InvoiceResult";
import DocumentSearch from "./components/DocumentSearch";
import "./App.css";

function App() {
  const [document, setDocument] = useState<any>(null);
  const [refresh, setRefresh] = useState(0);

  const handleCompleted = useCallback(() => {
    setRefresh((value) => value + 1);
  }, []);

  const handleUploaded = (result: any) => {
    setDocument(result);
    setRefresh(0);
  };

  const documentId = document?.document_id ?? document?.id;

  return (
    <main className="container">
      <header>
        <h1>Document Intelligence System</h1>
        <p>AI-powered document processing and invoice extraction</p>
      </header>

      <DocumentUpload onUploaded={handleUploaded} />

      {documentId && (
        <>
          <ProcessingStatus
            key={documentId}
            documentId={documentId}
            onCompleted={handleCompleted}
          />

          {refresh > 0 && (
            <InvoiceResult
              key={`${documentId}-${refresh}`}
              documentId={documentId}
            />
          )}
        </>
      )}

      <DocumentSearch />
    </main>
  );
}

export default App;
