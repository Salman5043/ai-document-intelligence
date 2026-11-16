
import { useState } from "react";
import { searchDocuments } from "../services/api";

export default function DocumentSearch() {
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState<"keyword" | "semantic">("keyword");
  const [results, setResults] = useState<any[]>([]);
  const [error, setError] = useState("");

  async function handleSearch() {
    if (!query.trim()) return;

    setError("");

    try {
      const data = await searchDocuments(query, mode);
      setResults(Array.isArray(data) ? data : data.results ?? []);
    } catch {
      setError("Search failed. Please check the API.");
    }
  }

  return (
    <section className="panel">
      <h2>Search documents</h2>

      <input
        placeholder="Search invoices or document content..."
        value={query}
        onChange={(event) => setQuery(event.target.value)}
      />

      <select
        value={mode}
        onChange={(event) =>
          setMode(event.target.value as "keyword" | "semantic")
        }
      >
        <option value="keyword">Keyword search</option>
        <option value="semantic">Semantic search</option>
      </select>

      <button onClick={handleSearch}>Search</button>

      {error && <p className="error">{error}</p>}

      <ul>
        {results.map((item, index) => (
          <li key={item.id ?? index}>
            <strong>{item.filename ?? `Document ${item.id}`}</strong>
            <p>{item.extracted_text?.slice(0, 200)}</p>
          </li>
        ))}
      </ul>
    </section>
  );
}
