
import axios from "axios";

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL,
  timeout: 30000,
});

export default api;

export async function uploadDocument(file: File) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await api.post("/documents/upload", formData);
  return response.data;
}

export async function getDocumentStatus(id: number | string) {
  const response = await api.get(`/documents/${id}/status`);
  return response.data;
}

export async function getDocumentResult(id: number | string) {
  const response = await api.get(`/documents/${id}/result`);
  return response.data;
}

export async function searchDocuments(
  query: string,
  mode: "keyword" | "semantic"
) {
  const response = await api.get("/documents/search", {
    params: { q: query, mode, limit: 20 },
  });
  return response.data;
}

export async function correctInvoice(
  id: number | string,
  correction: Record<string, unknown>
) {
  const response = await api.patch(`/invoices/${id}`, correction);
  return response.data;
}
