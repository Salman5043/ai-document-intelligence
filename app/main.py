
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.documents import router as documents_router
from app.api.routes.invoices import router as invoices_router

app = FastAPI(
    title="AI-Powered Document Intelligence System",
    description=(
        "Document upload, invoice extraction, validation, "
        "and intelligent search API."
    ),
    version="1.0.0",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=[
        "GET",
        "POST",
        "PATCH",
        "DELETE",
        "OPTIONS",
    ],
    allow_headers=[
        "Content-Type",
        "Authorization",
    ],
)

app.include_router(documents_router)
app.include_router(invoices_router)


@app.get("/")
def root():
    return {
        "message": "AI Document Intelligence API is running.",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
    }
