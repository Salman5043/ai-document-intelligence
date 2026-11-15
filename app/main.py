
from fastapi import FastAPI

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
