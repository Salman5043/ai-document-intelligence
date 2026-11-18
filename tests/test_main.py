
from fastapi.testclient import TestClient

from app.api.routes.documents import router as documents_router
from app.api.routes.invoices import router as invoices_router
from app.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "AI Document Intelligence API is running.",
        "docs": "/docs",
    }


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
    }


def test_api_documentation_is_available():
    response = client.get("/docs")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_openapi_schema():
    response = client.get("/openapi.json")

    assert response.status_code == 200

    schema = response.json()

    assert schema["info"]["title"] == (
        "AI-Powered Document Intelligence System"
    )
    assert schema["info"]["version"] == "1.0.0"


def test_document_router_is_registered():
    app_paths = set(app.openapi()["paths"].keys())

    document_paths = {
        route.path
        for route in documents_router.routes
        if hasattr(route, "path")
    }

    assert document_paths
    assert document_paths.issubset(app_paths)


def test_invoice_router_is_registered():
    app_paths = set(app.openapi()["paths"].keys())

    invoice_paths = {
        route.path
        for route in invoices_router.routes
        if hasattr(route, "path")
    }

    assert invoice_paths
    assert invoice_paths.issubset(app_paths)

def test_cors_allows_frontend_origin():
    response = client.options(
        "/",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == (
        "http://localhost:5173"
    )
    assert response.headers["access-control-allow-credentials"] == "true"


def test_cors_rejects_unconfigured_origin():
    response = client.options(
        "/",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert "access-control-allow-origin" not in response.headers
