# AI-Powered Document Intelligence System

An AI-powered document intelligence platform that processes PDFs, images, and spreadsheets to extract structured invoice information, validate financial data, and enable document search and question answering.

The system combines OCR, LLM-based information extraction, asynchronous processing, PostgreSQL, vector search, and a React-based frontend.

## Key Features

- **Multi-format document processing:** Supports PDF, JPG, JPEG, PNG, TIFF, XLSX, XLS, and CSV files.
- **OCR:** Extracts text from scanned PDFs and images using EasyOCR.
- **PDF processing:** Supports digital PDFs, scanned PDFs, and mixed-content documents.
- **Spreadsheet processing:** Extracts structured data from Excel workbooks and CSV files.
- **Invoice information extraction:** Uses an LLM to extract structured invoice fields.
- **Financial validation:** Checks invoice totals, subtotals, taxes, line items, and other financial values.
- **Duplicate detection:** Identifies potentially duplicate invoices.
- **Evidence-based extraction:** Retains source evidence and extraction metadata for review.
- **Asynchronous processing:** Uses Celery and Redis to process uploaded documents in the background.
- **Document search:** Supports keyword-based and semantic search.
- **Document question answering:** Uses a LangGraph-based RAG pipeline to answer questions using document content.
- **Invoice correction:** Allows extracted invoice information to be reviewed and corrected.
- **Processing recovery:** Uses processing leases and heartbeat tracking to support recovery of stale processing tasks.

## Technology Stack

| Component | Technology |
|---|---|
| Backend | Python, FastAPI |
| AI Workflow Orchestration | LangGraph |
| LLM Integration | LangChain, Groq |
| OCR | EasyOCR |
| PDF Processing | PyMuPDF, pdfplumber |
| Image Processing | Pillow |
| Spreadsheet Processing | Pandas, OpenPyXL, xlrd |
| Database | PostgreSQL |
| Vector Search | pgvector |
| Embeddings | FastEmbed |
| ORM | SQLAlchemy |
| Database Migrations | Alembic |
| Background Processing | Celery |
| Message Broker | Redis |
| Frontend | React, TypeScript |
| Frontend Build Tool | Vite |
| Testing | pytest |

## Project Structure

```text
ai-document-intelligence/
├── app/
│   ├── agent/
│   │   ├── graph.py
│   │   ├── nodes.py
│   │   ├── qa_graph.py
│   │   ├── qa_nodes.py
│   │   ├── qa_state.py
│   │   ├── state.py
│   │   └── validation_node.py
│   ├── api/
│   │   ├── routes/
│   │   │   ├── documents.py
│   │   │   └── invoices.py
│   │   └── schemas.py
│   ├── database/
│   │   ├── embeddings.py
│   │   ├── models.py
│   │   ├── persistence.py
│   │   ├── repository.py
│   │   ├── search.py
│   │   └── session.py
│   ├── document_processing/
│   │   ├── __init__.py
│   │   ├── chunker.py
│   │   ├── csv_processor.py
│   │   ├── dispatcher.py
│   │   ├── excel_processor.py
│   │   ├── image_processor.py
│   │   ├── models.py
│   │   ├── ocr_engine.py
│   │   └── pdf_processor.py
│   ├── extraction/
│   │   ├── llm.py
│   │   └── schemas.py
│   ├── validation/
│   │   └── validator.py
│   ├── workers/
│   │   ├── celery_app.py
│   │   └── tasks.py
│   ├── __init__.py
│   └── main.py
├── frontend/
│   ├── public/
│   │   ├── favicon.svg
│   │   └── icons.svg
│   ├── src/
│   │   ├── assets/
│   │   ├── components/
│   │   │   ├── DocumentSearch.tsx
│   │   │   ├── DocumentUpload.tsx
│   │   │   ├── InvoiceResult.tsx
│   │   │   └── ProcessingStatus.tsx
│   │   ├── services/
│   │   │   └── api.ts
│   │   ├── App.css
│   │   ├── App.tsx
│   │   ├── index.css
│   │   └── main.tsx
│   ├── .gitignore
│   ├── .oxlintrc.json
│   ├── index.html
│   ├── package.json
│   ├── package-lock.json
│   ├── README.md
│   ├── tsconfig.app.json
│   ├── tsconfig.json
│   ├── tsconfig.node.json
│   └── vite.config.ts
├── migrations/
│   ├── versions/
│   │   ├── 15e34abf93f1_add_document_processing_heartbeat.py
│   │   ├── 27b9177ff6e6_create_document_and_invoice_tables.py
│   │   ├── 6c4a2f8d91b3_add_document_processing_lease.py
│   │   ├── 741fb32bc809_add_document_storage_path.py
│   │   ├── c113f8376720_add_invoice_extraction_evidence_fields.py
│   │   └── ddc9a4fd2c98_add_document_chunks.py
│   ├── env.py
│   ├── README
│   └── script.py.mako
├── scripts/
│   ├── init_database.py
│   ├── test_database.py
│   └── test_extraction.py
├── src/
│   └── ai_document_intelligence/
│       └── __init__.py
├── storage/
│   └── uploads/
├── tests/
│   ├── conftest.py
│   ├── test_api_documents.py
│   ├── test_api_fixtures.py
│   ├── test_api_invoices.py
│   ├── test_chunker.py
│   ├── test_document_processing.py
│   ├── test_image_processor.py
│   ├── test_invoice_validation.py
│   ├── test_main.py
│   ├── test_qa_pipeline.py
│   ├── test_repository.py
│   ├── test_search.py
│   └── test_worker_tasks.py
├── .gitignore
├── .python-version
├── alembic.ini
├── docker-compose.yml
├── pyproject.toml
├── README.md
├── test_processing.py
└── uv.lock
```

## Directory Overview

| Directory | Description |
|---|---|
| `app/agent/` | LangGraph workflows for document processing, invoice extraction, validation, and question answering. |
| `app/api/` | FastAPI application routes and API schemas. |
| `app/database/` | Database models, persistence, repository operations, embeddings, and search functionality. |
| `app/document_processing/` | Document ingestion, PDF processing, image processing, OCR, spreadsheet parsing, and text chunking. |
| `app/extraction/` | LLM-based invoice extraction and structured extraction schemas. |
| `app/validation/` | Invoice and financial data validation. |
| `app/workers/` | Celery application configuration and asynchronous processing tasks. |
| `frontend/` | React and TypeScript user interface. |
| `migrations/` | Alembic database migration history. |
| `scripts/` | Database initialization and utility scripts. |
| `src/ai_document_intelligence/` | Python package namespace. |
| `storage/uploads/` | Storage location for uploaded documents. |
| `tests/` | Automated backend tests. |

## Prerequisites

Install the following before running the application:

- Python 3.12 or later
- [uv](https://docs.astral.sh/uv/)
- Node.js and npm
- Docker Desktop
- A Groq API key

## Installation and Setup

### 1. Clone the repository

```bash
git clone <repository-url>
cd ai-document-intelligence
```

### 2. Configure environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=your_supported_groq_model

DATABASE_URL=postgresql+psycopg://invoice_user:invoice_password@localhost:5432/invoice_intelligence

REDIS_URL=redis://localhost:6379/0

STORAGE_DIR=storage/uploads
MAX_UPLOAD_SIZE_MB=25
STALE_PROCESSING_MINUTES=30
```

Replace the Groq placeholders with your actual credentials and supported model. Adjust the upload-size setting according to your application configuration.

The database credentials must match the PostgreSQL settings in `docker-compose.yml`.

**Important:** Never commit your `.env` file or expose API keys and database credentials.

### 3. Start PostgreSQL and Redis

Start the database and Redis services using Docker Compose:

```bash
docker compose up -d
```

Check their status:

```bash
docker compose ps
```

### 4. Install Python dependencies

From the project root:

```bash
uv sync
```

### 5. Apply database migrations

Initialize the database schema:

```bash
uv run alembic upgrade head
```

### 6. Start the FastAPI backend

```bash
uv run uvicorn app.main:app --reload
```

The backend will be available at:

```text
http://localhost:8000
```

Interactive API documentation:

```text
http://localhost:8000/docs
```

### 7. Start the Celery worker

Open another terminal in the project root:

```bash
uv run celery -A app.workers.celery_app.celery_app worker --loglevel=info --pool=solo
```

The `solo` pool is recommended for local development on Windows.

### 8. Start Celery Beat

Open another terminal in the project root:

```bash
uv run celery -A app.workers.celery_app.celery_app beat --loglevel=info
```

Celery Beat schedules periodic stale-document recovery tasks.

### 9. Configure and start the frontend

Open another terminal:

```bash
cd frontend
```

Create `frontend/.env`:

```env
VITE_API_BASE_URL=http://localhost:8000
```

Install frontend dependencies:

```bash
npm ci
```

Start the development server:

```bash
npm run dev
```

The frontend will be available at:

```text
http://localhost:5173
```

## Application Workflow

1. Upload a supported document through the frontend or API.
2. The backend stores the uploaded document and schedules background processing.
3. Celery processes the document according to its file type.
4. Digital PDFs are processed for embedded text, while scanned PDFs and images use OCR.
5. Excel and CSV files are parsed into structured content.
6. The extraction workflow uses an LLM to identify invoice information.
7. Extracted invoice data is validated, with errors and warnings recorded for review.
8. Document content is stored and indexed for keyword and semantic search.
9. Users can review and correct extracted invoice information.
10. The question-answering pipeline retrieves relevant document content and generates evidence-based answers.

## API Capabilities

The backend provides endpoints for:

| Capability | Description |
|---|---|
| Document upload | Upload supported documents for processing. |
| Processing status | Check the current processing status of a document. |
| Document result | Retrieve extracted document and invoice information. |
| Document search | Search indexed content using keyword or semantic search. |
| Document question answering | Ask questions about a processed document. |
| Invoice correction | Update extracted invoice fields. |

For the exact endpoint paths, request parameters, and response schemas, use the interactive API documentation at:

`http://localhost:8000/docs`

## Testing

### Run the backend test suite

From the project root:

```bash
uv run pytest -v
```

### Run tests with coverage

```bash
uv run pytest --cov=app --cov-report=term-missing
```

### Run a specific test module

For example:

```bash
uv run pytest tests/test_qa_pipeline.py -v
```

### Run frontend linting

From the `frontend/` directory:

```bash
npm run lint
```

### Build the frontend

```bash
npm run build
```

## Database Migrations

Create a new migration after modifying database models:

```bash
uv run alembic revision --autogenerate -m "describe_schema_change"
```

Apply migrations:

```bash
uv run alembic upgrade head
```

Check the current migration revision:

```bash
uv run alembic current
```

## Stopping the Application

Stop the FastAPI server, Celery worker, and Celery Beat using `Ctrl+C` in their respective terminals.

Stop PostgreSQL and Redis:

```bash
docker compose down
```

To remove the persistent database and Redis volumes as well:

```bash
docker compose down -v
```

**Warning:** The `-v` option deletes the persisted database and Redis data.

## Notes and Limitations

- OCR accuracy depends on document quality, resolution, language, and layout.
- Extracted invoice information should be reviewed when validation errors, warnings, or low-confidence fields are present.
- LLM extraction is designed to use source evidence and avoid inventing missing values; generated results should still be verified against the original document.
- Semantic search similarity scores are retrieval measures, not probabilities or guarantees of factual correctness.
- Large or complex documents may require additional processing time and system resources.
- The frontend currently uploads documents individually rather than through a multi-file batch selection interface.

## Security Considerations

- Keep API keys and credentials in environment variables.
- Do not commit `.env` files or uploaded confidential documents.
- Use appropriate database credentials and access controls in production.
- Configure CORS origins and upload limits for the deployment environment.
- Review extracted financial information before using it for business decisions.

## License

Add the applicable license information before distributing this project.