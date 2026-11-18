
from app.database.models import Document


def test_db_session_fixture_can_persist_document(db_session):
    document = Document(
        filename="fixture_test.pdf",
        file_type="pdf",
        status="uploaded",
    )

    db_session.add(document)
    db_session.commit()

    saved_document = db_session.get(Document, document.id)

    assert saved_document is not None
    assert saved_document.filename == "fixture_test.pdf"


def test_db_session_fixture_rolls_back_between_tests(db_session):
    documents = db_session.query(Document).all()

    assert documents == []


def test_client_fixture_uses_test_database(client):
    response = client.get("/")

    assert response.status_code == 200
