from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


@patch("app.api.endpoints.documents.ingest_documents")
def test_document_ingest_endpoint_success(mock_ingest):
    mock_ingest.return_value = {
        "total_files": 1,
        "ingested_files": 1,
        "failed_files": 0,
        "total_chunks": 3,
        "results": [
            {
                "status": "ingested",
                "filename": "case.txt",
                "document_type": "txt",
                "document_id": "doc-1",
                "chunks": 3,
                "detail": "Ingested successfully.",
            }
        ],
    }

    response = client.post(
        "/api/documents/ingest",
        files=[("files", ("case.txt", b"some legal text", "text/plain"))],
    )

    assert response.status_code == 200
    data = response.json()
    assert data["ingested_files"] == 1
    assert data["total_chunks"] == 3


@patch("app.api.endpoints.documents.ingest_documents")
def test_document_ingest_endpoint_all_failed(mock_ingest):
    mock_ingest.return_value = {
        "total_files": 1,
        "ingested_files": 0,
        "failed_files": 1,
        "total_chunks": 0,
        "results": [
            {
                "status": "failed",
                "filename": "bad.bin",
                "document_type": None,
                "document_id": None,
                "chunks": 0,
                "detail": "Unsupported file type.",
            }
        ],
    }

    response = client.post(
        "/api/documents/ingest",
        files=[("files", ("bad.bin", b"\x00\x01", "application/octet-stream"))],
    )

    assert response.status_code == 400
    assert "No documents were ingested." in response.text


@patch("app.api.endpoints.reports.build_report_export")
def test_report_export_endpoint_pdf(mock_export):
    mock_export.return_value = (b"%PDF-1.4 fake", "application/pdf", "report.pdf")

    response = client.post(
        "/api/reports/export",
        json={"title": "Test Report", "content": "Body", "format": "pdf"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert "attachment; filename=\"report.pdf\"" == response.headers.get(
        "content-disposition"
    )
