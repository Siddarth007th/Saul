from __future__ import annotations

import logging

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.models.schemas import DocumentIngestionResponse
from app.services.document_ingestion import UploadedFilePayload, ingest_documents

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/documents/ingest", response_model=DocumentIngestionResponse)
async def ingest_documents_endpoint(
    files: list[UploadFile] = File(...),
) -> DocumentIngestionResponse:
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required.")

    payloads: list[UploadedFilePayload] = []
    for upload in files:
        if not upload.filename:
            raise HTTPException(status_code=400, detail="Each upload must include a filename.")

        payloads.append(
            UploadedFilePayload(
                filename=upload.filename,
                content_type=upload.content_type,
                data=await upload.read(),
            )
        )

    try:
        summary = ingest_documents(payloads)
    except Exception as exc:
        logger.exception("Document ingestion failed")
        raise HTTPException(status_code=503, detail="Document ingestion failed.") from exc

    if int(summary["ingested_files"]) == 0:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "No documents were ingested.",
                "results": summary["results"],
            },
        )

    return DocumentIngestionResponse.model_validate(summary)
