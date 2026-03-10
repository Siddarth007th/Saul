from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from app.models.schemas import ReportExportRequest
from app.services.report_export import build_report_export

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/reports/export")
async def export_report(request: ReportExportRequest) -> Response:
    content = request.content.strip()
    if not content:
        raise HTTPException(status_code=400, detail="Report content cannot be empty.")

    try:
        payload, media_type, filename = build_report_export(
            title=request.title,
            content=content,
            fmt=request.format,
            filename=request.filename,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Report export failed")
        raise HTTPException(status_code=503, detail="Report export failed.") from exc

    return Response(
        content=payload,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
