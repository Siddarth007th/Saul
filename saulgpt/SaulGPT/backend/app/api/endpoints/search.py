from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.dependencies import get_vector_store
from app.models.schemas import SearchRequest, SearchResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/search")
async def search(request: SearchRequest) -> SearchResponse:
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        store = get_vector_store()
        results = store.search_similar(
            request.query,
            request.top_k,
            request.filter_act,
        )
    except Exception as exc:
        exc_type = type(exc).__name__
        exc_module = type(exc).__module__ or ""

        # psycopg2.OperationalError → pgvector down
        if "OperationalError" in exc_type and "psycopg2" in exc_module:
            logger.error("pgvector connection error: %s", exc)
            raise HTTPException(
                status_code=503, detail="pgvector is unavailable"
            ) from exc

        logger.exception("Search failed")
        raise HTTPException(status_code=503, detail="Service unavailable") from exc

    return SearchResponse(results=results, query=request.query)
