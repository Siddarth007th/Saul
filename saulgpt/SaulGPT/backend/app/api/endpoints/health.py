import httpx
from fastapi import APIRouter

from app.core.config import settings

router = APIRouter()


@router.get("/health")
async def health_check():
    ollama_ok = False
    pgvector_ok = False

    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(f"{settings.OLLAMA_BASE_URL}/api/tags")
            ollama_ok = response.status_code == 200
    except Exception:
        ollama_ok = False

    try:
        import psycopg2

        conn = psycopg2.connect(settings.DATABASE_URL)
        conn.close()
        pgvector_ok = True
    except Exception:
        pgvector_ok = False

    return {"status": "ok", "ollama": ollama_ok, "pgvector": pgvector_ok}
