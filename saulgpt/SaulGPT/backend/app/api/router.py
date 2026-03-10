from fastapi import APIRouter

from app.api.endpoints import chat, documents, health, reports, search

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(chat.router)
api_router.include_router(search.router)
api_router.include_router(documents.router)
api_router.include_router(reports.router)
