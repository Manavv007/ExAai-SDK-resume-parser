from fastapi import APIRouter

from agent.config import get_settings

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict[str, str]:
    settings = get_settings()
    return {
        "status": "ok",
        "agent_version": settings.agent_version,
        "model": settings.gemini_model_id,
    }
