from fastapi import APIRouter

from agent.config import get_settings

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict[str, str]:
    settings = get_settings()
    from agent.llm_client import model_version_label, resolve_llm_provider

    return {
        "status": "ok",
        "agent_version": settings.agent_version,
        "screening_mode": settings.screening_mode,
        "llm_provider": resolve_llm_provider(settings),
        "model": model_version_label(settings),
    }
