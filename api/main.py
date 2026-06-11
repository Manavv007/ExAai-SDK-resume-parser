"""FastAPI application entrypoint."""

from fastapi import FastAPI

from agent.config import get_settings
from agent.logging_config import configure_logging
from api.health import router as health_router
from api.middleware import ApiKeyMiddleware, RequestIdMiddleware, TimingMiddleware
from api.routes import router as screening_router

settings = get_settings()
configure_logging(settings.log_level)

app = FastAPI(
    title="EXAai-ADK",
    description=(
        "Resume screening agent with Exa enrichment and Gemini scoring.\n\n"
        "**POST /screen in Swagger UI:** set **api_key**, upload **resume** (PDF/DOCX/txt), "
        "paste the job description into **jd_text**. Leave any removed **jd** file field "
        "empty — use jd_text only in /docs. Production clients may use Bearer auth."
    ),
    version=settings.agent_version,
    swagger_ui_parameters={"persistAuthorization": True},
)

app.add_middleware(TimingMiddleware)
app.add_middleware(RequestIdMiddleware)
app.add_middleware(ApiKeyMiddleware)
app.include_router(health_router)
app.include_router(screening_router)


@app.on_event("startup")
def _resync_env_on_startup() -> None:
    """Re-apply .env after reload so a saved .env edit is picked up without a manual restart."""
    get_settings()
