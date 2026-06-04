"""FastAPI application entrypoint."""

from fastapi import FastAPI

from agent.config import get_settings
from api.health import router as health_router
from api.middleware import ApiKeyMiddleware, RequestIdMiddleware, TimingMiddleware
from api.routes import router as screening_router

settings = get_settings()

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
