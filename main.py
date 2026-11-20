import hmac
from contextlib import asynccontextmanager
from typing import Any
from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from intent_router import HistoryItem, IntentResponse, IntentRouter


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=("jarvis-server/.env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    jarvis_secret_token: str = Field(..., min_length=1)
    llm_api_key: str = Field(..., min_length=1)
    llm_base_url: str = Field(
        default="https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    llm_model: str = Field(default="gemini-3.8-flash")

    @field_validator("jarvis_secret_token", "llm_api_key")
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("Field must not be empty or whitespace only")
        return trimmed


settings = Settings()


class CommandRequest(BaseModel):
    message: str = Field(..., min_length=1)
    history: list[HistoryItem] = Field(default_factory=list)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Lifecycle startup: Initialize async OpenAI client & intent router
    app.state.router = IntentRouter(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        model=settings.llm_model,
    )
    try:
        yield
    finally:
        # Lifecycle shutdown: Gracefully close connection sessions
        await app.state.router.close()


app = FastAPI(
    title="JARVIS Server",
    description="HUD Terminal Backend & Intent Router",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS configuration allowing cross-origin requests from web HUD frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def verify_bearer_token(authorization: str | None = Header(default=None)) -> None:
    """Validate incoming Authorization: Bearer <token> with constant-time comparison."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid Bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    provided_token = authorization.removeprefix("Bearer ").strip()
    expected_token = settings.jarvis_secret_token

    if not hmac.compare_digest(
        provided_token.encode("utf-8"),
        expected_token.encode("utf-8"),
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe returning server health status."""
    return {"status": "online", "version": "1.0.0"}


@app.post(
    "/command",
    response_model=IntentResponse,
    dependencies=[Depends(verify_bearer_token)],
)
async def command(req: CommandRequest, request: Request) -> IntentResponse:
    """Analyze incoming voice/text command, route intent, and return normalized payload."""
    router: IntentRouter = request.app.state.router
    return await router.route(message=req.message, history=req.history)