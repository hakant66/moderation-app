# backend/config.py
from pydantic import BaseModel, Field, field_validator
from typing import List
import os

# Optional: python-dotenv support (loads .env if present)
try:
    from dotenv import load_dotenv
    load_dotenv()  # looks for .env upward from CWD
except Exception:
    pass

class Settings(BaseModel):
    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")
    torch_filter_url: str = Field("http://localhost:9000", alias="TORCH_FILTER_URL")
    filter_block: float = Field(0.97, alias="FILTER_BLOCK")
    filter_allow: float = Field(0.03, alias="FILTER_ALLOW")
    filter_early_allow: bool = Field(False, alias="FILTER_EARLY_ALLOW")
    cors_origins: List[str] = Field(["*"], alias="CORS_ORIGINS")
    policy_file: str = Field(default="backend/policy.yaml", alias="POLICY_FILE")

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_csv(cls, v):
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return v

def get_settings() -> Settings:
    # Pydantic reads from env by aliases; dotenv already loaded if present.
    values = {k: v for k, v in os.environ.items()}
    return Settings.model_validate(values)

settings = get_settings()
