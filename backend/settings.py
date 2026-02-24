from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    cors_origins: list[str] = ["http://localhost:5173"]
    max_workers: int = 4

    model_config = SettingsConfigDict(env_prefix="ALMA_", env_file=".env", extra="ignore")


settings = Settings()
