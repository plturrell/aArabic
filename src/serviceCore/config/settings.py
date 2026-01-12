"""
Configuration management using environment variables
"""

import os
from pathlib import Path
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field, validator, AliasChoices


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Application
    app_name: str = Field(default="AI Nucleus Backend", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    debug: bool = Field(default=False, env="DEBUG")
    environment: str = Field(default="development", env="ENVIRONMENT")
    
    # Server
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    
    # CORS
    cors_origins: List[str] = Field(
        default=["http://localhost:8000", "http://127.0.0.1:8000"],
        env="CORS_ORIGINS"
    )
    cors_allow_credentials: bool = Field(default=True, env="CORS_ALLOW_CREDENTIALS")
    
    # Shimmy Local LLM Backend
    shimmy_backend_url: str = Field(
        default="http://127.0.0.1:11435",
        validation_alias=AliasChoices("SHIMMY_BACKEND_URL", "SHIMMY_URL")
    )

    # Default model for local LLM
    default_local_model: str = Field(
        default="glm4:9b",
        env="DEFAULT_LOCAL_MODEL"
    )
    
    # Memgraph (used by Rust backend)
    memgraph_host: str = Field(default="localhost", env="MEMGRAPH_HOST")
    memgraph_port: int = Field(default=7687, env="MEMGRAPH_PORT")
    memgraph_enabled: bool = Field(default=False, env="MEMGRAPH_ENABLED")
    
    # Model Paths
    models_dir: str = Field(
        default="./models/arabic_models",
        env="MODELS_DIR"
    )
    camelbert_path: Optional[str] = Field(default=None, env="CAMELBERT_PATH")
    m2m100_path: Optional[str] = Field(default=None, env="M2M100_PATH")
    
    # Security
    api_key: Optional[str] = Field(default=None, env="API_KEY")
    enable_auth: bool = Field(default=False, env="ENABLE_AUTH")
    rate_limit_per_minute: int = Field(default=60, env="RATE_LIMIT_PER_MINUTE")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")  # json or text
    
    # Workflow Storage
    workflow_storage_type: str = Field(default="memory", env="WORKFLOW_STORAGE_TYPE")  # memory or database
    workflow_db_url: Optional[str] = Field(default=None, env="WORKFLOW_DB_URL")
    
    @validator("cors_origins", pre=True)
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    @validator("camelbert_path", pre=True, always=True)
    def set_camelbert_path(cls, v, values):
        if v is None:
            models_dir = values.get("models_dir", "./models/arabic_models")
            return str(Path(models_dir) / "camelbert-dialect-financial")
        return v
    
    @validator("m2m100_path", pre=True, always=True)
    def set_m2m100_path(cls, v, values):
        if v is None:
            models_dir = values.get("models_dir", "./models/arabic_models")
            return str(Path(models_dir) / "m2m100-418M")
        return v
    
    @property
    def is_production(self) -> bool:
        return self.environment == "production"
    
    @property
    def is_development(self) -> bool:
        return self.environment == "development"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()
