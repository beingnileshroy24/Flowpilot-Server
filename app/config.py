from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PROJECT_NAME: str = "Nexucon FlowPilot Manager"
    MONGODB_URL: str = "mongodb://localhost:27017"
    DATABASE_NAME: str = "flowpilot"
    JWT_SECRET: str = "super-secret-key-change-in-production-123456789"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440
    API_V1_STR: str = "/api/v1"
    AI_ENGINE_URL: str = "http://127.0.0.1:8001"
    MLX_MODEL: str = "mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit"
    USE_MOCK_LLM: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

settings = Settings()
