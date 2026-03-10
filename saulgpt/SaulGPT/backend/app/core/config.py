from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "saulgpt"
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/saulgpt"
    EMBEDDING_MODEL: str = "BAAI/bge-base-en-v1.5"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
