from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central management for settings and configurations."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    db_user: str = "root"
    db_password: str = ""
    db_host: str = "localhost"
    db_name: str = "fhs_coredb_local"

    duckdb_path: str = "./data/caliperlens.duckdb"

    gemini_api_key: str = ""
    langsmith_api_key: str = ""
    langchain_tracing_v2: bool = False
    langchain_project: str = "caliperlens"

    secret_key: str = "change-me-secret-key"

    demo_user: str = ""
    demo_password_hash: str = ""

    # parity seams — arch §10 local vs AWS
    env: str = "development"
    llm_backend: str = "local"
    auth_backend: str = "local"
    sandbox_backend: str = "local"
    vector_backend: str = "local"

    @property
    def database_uri(self) -> str:
        from urllib.parse import quote_plus

        user = quote_plus(self.db_user)
        pwd = quote_plus(self.db_password)
        name = quote_plus(self.db_name)
        return f"mysql+pymysql://{user}:{pwd}@{self.db_host}/{name}"


settings = Settings()
