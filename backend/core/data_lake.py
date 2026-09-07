"""Data lake seam — arch §5.6. Local data/ vs S3 Parquet (SSE-KMS, per-org partitioned)."""

import os

from backend.utils.config import settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class LocalParquetLake:
    def __init__(self, base_path: str | None = None):
        self.base_path = base_path or "data"

    def extract(self, table: str) -> str:
        # local: dbt reads MySQL directly; this is a no-op seam so local tests pass
        logger.info(f"Local lake extract {table} -> {self.base_path}")
        return f"{self.base_path}/{table}.parquet"


class S3ParquetLake:
    def __init__(self, bucket: str | None = None):
        self.bucket = bucket or os.environ.get("S3_LAKE_BUCKET", "caliperlens-lake")

    def extract(self, table: str) -> str:
        # prod: MySQL -> S3 Parquet (versioned, per-org partitioned, SSE-KMS)
        # ponytail: stub — real extract uses ECS cron + dbt-duckdb with Postgres ATTACHed
        raise NotImplementedError("S3 lake not configured in local env")


def get_data_lake(backend: str | None = None):
    b = (backend or settings.vector_backend or "local").lower() if backend is None else backend.lower()
    # reuse DATA_LAKE_BACKEND if set, else infer from backend arg
    env_b = os.environ.get("DATA_LAKE_BACKEND", "").lower()
    if env_b == "s3" or b == "s3":
        return S3ParquetLake()
    return LocalParquetLake()
