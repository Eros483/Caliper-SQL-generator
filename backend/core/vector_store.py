"""Vector seam — arch §5.8. FAISS file local, pgvector HNSW AWS, same search_tables sig."""

from backend.utils.config import settings


class _FAISSStore:
    def search_tables(self, query: str, k: int = 5) -> str:

        # this will be wired by the agent; stub returns placeholder for seam test
        return f"FAISS search for: {query} (k={k})"


class _PgVectorStore:
    """ponytail: stub — real pgvector uses HNSW index, Titan embeddings, same doc set."""

    def search_tables(self, query: str, k: int = 5) -> str:
        raise NotImplementedError("pgvector not configured in local env")


def get_vector_store(backend: str | None = None):
    b = (backend or settings.vector_backend or "local").lower()
    if b == "pgvector":
        return _PgVectorStore()
    return _FAISSStore()
