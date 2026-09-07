"""Data path + vectors — arch §5.6/§5.8. S3 lake seam and pgvector seam, local parity."""


class TestDataLakeSeam:
    def test_data_lake_factory_exists(self):
        from backend.core.data_lake import get_data_lake

        local = get_data_lake(backend="local")
        assert hasattr(local, "extract")
        s3 = get_data_lake(backend="s3")
        assert hasattr(s3, "extract")

    def test_local_lake_uses_data_dir(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DATA_LAKE_BACKEND", "local")
        from backend.core.data_lake import LocalParquetLake

        lake = LocalParquetLake(base_path=str(tmp_path))
        assert lake.base_path == str(tmp_path)


class TestPgVectorSeam:
    def test_vector_factory_local(self):
        from backend.core.vector_store import get_vector_store

        store = get_vector_store(backend="local")
        assert hasattr(store, "search_tables")

    def test_vector_factory_pgvector_stub_same_sig(self):
        from backend.core.vector_store import get_vector_store

        store = get_vector_store(backend="pgvector")
        assert hasattr(store, "search_tables")
        import inspect

        sig = inspect.signature(store.search_tables)
        assert "query" in sig.parameters

    def test_search_tables_signature_preserved_in_rag(self):
        import inspect

        from backend.src.rag_manager import SchemaRAG

        sig = inspect.signature(SchemaRAG.search_tables)
        assert "query" in sig.parameters
        assert "k" in sig.parameters
