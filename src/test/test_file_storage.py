from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest

from safer_streets_core.file_storage import SRC_MTIME_KEY, LocalFileStorage, blob_mtime


@pytest.fixture
def storage_dir(tmp_path: Path) -> Path:
    (tmp_path / "alpha.txt").write_bytes(b"hello")
    (tmp_path / "beta.txt").write_bytes(b"world")
    (tmp_path / "alpha.csv").write_bytes(b"a,b,c")
    return tmp_path


class TestLocalFileStorage:
    def test_defaults_to_data_dir(self, monkeypatch, tmp_path):
        monkeypatch.setattr("safer_streets_core.file_storage.data_dir", lambda: tmp_path)
        store = LocalFileStorage()
        assert store._path == tmp_path

    def test_list_all(self, storage_dir):
        store = LocalFileStorage(storage_dir)
        assert set(store.list().collect()) == {"alpha.txt", "beta.txt", "alpha.csv"}

    def test_list_with_prefix(self, storage_dir):
        store = LocalFileStorage(storage_dir)
        assert set(store.list("alpha").collect()) == {"alpha.txt", "alpha.csv"}

    def test_read(self, storage_dir):
        store = LocalFileStorage(storage_dir)
        buffer = store.read("alpha.txt")
        assert isinstance(buffer, BytesIO)
        assert buffer.read() == b"hello"

    def test_metadata(self, storage_dir):
        store = LocalFileStorage(storage_dir)
        meta = store.metadata("alpha.txt")
        assert meta.st_size == len(b"hello")

    def test_write_file_is_readonly(self, storage_dir):
        store = LocalFileStorage(storage_dir)
        with pytest.raises(NotImplementedError):
            store.write_file(storage_dir, "alpha.txt")

    def test_delete_file_is_readonly(self, storage_dir):
        store = LocalFileStorage(storage_dir)
        with pytest.raises(NotImplementedError):
            store.delete_file("alpha.txt")

    def test_write_buffer_is_readonly(self, storage_dir):
        store = LocalFileStorage(storage_dir)
        with pytest.raises(NotImplementedError):
            store.write_buffer(BytesIO(b"x"), "new.txt")


class TestBlobMtime:
    def test_prefers_recorded_src_mtime(self):
        # the blob carries the source file's mtime; its own upload time is later and ignored
        props = SimpleNamespace(
            metadata={SRC_MTIME_KEY: "5000.0"},
            last_modified=datetime.fromtimestamp(9999.0, tz=UTC),
        )
        assert blob_mtime(props) == 5000.0

    def test_falls_back_to_last_modified(self):
        # blobs uploaded before src_mtime was recorded have no such metadata
        props = SimpleNamespace(metadata={}, last_modified=datetime.fromtimestamp(8000.0, tz=UTC))
        assert blob_mtime(props) == 8000.0

    def test_falls_back_when_metadata_is_none(self):
        props = SimpleNamespace(metadata=None, last_modified=datetime.fromtimestamp(7000.0, tz=UTC))
        assert blob_mtime(props) == 7000.0
