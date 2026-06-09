from io import BytesIO
from pathlib import Path

import pytest

from safer_streets_core.file_storage import LocalFileStorage


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
