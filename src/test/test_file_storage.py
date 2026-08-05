from datetime import UTC, datetime
from hashlib import md5
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from azure.core.exceptions import ResourceNotFoundError

from safer_streets_core.file_storage import (
    SRC_MTIME_KEY,
    AzureBlobStorage,
    LocalFileStorage,
    UpdatePolicy,
    blob_mtime,
)


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


class FakeBlobClient:
    def __init__(self, filename, *, exists=True, properties=None):
        self.filename = filename
        self._exists = exists
        self._properties = properties
        self.uploaded: dict[str, Any] | None = None
        self.deleted = False

    def exists(self):
        return self._exists

    def get_blob_properties(self):
        if not self._exists:
            raise ResourceNotFoundError("not found")
        return self._properties

    def upload_blob(self, fd, *, overwrite, metadata):
        self.uploaded = {"data": fd.read(), "overwrite": overwrite, "metadata": metadata}
        self._exists = True

    def delete_blob(self):
        self.deleted = True
        self._exists = False


class FakeContainerClient:
    def __init__(self, account_url, container, credential):
        self.account_url = account_url
        self.container = container
        self.credential = credential
        self.blobs = {}

    def list_blobs(self, name_starts_with=None):
        names = self.blobs if not name_starts_with else [n for n in self.blobs if n.startswith(name_starts_with)]
        return [SimpleNamespace(name=n) for n in names]

    def download_blob(self, filename):
        return SimpleNamespace(readall=lambda: self.blobs[filename])

    def get_blob_client(self, filename):
        return self.blobs.setdefault(f"__client__{filename}", FakeBlobClient(filename))


@pytest.fixture
def fake_client(monkeypatch):
    monkeypatch.setattr("safer_streets_core.file_storage.DefaultAzureCredential", lambda: "fake-credential")
    monkeypatch.setattr("safer_streets_core.file_storage.ContainerClient", FakeContainerClient)
    store = AzureBlobStorage("https://example.blob.core.windows.net", "mycontainer")
    return store


class TestAzureBlobStorage:
    def test_init_creates_client_with_credential(self, fake_client):
        assert fake_client._credential == "fake-credential"
        assert isinstance(fake_client._client, FakeContainerClient)
        assert fake_client._client.account_url == "https://example.blob.core.windows.net"
        assert fake_client._client.container == "mycontainer"

    def test_list(self, fake_client):
        fake_client._client.blobs = {"a.txt": b"", "b.txt": b""}
        assert set(fake_client.list().collect()) == {"a.txt", "b.txt"}

    def test_read(self, fake_client):
        fake_client._client.blobs["a.txt"] = b"hello"
        buffer = fake_client.read("a.txt")
        assert isinstance(buffer, BytesIO)
        assert buffer.read() == b"hello"

    def test_metadata_returns_properties_when_found(self, fake_client):
        props = SimpleNamespace(last_modified=datetime.fromtimestamp(1000.0, tz=UTC))
        fake_client._client.blobs["__client__a.txt"] = FakeBlobClient("a.txt", exists=True, properties=props)
        assert fake_client.metadata("a.txt") is props

    def test_metadata_returns_none_when_not_found(self, fake_client):
        fake_client._client.blobs["__client__missing.txt"] = FakeBlobClient("missing.txt", exists=False)
        assert fake_client.metadata("missing.txt") is None

    def test_needs_update_true_when_remote_missing(self, fake_client):
        fake_client._client.blobs["__client__new.txt"] = FakeBlobClient("new.txt", exists=False)
        assert fake_client.needs_update(Path("."), "new.txt", UpdatePolicy.IGNORE) is True

    def test_needs_update_force_always_true(self, fake_client):
        props = SimpleNamespace(metadata={}, last_modified=datetime.fromtimestamp(1.0, tz=UTC))
        fake_client._client.blobs["__client__a.txt"] = FakeBlobClient("a.txt", properties=props)
        assert fake_client.needs_update(Path("."), "a.txt", UpdatePolicy.FORCE) is True

    def test_needs_update_ignore_always_false(self, fake_client):
        props = SimpleNamespace(metadata={}, last_modified=datetime.fromtimestamp(1.0, tz=UTC))
        fake_client._client.blobs["__client__a.txt"] = FakeBlobClient("a.txt", properties=props)
        assert fake_client.needs_update(Path("."), "a.txt", UpdatePolicy.IGNORE) is False

    def test_needs_update_newer_compares_mtime(self, fake_client, tmp_path):
        local_file = tmp_path / "a.txt"
        local_file.write_bytes(b"hello")
        remote_older = SimpleNamespace(metadata={}, last_modified=datetime.fromtimestamp(1.0, tz=UTC))
        fake_client._client.blobs["__client__a.txt"] = FakeBlobClient("a.txt", properties=remote_older)
        assert fake_client.needs_update(tmp_path, "a.txt", UpdatePolicy.NEWER) is True

        remote_newer = SimpleNamespace(metadata={}, last_modified=datetime.fromtimestamp(9999999999.0, tz=UTC))
        fake_client._client.blobs["__client__a.txt"] = FakeBlobClient("a.txt", properties=remote_newer)
        assert fake_client.needs_update(tmp_path, "a.txt", UpdatePolicy.NEWER) is False

    def test_needs_update_different_compares_md5(self, fake_client, tmp_path):
        local_file = tmp_path / "a.txt"
        local_file.write_bytes(b"hello")
        matching = SimpleNamespace(content_settings=SimpleNamespace(content_md5=md5(b"hello").digest()))
        fake_client._client.blobs["__client__a.txt"] = FakeBlobClient("a.txt", properties=matching)
        assert fake_client.needs_update(tmp_path, "a.txt", UpdatePolicy.DIFFERENT) is False

        differing = SimpleNamespace(content_settings=SimpleNamespace(content_md5=md5(b"other").digest()))
        fake_client._client.blobs["__client__a.txt"] = FakeBlobClient("a.txt", properties=differing)
        assert fake_client.needs_update(tmp_path, "a.txt", UpdatePolicy.DIFFERENT) is True

    def test_write_file_skips_existing_without_overwrite(self, fake_client, tmp_path):
        (tmp_path / "a.txt").write_bytes(b"hello")
        blob_client = FakeBlobClient("a.txt", exists=True)
        fake_client._client.blobs["__client__a.txt"] = blob_client
        assert fake_client.write_file(tmp_path, "a.txt", overwrite=False) is False
        assert blob_client.uploaded is None

    def test_write_file_uploads_with_src_mtime_and_custom_metadata(self, fake_client, tmp_path):
        (tmp_path / "a.txt").write_bytes(b"hello")
        blob_client = FakeBlobClient("a.txt", exists=False)
        fake_client._client.blobs["__client__a.txt"] = blob_client
        result = fake_client.write_file(tmp_path, "a.txt", metadata={"custom": "value"})
        assert result is True
        assert blob_client.uploaded is not None
        assert blob_client.uploaded["data"] == b"hello"
        assert blob_client.uploaded["metadata"]["custom"] == "value"
        assert SRC_MTIME_KEY in blob_client.uploaded["metadata"]

    def test_write_file_overwrite_true_replaces_existing(self, fake_client, tmp_path):
        (tmp_path / "a.txt").write_bytes(b"new content")
        blob_client = FakeBlobClient("a.txt", exists=True)
        fake_client._client.blobs["__client__a.txt"] = blob_client
        result = fake_client.write_file(tmp_path, "a.txt", overwrite=True)
        assert result is True
        assert blob_client.uploaded is not None
        assert blob_client.uploaded["overwrite"] is True

    def test_delete_file_returns_false_when_absent(self, fake_client):
        blob_client = FakeBlobClient("missing.txt", exists=False)
        fake_client._client.blobs["__client__missing.txt"] = blob_client
        assert fake_client.delete_file("missing.txt") is False
        assert blob_client.deleted is False

    def test_delete_file_deletes_when_present(self, fake_client):
        blob_client = FakeBlobClient("a.txt", exists=True)
        fake_client._client.blobs["__client__a.txt"] = blob_client
        assert fake_client.delete_file("a.txt") is True
        assert blob_client.deleted is True

    def test_write_buffer_not_implemented(self, fake_client):
        with pytest.raises(NotImplementedError):
            fake_client.write_buffer(BytesIO(b"x"), "name.txt")
