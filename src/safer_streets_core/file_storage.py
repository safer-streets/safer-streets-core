from io import BytesIO
from pathlib import Path
from typing import Protocol

from azure.identity import DefaultAzureCredential
from azure.storage.blob import ContainerClient
from itrx import Itr

from safer_streets_core.utils import data_dir

# TODO? async https://learn.microsoft.com/en-us/azure/storage/blobs/storage-blob-upload-python#upload-blobs-asynchronously


class DataSource(Protocol):
    def list(self, startswith: str | None = None) -> Itr[str]: ...

    def read(self, filename: str) -> BytesIO: ...

    def write_file(self, root_path: Path, filename: str, *, overwrite: bool = False) -> bool: ...

    def delete_file(self, filename: str) -> bool: ...

    def write_buffer(self, data: BytesIO, name: str): ...


class LocalFileStorage:
    """
    Interchangeable wrapper for accessing local storage
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or data_dir()

    def list(self, startswith: str | None = None) -> Itr[str]:
        pattern = f"{startswith}*" if startswith else "**/*"
        return Itr(self._path.glob(pattern)).map(lambda f: f.name)

    def read(self, filename: str) -> BytesIO:
        with open(self._path / filename, "rb") as fd:
            return BytesIO(fd.read())

    def write_file(self, root_path: Path, filename: str, *, overwrite: bool = False) -> bool:
        raise NotImplementedError("LocalFileStorage only supports readonly access")

    def delete_file(self, filename: str) -> bool:
        raise NotImplementedError("LocalFileStorage only supports readonly access")

    def write_buffer(self, data: BytesIO, name: str):
        raise NotImplementedError("LocalFileStorage only supports readonly access")


class AzureBlobStorage:
    """
    Uses a service principal, credentials should be stored in .env:
    AZURE_CLIENT_ID
    AZURE_CLIENT_SECRET
    AZURE_TENANT_ID
    """

    def __init__(self, account_url: str, container: str, readonly: bool = True) -> None:
        self._credential = DefaultAzureCredential()
        self._client = ContainerClient(account_url, container, self._credential)

    def list(self, startswith: str | None = None) -> Itr[str]:
        return Itr(self._client.list_blobs(name_starts_with=startswith)).map(lambda f: f.name)

    def read(self, filename: str) -> BytesIO:
        return BytesIO(self._client.download_blob(filename).readall())

    def write_file(self, root_path: Path, filename: str, *, overwrite: bool = False) -> bool:
        """
        Write file to azure (path in container will be filename)
        """
        blob_client = self._client.get_blob_client(filename)
        if not overwrite and blob_client.exists():
            return False
        with open(root_path / filename, "rb") as fd:
            blob_client.upload_blob(fd, overwrite=overwrite)
        return True

    def delete_file(self, filename: str) -> bool:
        blob_client = self._client.get_blob_client(filename)
        if not blob_client.exists():
            return False
        blob_client.delete_blob()
        return True

    def write_buffer(self, data: BytesIO, name: str):
        raise NotImplementedError()
