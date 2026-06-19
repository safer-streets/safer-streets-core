from enum import StrEnum
from hashlib import md5
from io import BytesIO
from pathlib import Path
from typing import Any, Protocol, cast

from azure.core.exceptions import ResourceNotFoundError
from azure.identity import DefaultAzureCredential
from azure.storage.blob import ContainerClient
from itrx import Itr

from safer_streets_core.utils import data_dir

# TODO? async https://learn.microsoft.com/en-us/azure/storage/blobs/storage-blob-upload-python#upload-blobs-asynchronously


class UpdatePolicy(StrEnum):
    """When a destination blob already exists, decide whether a local file replaces it."""

    IGNORE = "ignore"  # don't overwrite if destination exists
    NEWER = "newer"  # overwrite if the local file is newer than the remote one
    DIFFERENT = "different"  # overwrite if the md5 sums differ
    FORCE = "force"  # always overwrite


# User-metadata key under which an upload stashes the source file's mtime. Azure's own last_modified
# reflects the moment of upload, not the content's modification time, so we record the source mtime
# ourselves (cf. ``scp -p``) to make modification-time comparisons meaningful across uploads/machines.
SRC_MTIME_KEY = "src_mtime"


def blob_mtime(properties: Any) -> float:
    """The source file's mtime for a blob: the ``src_mtime`` stamped at upload when present, falling
    back to the blob's own last-modified time for blobs written before ``src_mtime`` was recorded."""
    src = (properties.metadata or {}).get(SRC_MTIME_KEY)
    return float(src) if src is not None else properties.last_modified.timestamp()


class DataSource(Protocol):
    def list(self, startswith: str | None = None) -> Itr[str]: ...

    def read(self, filename: str) -> BytesIO: ...

    def metadata(self, filename: str) -> Any: ...

    def write_file(
        self, root_path: Path, filename: str, *, overwrite: bool = False, metadata: dict[str, str] | None = None
    ) -> bool: ...

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

    def metadata(self, filename: str) -> Any:
        return (self._path / filename).stat()

    def write_file(
        self, root_path: Path, filename: str, *, overwrite: bool = False, metadata: dict[str, str] | None = None
    ) -> bool:
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
        return BytesIO(cast(bytes, self._client.download_blob(filename).readall()))

    def metadata(self, filename: str) -> Any:
        try:
            return self._client.get_blob_client(filename).get_blob_properties()
        except ResourceNotFoundError:
            return None

    def needs_update(self, root_path: Path, filename: str, policy: UpdatePolicy) -> bool:
        """Whether the local ``root_path / filename`` should be (re-)uploaded under ``policy``.

        A blob absent remotely is always uploaded; otherwise the remote blob's metadata decides:
        ``NEWER`` compares last-modified against the local mtime, ``DIFFERENT`` compares md5 sums,
        ``FORCE`` always re-uploads and ``IGNORE`` never does.
        """
        remote_meta = self.metadata(filename)
        if remote_meta is None:
            return True
        match policy:
            case UpdatePolicy.FORCE:
                return True
            case UpdatePolicy.NEWER:
                return blob_mtime(remote_meta) < (root_path / filename).stat().st_mtime
            case UpdatePolicy.DIFFERENT:
                with (root_path / filename).open("rb") as fd:
                    local_md5 = md5(fd.read()).digest()
                return local_md5 != remote_meta.content_settings.content_md5
            case _:  # IGNORE
                return False

    def write_file(
        self, root_path: Path, filename: str, *, overwrite: bool = False, metadata: dict[str, str] | None = None
    ) -> bool:
        """
        Write file to azure (path in container will be filename).

        The source file's mtime is recorded as blob metadata under ``SRC_MTIME_KEY`` (see ``blob_mtime``),
        preserving the content's modification time across the upload (cf. ``scp -p``); any ``metadata``
        passed by the caller is merged in and takes precedence.
        """
        blob_client = self._client.get_blob_client(filename)
        if not overwrite and blob_client.exists():
            return False
        src_path = root_path / filename
        meta = {SRC_MTIME_KEY: str(src_path.stat().st_mtime), **(metadata or {})}
        with open(src_path, "rb") as fd:
            blob_client.upload_blob(fd, overwrite=overwrite, metadata=meta)
        return True

    def delete_file(self, filename: str) -> bool:
        blob_client = self._client.get_blob_client(filename)
        if not blob_client.exists():
            return False
        blob_client.delete_blob()
        return True

    def write_buffer(self, data: BytesIO, name: str):
        raise NotImplementedError()
