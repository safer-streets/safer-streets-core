from io import BytesIO
from pathlib import Path

from azure.core.exceptions import ResourceExistsError
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobProperties, ContainerClient
from itrx import Itr


class AzureBlobStorage:
    """Uses a service principal"""

    def __init__(self, account_url: str, container: str) -> None:
        self._credential = DefaultAzureCredential()
        self._client = ContainerClient(account_url, container, self._credential)

    def list(self, startswith: str | None = None) -> Itr[BlobProperties]:
        return Itr(self._client.list_blobs(name_starts_with=startswith))

    def read(self, filename: str) -> BytesIO:
        return BytesIO(self._client.download_blob(filename).readall())

    def write_file(self, root_path: Path, filename: str) -> bool:
        """
        Write file to azure (path in container will be filename)
        """
        try:
            blob_client = self._client.get_blob_client(filename)
            with open(root_path / filename, "rb") as fd:
                blob_client.upload_blob(fd)
        except ResourceExistsError:
            return False
        return True

    def delete_file(self, filename: str) -> bool:
        raise NotImplementedError()

    # TODO update_file?

    def write_buffer(self, data: BytesIO, name: str):
        raise NotImplementedError()
