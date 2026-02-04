from enum import StrEnum
from hashlib import md5

import typer

from safer_streets_core.file_storage import AzureBlobStorage
from safer_streets_core.utils import data_dir

app = typer.Typer()

files_to_skip = (
    "police_uk_crime_data_latest.zip",
    "assault-by-LSOA-and-month.csv",
    "assault-by-LSOA.csv",
)


class UpdatePolicy(StrEnum):
    IGNORE = "ignore"  # don't overwrite if destination exists
    NEWER = "newer"  # overwrite if source is newer
    DIFFERENT = "different"  # overwrite if md5 sums differ
    FORCE = "force"  # always overwrite


@app.command("sync")
def sync(*, update: UpdatePolicy = UpdatePolicy.IGNORE) -> None:
    """
    Synchronizes local files from the data directory to an Azure Blob Storage container.

    Args:
        update (UpdatePolicy, optional): Policy for updating existing files on Azure Blob Storage.
            - UpdatePolicy.IGNORE: Skip files that already exist remotely (default).
            - UpdatePolicy.DIFFERENT: Upload files if their content differs (by MD5 hash).
            - UpdatePolicy.NEWER: Upload files if the local file is newer than the remote one.

    Notes:
        - The Azure Blob Storage account URL and container name are currently hardcoded.
        - Recursively uploads all files in the data directory, skipping directories and files listed in `files_to_skip`.
        - Prints the status of each file (skipped, uploaded, or error).
        - Overwriting behavior is controlled by the `update` policy.
        - TODO: Make account URL and container configurable.
        - TODO: Add support for wildcards in file selection.
    """

    # TODO make configurable?
    account_url = "https://saferstreets.blob.core.windows.net"
    container = "safer-streets-data"

    azure_storage = AzureBlobStorage(account_url, container)

    # files = LocalFileStorage().list()
    src = data_dir()

    # TODO wildcards?
    files = src.glob("**/*")

    for file in files:
        if file.is_dir():
            continue

        name = str(file.relative_to(data_dir()))
        if name in files_to_skip:
            print(f"{name} skipped")
            continue

        remote_meta = azure_storage.metadata(name)

        to_update = False
        if remote_meta is None:
            to_update = True
        elif update == UpdatePolicy.DIFFERENT:
            with file.open("rb") as fd:
                local_md5 = md5(fd.read()).digest()
            if local_md5 != remote_meta.content_settings.content_md5:
                to_update = True
        elif update == UpdatePolicy.NEWER and remote_meta.last_modified.timestamp() < file.stat().st_mtime:
            to_update = True

        if to_update:
            result = azure_storage.write_file(src, name, overwrite=to_update)
            print(f"{name} {'uploaded' if result else 'error'}")


# without this the subcommand sync disappears
@app.callback()
def _callback():
    pass


def main() -> None:
    app()


if __name__ == "__main__":
    main()
