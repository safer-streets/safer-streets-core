import typer

from safer_streets_core.file_storage import AzureBlobStorage, UpdatePolicy
from safer_streets_core.utils import blob_storage_url, data_dir

app = typer.Typer()

files_to_skip = (
    "police_uk_crime_data_latest.zip",
    "test.duckdb",
    "assault-by-LSOA-and-month.csv",
    "assault-by-LSOA.csv",
    "hex_crime_dataset2_2025-03.parquet",
    "great-britain-260326.osm.pbf",
)


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
        - The container name is currently hardcoded; the account URL comes from SAFER_STREETS_BLOB_STORAGE.
        - Recursively uploads all files in the data directory, skipping directories and files listed in `files_to_skip`.
        - Prints the status of each file (skipped, uploaded, or error).
        - Overwriting behavior is controlled by the `update` policy.
        - TODO: Make container configurable.
        - TODO: Add support for wildcards in file selection.
    """

    # TODO make configurable?
    container = "safer-streets-data"

    azure_storage = AzureBlobStorage(blob_storage_url(), container)

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

        if azure_storage.needs_update(src, name, update):
            result = azure_storage.write_file(src, name, overwrite=True)
            print(f"{name} {'uploaded' if result else 'error'}")


# without this the subcommand sync disappears
@app.callback()
def _callback():
    pass


def main() -> None:
    app()


if __name__ == "__main__":
    main()
