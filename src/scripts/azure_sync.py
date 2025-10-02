import typer
from itrx import Itr

from safer_streets_core.azure_blob_storage import AzureBlobStorage
from safer_streets_core.utils import data_dir


app = typer.Typer()

files_to_skip = (
    "UK_OAC_Final.csv", # possibly safeguarded
    "test.duckdb",
    "police_uk_crime_data_latest.zip"
)



@app.command("sync")
def sync(*, keep_existing: bool = True) -> None:
    """
    Synchronizes local files from the data directory to an Azure Blob Storage container.

    Args:
        keep_existing (bool, optional): If True, existing files in Azure Blob Storage will not be overwritten.
            Defaults to True.

    Notes:
        - The Azure Blob Storage account URL and container name are currently hardcoded.
        - The function uploads all files found recursively in the data directory, skipping directories and files listed in `files_to_skip`.
        - Prints the status of each file (skipped, uploaded, or exists).
        - Overwriting behaviour is controlled by the `keep_existing` flag.
        - TODOs:
            - Make account URL and container configurable.
            - Add support for wildcards in file selection.
    """
    # TODO make configurable?
    account_url = "https://saferstreets.blob.core.windows.net"
    container = "safer-streets-data"

    azure_storage = AzureBlobStorage(account_url, container)

    src = data_dir()

    # TODO wildcards?
    files = src.glob("**/*")

    for file in files:
        name = str(file.relative_to(data_dir()))
        if file.is_dir():
            continue
        if name in files_to_skip:
            print(f"{name} skipped")
            continue
        result = azure_storage.write_file(src, name, overwrite=not keep_existing)
        print(f"{name} {'uploaded' if result else 'exists'}")


# without this the subcommand sync disappears
@app.callback()
def _callback():
    pass

def main() -> None:
    app()


if __name__ == "__main__":
    main()
