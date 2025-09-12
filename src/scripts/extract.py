from io import BytesIO
from zipfile import ZipFile

import pandas as pd
import requests
import typer
from tqdm import tqdm

from safer_streets_core.utils import data_dir

# This script extracts street-level crime data from zipped CSV files and saves them as Parquet files.
# archives can be downloaded from https://data.police.uk/data/ (see also download_archive in utils.py)

OUT_PATH = data_dir() / "extracted"
OUT_PATH.mkdir(parents=True, exist_ok=True)

app = typer.Typer()


@app.command(name="all")
def all_() -> None:
    # this extracts all street-level crime data in reverse order - if a file already exists, it will be newer
    # so it is skipped
    for zipfile in sorted(data_dir().glob("police_uk_crime_data_*.zip"), reverse=True):
        print(f"Extracting {zipfile}...")
        with ZipFile(zipfile) as bulk_data:
            for file in bulk_data.namelist():
                outfile = OUT_PATH / file.split("/")[-1].replace(".csv", ".parquet")
                if "street" not in file or outfile.exists():
                    continue
                (
                    pd.read_csv(bulk_data.open(file))
                    .set_index("Crime ID")
                    .drop(columns=["Last outcome category", "Context"])
                ).to_parquet(
                    outfile,
                    index=True,
                )


@app.command()
def latest(*, keep_existing: bool = False) -> None:
    # extract the latest archive and overwrite any existing data
    zipfile = "https://data.police.uk/data/archive/latest.zip"
    print(f"Downloading {zipfile}...")
    response = requests.get(zipfile, stream=True)
    file_size = int(response.headers.get("content-length", 0))
    buffer = BytesIO()

    with tqdm(total=file_size, unit="B", unit_scale=True) as progress_bar:
        # with open(zipfile, "wb") as fd:
        for data in response.iter_content(1024**2):
            progress_bar.update(len(data))
            buffer.write(data)

    print(f"Extracting {zipfile}...")
    with ZipFile(buffer) as bulk_data:
        for file in tqdm(bulk_data.namelist()):
            outfile = OUT_PATH / file.split("/")[-1].replace(".csv", ".parquet")
            if "street" not in file or (keep_existing and outfile.exists()):
                continue
            (
                pd.read_csv(bulk_data.open(file))
                .set_index("Crime ID")
                .drop(columns=["Last outcome category", "Context"])
            ).to_parquet(
                outfile,
                index=True,
            )


@app.command()
def summary() -> None:
    data = []
    for file in (data_dir() / "extracted").glob("*-street.parquet"):
        month = file.name[:7]
        force = file.name[8:].replace("-street.parquet", "")
        data.append((month, force))
    summary = pd.DataFrame(index=pd.MultiIndex.from_tuples(data), data={"ok": True})
    typer.echo(summary.unstack(level=1, fill_value=False).sum())


def main() -> None:
    app()


if __name__ == "__main__":
    main()
