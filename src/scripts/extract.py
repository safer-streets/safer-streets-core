from io import BytesIO
from pathlib import Path
from typing import Annotated
from zipfile import ZipFile

import pandas as pd
import requests
import typer
from tqdm import tqdm

from safer_streets_core.database import duckdb_context
from safer_streets_core.utils import archive_path, data_dir, database_path, latest_month  # ty:ignore[deprecated]

# This script extracts street-level crime data from zipped CSV files and saves them as Parquet files.
# or creates a duckdb table
# archives can be downloaded from https://data.police.uk/data/ (see also download_archive in utils.py)

OUT_PATH = data_dir() / "extracted"

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


# TODO deprecate in favour of DB
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
def to_database(db_path: Path | None = None, force_download: bool = False) -> None:
    # db_path defaults to the standard database (database_path()); the orchestrator passes an
    # explicit staging path so output can be redirected without touching the live database.
    db_path = db_path or database_path()

    # cache the bulk archive under data_dir() so it can be reused across runs
    archive = archive_path("latest")
    if force_download or not archive.exists():
        # extract the latest archive and overwrite any existing data
        zipfile = "https://data.police.uk/data/archive/latest.zip"
        print(f"Downloading {zipfile} to {archive}...")
        response = requests.get(zipfile, stream=True)
        file_size = int(response.headers.get("content-length", 0))

        with open(archive, "wb") as fd, tqdm(total=file_size, unit="B", unit_scale=True) as progress_bar:
            for data in response.iter_content(1024**2):
                progress_bar.update(len(data))
                fd.write(data)
    else:
        print(f"Using cached {archive}...")

    print(f"Creating table crime_data in {db_path}...")

    with duckdb_context(db_path, writeable=True) as con:
        con.execute("INSTALL zipfs FROM community;LOAD zipfs;")

        # limited support for **/ glob, but ????-?? is a reasonable workaround
        con.execute(f"""
        DROP TABLE IF EXISTS crime_data;
        CREATE TABLE crime_data AS
        SELECT * FROM read_csv('zip://{archive}/????-??/*-street.csv', normalize_names = true);
        ALTER TABLE crime_data ADD COLUMN geom GEOMETRY;
        UPDATE crime_data
        SET geom = ST_Transform(
                ST_Point(longitude, latitude),
                'EPSG:4326',
                'EPSG:27700',
                always_xy := true
            )
        -- WHERE latitude IS NOT NULL AND longitude IS NOT NULL;
        """)


@app.command()
def summary(years: Annotated[int, typer.Option(min=1)] = 4) -> None:
    year_itr = range(latest_month().year, latest_month().year - years, -1)  # ty:ignore[deprecated]
    results = []
    for year in year_itr:
        data = []
        pattern = f"{year}*-street.parquet"
        for file in (data_dir() / "extracted").glob(pattern):
            month = file.name[:7]
            force = file.name[8:].replace("-street.parquet", "")
            data.append((month, force))
        summary = pd.DataFrame(index=pd.MultiIndex.from_tuples(data), data={"ok": True})
        results.append(summary.unstack(level=1, fill_value=False).sum().rename(year))
    typer.echo(pd.concat(results, axis=1).fillna(0).astype(int))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
