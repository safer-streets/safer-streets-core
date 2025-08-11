from pathlib import Path
from zipfile import ZipFile

import pandas as pd

# This script extracts street-level crime data from zipped CSV files and saves them as Parquet files.
# archives can be downloaded from https://data.police.uk/data/ (see also download_archive in utils.py)

OUT_PATH = Path("./data/extracted/")


def extract_all() -> None:
    # this extracts all street-level crime data in reverse order - if a file already exists, it will be newer
    # so it is skipped
    for zipfile in sorted(Path("./data/").glob("police_uk_crime_data_*.zip"), reverse=True):
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


def extract_latest(*, keep_existing: bool = False) -> None:
    zipfile = "./data/police_uk_crime_data_latest.zip"
    print(f"Extracting {zipfile}...")
    with ZipFile(zipfile) as bulk_data:
        for file in bulk_data.namelist():
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


def extract_summary() -> pd.DataFrame:
    data = []
    for file in Path("./data/extracted").glob("*-street.parquet"):
        print(file.name)
        month = file.name[:7]
        force = file.name[8:].replace("-street.parquet", "")
        data.append((month, force))
    return pd.DataFrame(index=pd.MultiIndex.from_tuples(data), data={"ok": True})


if __name__ == "__main__":
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    extract_latest(keep_existing=True)
    summary = extract_summary()
    print(summary.unstack(level=1).fillna(False).sum())
