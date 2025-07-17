from pathlib import Path
from zipfile import ZipFile

import pandas as pd

# This script extracts street-level crime data from zipped CSV files and saves them as Parquet files.
# archives can be downloaded from https://data.police.uk/data/ (see also download_archive in utils.py)


def main() -> None:
    out_path = Path("./data/extracted/")

    out_path.mkdir(parents=True, exist_ok=True)

    # this extracts all street-level crime data in reverse order - if a file already exists, it will be newer
    # so it is skipped
    for zipfile in sorted(Path("./data/").glob("police_uk_crime_data_*.zip"), reverse=True):
        print(f"Extracting {zipfile}...")
        with ZipFile(zipfile) as bulk_data:
            for file in bulk_data.namelist():
                outfile = out_path / file.split("/")[-1].replace(".csv", ".parquet")
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


if __name__ == "__main__":
    main()
