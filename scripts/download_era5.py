from pathlib import Path
from tqdm import tqdm
import cdsapi


DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

DATASET = "reanalysis-era5-single-levels"


def download_era5_year(year):
    output_path = (
        DATA_DIR / f"era5_atmos_uk_jja_{year}.nc"
    )

    if output_path.exists():
        print(f"Already exists: {output_path}")
        return

    request = {
        "product_type": ["reanalysis"],
        "variable": [
            "2m_temperature",
            "2m_dewpoint_temperature",
            "mean_sea_level_pressure",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
        ],
        "year": [str(year)],
        "month": ["06", "07", "08"],
        "day": [
            f"{day:02d}"
            for day in range(1, 32)
        ],
        "time": [
            f"{hour:02d}:00"
            for hour in range(24)
        ],
        "area": [
            60,   # North
            -12,  # West
            48,   # South
            4,    # East
        ],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }

    print(f"Downloading ERA5 for {year}...")

    client = cdsapi.Client()

    client.retrieve(
        DATASET,
        request,
        str(output_path),
    )

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    years = range(2010, 2021)

    for year in tqdm(years, desc="ERA5 years"):
        download_era5_year(year)