from pathlib import Path
import xarray as xr


def load_era5_years(data_dir, years):
    """
    Load yearly ERA5 NetCDF files.

    Parameters
    ----------
    data_dir :
        Directory containing the ERA5 files.
    years :
        Years to load.

    Returns
    -------
    ...
    """

    datasets = []
    data_dir = Path(data_dir)

    for year in years:

        # 1. construct filepath safely using Path object
        
        output_path = data_dir / f"era5_t2m_uk_jja_{year}.nc"

        if not output_path.exists():
            raise FileNotFoundError(
                f"ERA5 file not found: {output_path}"
            )
        # 3. open it
        ds = xr.open_dataset(output_path)
        # 4. append dataset
        datasets.append(ds)

    return datasets


if __name__ == "__main__":
    datasets = load_era5_years("data", range(2010, 2021))

    print(len(datasets))
    print(datasets[0]["valid_time"].dt.year.values[0])
    print(datasets[-1]["valid_time"].dt.year.values[0])

    