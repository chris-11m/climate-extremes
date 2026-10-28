import numpy as np

from climate_extremes.data import load_era5_years
from climate_extremes.features import extract_temperature_point, build_forecast_rows, temporal_split, standardize_features


def test_extract_temperature_point():
    datasets = load_era5_years("data", [2010])

    original_k = datasets[0]["t2m"].isel(
        latitude=20,
        longitude=30,
    )

    point_c = extract_temperature_point(
        datasets[0],
        latitude_index=20,
        longitude_index=30,
    )

    # Correct number of hourly JJA observations
    assert point_c.shape == (2208,)

    # Correct grid point
    assert float(point_c.latitude) == 55.0
    assert float(point_c.longitude) == -4.5

    # Correct units metadata
    assert point_c.attrs["units"] == "degC"

    # Actually verify the Kelvin -> Celsius conversion
    assert np.allclose(
        point_c.values,
        original_k.values - 273.15,
    )


def test_build_forecast_rows():
    datasets = load_era5_years("data", [2010])

    point_c = extract_temperature_point(
        datasets[0],
        latitude_index=20,
        longitude_index=30,
    )

    threshold = 18.648992919921877

    rows = build_forecast_rows(
        point_c,
        threshold,
        lag_hours=24,
    )

    assert len(rows) == 2160

    assert rows["forecast_time"].iloc[0] == point_c["valid_time"].values[24]
    assert rows["forecast_time"].iloc[-1] == point_c["valid_time"].values[-25]

    assert np.allclose(
        rows["temperature_yesterday"].values,
        point_c.values[:-48],
    )

    assert np.allclose(
        rows["temperature_now"].values,
        point_c.values[24:-24],
    )

    expected_target = (
        point_c.values[48:] > threshold
    ).astype(int)

    assert np.array_equal(
        rows["extreme_24h"].values,
        expected_target,
    )

def test_temporal_split():
    datasets = load_era5_years("data", range(2010,2021))
    threshold = 18.648

    from climate_extremes.features import build_forecast_dataset 

    df = build_forecast_dataset(datasets, threshold , latitude_index=20, longitude_index=30, lag_hours=24)

    train_df, val_df, test_df = temporal_split(df)

    # Expected sample counts
    assert len(train_df) == 19440
    assert len(val_df) == 2160
    assert len(test_df) == 2160

    # Make sure years went into the correct partitions
    assert train_df["year"].min() == 2010
    assert train_df["year"].max() == 2018

    assert val_df["year"].unique().tolist() == [2019]
    assert test_df["year"].unique().tolist() == [2020]

def test_standardization_uses_training_statistics():
    datasets = load_era5_years(
        "data",
        range(2010, 2021),
    )

    threshold = 18.648

    from climate_extremes.features import build_forecast_dataset

    df = build_forecast_dataset(
        datasets,
        threshold,
        latitude_index=20,
        longitude_index=30,
        lag_hours=24,
    )

    train_df, val_df, test_df = temporal_split(df)

    train_scaled, val_scaled, test_scaled, stats = (
        standardize_features(
            train_df,
            val_df,
            test_df,
            features=["temperature_now"],
        )
    )

    train_mean = train_df["temperature_now"].mean()
    train_std = train_df["temperature_now"].std()

    assert np.isclose(
        stats["temperature_now"]["mean"],
        train_mean,
    )

    assert np.isclose(
        stats["temperature_now"]["std"],
        train_std,
    )

    # Training data should be centred around zero.
    assert np.isclose(
        train_scaled["temperature_now_z"].mean(),
        0.0,
        atol=1e-6,
    )

    # Validation must use TRAIN statistics.
    expected_val = (
        val_df["temperature_now"] - train_mean
    ) / train_std

    assert np.allclose(
        val_scaled["temperature_now_z"],
        expected_val,
    )

    # Test must also use TRAIN statistics.
    expected_test = (
        test_df["temperature_now"] - train_mean
    ) / train_std

    assert np.allclose(
        test_scaled["temperature_now_z"],
        expected_test,
    )
