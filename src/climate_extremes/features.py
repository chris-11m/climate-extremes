import xarray as xr
import pandas as pd
import numpy as np

def extract_temperature_point(ds, latitude_index, longitude_index):

    point = ds["t2m"].isel(latitude=latitude_index,longitude=longitude_index)
    point_c = point - 273.15

    point_c.attrs["units"] = "degC"

    return point_c
       


def compute_extreme_threshold(
    datasets,
    latitude_index,
    longitude_index,
    quantile=0.95,
    train_end_year=2018,):
    training_data = []

    for ds in datasets:

        point_c = extract_temperature_point(
            ds,latitude_index=latitude_index,longitude_index=longitude_index,)

        year = point_c.valid_time.dt.year.values[0]

        if year <= train_end_year:
            training_data.append(point_c)

    # concatenate training_data along valid_time
    training_temperature = xr.concat(
        training_data,
        dim="valid_time"
    )
    print("Number of training years:", len(training_data))
    print("Combined shape:", training_temperature.shape)

    for point in training_data:
        print(
            int(point.valid_time.dt.year.values[0]),
            point.shape,
            float(point.mean()),
            float(point.quantile(0.95)),
        )
    # calculate requested quantile
    threshold = training_temperature.quantile(quantile)

    return float(threshold)


def build_forecast_rows(point_c, threshold, lag_hours=24):
    """
    Build 24-hour-ahead extreme-temperature forecast samples
    for one continuous yearly temperature series.
    """

    n = len(point_c)

    temperature_yesterday = point_c.isel(valid_time=slice(0, n - 2*lag_hours))
    
    temperature_now = point_c.isel(valid_time=slice(lag_hours, n - lag_hours))
    temperature_future = point_c.isel(valid_time=slice(2*lag_hours, n))

    extreme_future =  (temperature_future > threshold).astype(int)

    df = pd.DataFrame({
        "forecast_time": temperature_now["valid_time"].values,
        "temperature_yesterday": temperature_yesterday.values,
        "temperature_now": temperature_now.values,
        "extreme_24h": extreme_future.values,
    })

    return df


def build_forecast_dataset(datasets,threshold,latitude_index,longitude_index,lag_hours=24,):
    all_rows = []

    for ds in datasets:
        # 1. extract the temperature point
        point_c = extract_temperature_point(ds, latitude_index, longitude_index)
        # 2. build this year's forecast rows
        df_of_the_year = build_forecast_rows(point_c, threshold, lag_hours)
        # 3. append those rows to all_rows
        
        df_of_the_year['year'] = df_of_the_year["forecast_time"].dt.year
        all_rows.append(df_of_the_year)
    # concatenate the yearly DataFrames
    df = pd.concat(all_rows, ignore_index=True)

    return df

def add_time_features(df):
    """
    Add cyclical hour-of-day features.
    """
    df = df.copy()
    df["hour"] = df["forecast_time"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    return df


def temporal_split(df,train_end_year=2018,validation_year=2019,test_year=2020,):
    """
    Split forecast samples chronologically.

    Training:   years <= train_end_year
    Validation: validation_year
    Test:       test_year
    """
    train_df = df[df["year"] <= train_end_year].copy()
    val_df = df[df["year"] == validation_year].copy()
    test_df = df[df["year"] == test_year].copy()

    return train_df, val_df, test_df

def standardize_features(train_df, val_df, test_df, features):
    """
    Standardize selected features using training statistics only.
    """

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    scaling_stats = {} #dictionary to save the scaling stats

    for feature in features:
        mean = train_df[feature].mean()
        std = train_df[feature].std()

        scaling_stats[feature] = {"mean": mean,"std": std,}

        train_df[f"{feature}_z"] = (train_df[feature] - mean) / std
        val_df[f"{feature}_z"] = (val_df[feature] - mean) / std
        test_df[f"{feature}_z"] = (test_df[feature] - mean) / std

    return train_df, val_df, test_df, scaling_stats

def make_model_arrays(
    train_df, val_df, test_df, feature_columns, target_column="extreme_24h"):
    """
    Convert DataFrames into NumPy feature matrices and target vectors.
    """

    X_train = train_df[feature_columns].to_numpy()
    y_train = train_df[target_column].to_numpy()

    X_val = val_df[feature_columns].to_numpy()
    y_val = val_df[target_column].to_numpy()

    X_test = test_df[feature_columns].to_numpy()
    y_test = test_df[target_column].to_numpy()

    return (X_train,y_train,X_val,y_val,X_test,y_test)