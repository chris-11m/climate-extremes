from climate_extremes.data import load_era5_years
from climate_extremes.models import train_logistic_regression, train_xgboost
from climate_extremes.evaluation import evaluate_probabilities,climatology_probabilities,calibration_table
from climate_extremes.features import (
    compute_extreme_threshold,
    build_forecast_dataset,
    add_time_features,
    temporal_split,
    standardize_features,
    make_model_arrays,
)


datasets = load_era5_years(
    "data",
    range(2010, 2021),
)

threshold = compute_extreme_threshold(
    datasets,
    latitude_index=20,
    longitude_index=30,
    quantile=0.95,
    train_end_year=2018,
)

df = build_forecast_dataset(
    datasets,
    threshold,
    latitude_index=20,
    longitude_index=30,
    lag_hours=24,
)

df = add_time_features(df)

train_df, val_df, test_df = temporal_split(df)

train_df, val_df, test_df, scaling_stats = standardize_features(
    train_df,
    val_df,
    test_df,
    features=[
        "temperature_yesterday",
        "temperature_now",
    ],
)

feature_columns = [
    "temperature_yesterday_z",
    "temperature_now_z",
    "hour_sin",
    "hour_cos",
]

(X_train,y_train,X_val,y_val,X_test,y_test,) = make_model_arrays(
    train_df,
    val_df,
    test_df,
    feature_columns,
)


print("Threshold:", threshold)

print("\nDataset sizes")
print("Train:", X_train.shape, y_train.shape)
print("Validation:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test.shape)

print("\nTarget prevalence")
print("Train:", y_train.mean())
print("Validation:", y_val.mean())
print("Test:", y_test.mean())

print("\nScaling statistics")
print(scaling_stats)

model = train_logistic_regression(X_train, y_train)

p_train = model.predict_proba(X_train)[:, 1]
p_val = model.predict_proba(X_val)[:, 1]

print("\nLogistic regression")
print("Intercept:", model.intercept_[0])

for feature, coefficient in zip(
    feature_columns,
    model.coef_[0],
):
    print(f"{feature}: {coefficient:.4f}")


train_metrics = evaluate_probabilities(
    y_train,
    p_train,
)

val_metrics = evaluate_probabilities(
    y_val,
    p_val,
)

print("\nTrain metrics")
for name, value in train_metrics.items():
    print(f"{name}: {value:.4f}")

print("\nValidation metrics")
for name, value in val_metrics.items():
    print(f"{name}: {value:.4f}")


p_climatology_val = climatology_probabilities(
    y_train,
    len(y_val),
)

climatology_val_metrics = evaluate_probabilities(y_val,p_climatology_val,)

print("\nClimatology baseline - validation")
for name, value in climatology_val_metrics.items():
    print(f"{name}: {value:.4f}")


val_calibration = calibration_table(
    y_val,
    p_val,
    n_bins=10,
)

print("\nValidation calibration")
print(val_calibration)

val_calibration_quantile = calibration_table(
    y_val,
    p_val,
    n_bins=10,
    strategy="quantile",
)

print("\nValidation calibration - quantile bins")
print(val_calibration_quantile)


xgb_model = train_xgboost(
    X_train,
    y_train,
)

p_xgb_train = xgb_model.predict_proba(X_train)[:, 1]
p_xgb_val = xgb_model.predict_proba(X_val)[:, 1]

xgb_train_metrics = evaluate_probabilities(
    y_train,
    p_xgb_train,
)

xgb_val_metrics = evaluate_probabilities(
    y_val,
    p_xgb_val,
)

print("\nXGBoost - train")
for name, value in xgb_train_metrics.items():
    print(f"{name}: {value:.4f}")

print("\nXGBoost - validation")
for name, value in xgb_val_metrics.items():
    print(f"{name}: {value:.4f}")