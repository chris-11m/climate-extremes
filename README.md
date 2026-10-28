# Climate Extremes Forecasting

Machine-learning project for predicting extreme summer temperatures 24 hours ahead using ERA5 reanalysis data over the UK.

The project focuses not only on predictive performance, but also on building a leakage-safe forecasting pipeline with realistic temporal validation, probabilistic evaluation, and reproducible feature engineering.

## Project Goal

The current forecasting task is:

> Estimate the probability that temperature at a fixed UK grid point will exceed an extreme-temperature threshold 24 hours in the future, using information available at the forecast time.

Rather than defining an extreme using a universal temperature such as 30°C, the threshold is derived from the historical temperature distribution.

To avoid information leakage, the extreme threshold is estimated using the training period only and then kept fixed when evaluating later years.

## Data

The project uses hourly ERA5 single-level reanalysis data from the Copernicus Climate Data Store.

Current study period:

- June–August (JJA)
- 2010–2020
- UK regional domain: 48°N–60°N, 12°W–4°E
- 0.25° × 0.25° ERA5 grid

The initial model is developed at a single grid point at approximately:

- 55°N
- 4.5°W

The raw ERA5 NetCDF files are intentionally excluded from this repository because of their size. The download workflow is provided in `scripts/download_era5.py`.

## Forecast Design

For a forecast issued at time \(t\), the target is whether temperature at \(t+24\) hours is extreme.

The initial predictors are:

- temperature at time \(t\)
- temperature at \(t-24\) hours
- cyclic hour-of-day features

All predictors must be available at or before forecast time \(t\). Future information is used only to construct the target.

Lagged samples are constructed separately within each summer to prevent artificial transitions between August of one year and June of the next.

## Extreme Definition

The extreme-temperature threshold is defined as the 95th percentile of temperature during the training period.

Current temporal design:

| Partition | Years |
|---|---|
| Training | 2010–2018 |
| Validation | 2019 |
| Test | 2020 |

The threshold is calculated using **training data only** and then frozen for validation and testing.

For the current grid point, the training-derived 95th-percentile threshold is approximately **18.65°C**.

The 2020 test period is kept untouched while model and feature choices are developed using the training and validation periods.

## Models

The current pipeline compares:

- climatological probability baseline
- logistic regression
- XGBoost

Logistic regression provides an interpretable probabilistic baseline, while XGBoost tests whether nonlinear relationships and feature interactions improve forecasting performance.

## Evaluation

Because extreme temperatures are relatively rare, accuracy alone is not an appropriate evaluation metric.

Models are evaluated using:

- log loss
- Brier score
- ROC AUC
- calibration

A climatology baseline based only on the training-period extreme frequency is used as a reference.

### Current validation results

| Model | Log Loss | Brier Score | ROC AUC |
|---|---:|---:|---:|
| Climatology | 0.2693 | 0.0692 | 0.5000 |
| Logistic Regression | 0.1747 | 0.0522 | 0.9155 |
| XGBoost | 0.1728 | 0.0520 | 0.9137 |

Both machine-learning models substantially outperform the climatological baseline. XGBoost currently provides only a small improvement in probabilistic loss over logistic regression, suggesting that additional physically meaningful predictors may be more valuable than increasing model complexity alone.

## Atmospheric Feature Extension

The next stage extends the ERA5 predictors beyond temperature.

Additional variables being incorporated are:

- 2 m dew-point temperature
- mean sea-level pressure
- 10 m zonal wind component
- 10 m meridional wind component

These variables provide information about moisture, synoptic pressure conditions, and atmospheric flow.

Potential derived predictors include wind speed and temperature–dew-point spread.

The same leakage-safe forecasting and temporal validation framework will be retained when evaluating these features.

## Project Structure

```text
climate-extremes/
├── configs/
│   └── baseline.yaml
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── 01_era5_exploration.ipynb
├── scripts/
│   ├── build_dataset.py
│   └── download_era5.py
├── src/
│   └── climate_extremes/
│       ├── data.py
│       ├── evaluation.py
│       ├── features.py
│       └── models.py
├── tests/
│   └── test_features.py
├── .gitignore
├── pyproject.toml
└── README.md
```

## Installation

Create and activate a Python environment, then install the project in editable mode:

```bash
python -m pip install -e .
```

The project currently uses packages including:

- NumPy
- pandas
- xarray
- scikit-learn
- XGBoost
- cdsapi

## Running the Pipeline

Download the required ERA5 data:

```bash
python scripts/download_era5.py
```

Build and inspect the forecasting dataset:

```bash
python scripts/build_dataset.py
```

Run the test suite:

```bash
python -m pytest
```

ERA5 data are not committed to Git and must be downloaded separately.

## Planned Work

Current development priorities include:

1. integrate dew point, pressure, and wind predictors
2. compare atmospheric feature sets using the fixed 2019 validation period
3. examine feature importance and model calibration
4. evaluate the final selected model on the untouched 2020 test period
5. investigate sequence models such as LSTMs once strong tabular baselines are established
6. extend the analysis beyond a single grid point to investigate spatial generalization

## Motivation

Extreme-temperature forecasting is a useful test case for climate-risk modelling because it combines rare-event prediction, temporal dependence, non-stationarity, probabilistic forecasting, and careful treatment of data leakage.

The project is designed to explore these issues while building an end-to-end machine-learning workflow around real atmospheric reanalysis data.