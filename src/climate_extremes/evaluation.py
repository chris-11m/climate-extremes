from sklearn.metrics import (log_loss,roc_auc_score, brier_score_loss,)
import numpy as np
import pandas as pd

def evaluate_probabilities(y_true, probabilities):
    """
    Evaluate probabilistic binary predictions.
    """

    return {
        "log_loss": log_loss(y_true, probabilities),
        "brier_score": brier_score_loss(
            y_true,
            probabilities,
        ),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "mean_predicted_probability": probabilities.mean(),
        "observed_prevalence": y_true.mean(),
    }

def climatology_probabilities(y_train, n_samples):
    """
    Predict the training-set extreme prevalence for every sample.

    The climatology probability is learned from training data only.
    """
    train_prevalence = y_train.mean()

    probabilities = np.full(
        n_samples,
        train_prevalence,
        dtype=float,
    )

    return probabilities


def calibration_table(
    y_true,
    probabilities,
    n_bins=10,
    strategy="uniform",
):
    """
    Compare predicted probabilities with observed event frequency.

    strategy:
        "uniform"  -> equal-width probability bins
        "quantile" -> approximately equal number of samples per bin
    """

    df = pd.DataFrame({
        "y_true": y_true,
        "probability": probabilities,
    })

    if strategy == "uniform":
        df["bin"] = pd.cut(
            df["probability"],
            bins=np.linspace(0, 1, n_bins + 1),
            include_lowest=True,
        )

    elif strategy == "quantile":
        df["bin"] = pd.qcut(
            df["probability"],
            q=n_bins,
            duplicates="drop",
        )

    else:
        raise ValueError(
            "strategy must be 'uniform' or 'quantile'"
        )

    calibration = (
        df.groupby("bin", observed=False)
        .agg(
            count=("y_true", "size"),
            mean_predicted_probability=("probability", "mean"),
            observed_frequency=("y_true", "mean"),
        )
        .reset_index()
    )

    return calibration