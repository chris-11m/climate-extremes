from sklearn.linear_model import LogisticRegression


def train_logistic_regression(X_train, y_train):
    """
    Train a logistic regression baseline classifier.
    """

    model = LogisticRegression(
        max_iter=1000,
    )

    model.fit(X_train, y_train)

    return model

from xgboost import XGBClassifier


def train_xgboost(X_train, y_train):
    """
    Train an XGBoost classifier for extreme-temperature prediction.
    """

    model = XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
    )

    model.fit(X_train, y_train)

    return model