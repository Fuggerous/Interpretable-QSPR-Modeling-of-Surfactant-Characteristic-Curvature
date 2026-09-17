"""Model comparison and the nested cross-validated XGBoost pipeline used
for the final Cc model."""
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from . import config

CANDIDATE_MODELS = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "Elastic Net Regression": ElasticNet(),
    "Decision Tree Regression": DecisionTreeRegressor(),
    "Random Forest Regression": RandomForestRegressor(),
    "Gradient Boosting Regression": GradientBoostingRegressor(),
    "XGBoost Regression": XGBRegressor(),
}

XGB_PARAM_GRID = {
    "n_estimators": [900, 1000, 1100, 1200],
    "max_depth": [3],
    "learning_rate": [0.01, 0.02],
    "subsample": [1.0, 0.8, 0.6],
    "reg_lambda": [1, 5, 10],
    "reg_alpha": [0, 0.1, 1],
    "gamma": [0, 0.1, 1],
    "min_child_weight": [5, 10],
}


def shap_rank_features(model, X: pd.DataFrame) -> np.ndarray:
    """Column indices of X, ordered by descending mean(|SHAP value|)."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    return np.argsort(mean_abs_shap)[::-1]


def nested_cv_xgboost(X: pd.DataFrame, y: np.ndarray,
                       n_splits: int = 5,
                       random_state: int = config.RANDOM_STATE) -> pd.DataFrame:
    """Outer K-fold estimate of generalization performance.

    Each outer fold re-ranks features by SHAP on a quick XGBoost fit, then
    an inner grid search picks the hyperparameters and feature-set size
    (by number of top-ranked SHAP features) that minimize inner CV error.
    The winning configuration for that fold is then scored on the outer
    test split. Returns one row per outer fold.
    """
    outer_kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    max_features = X.shape[1]
    outer_results = []

    for fold, (train_idx, test_idx) in enumerate(outer_kf.split(X, y), start=1):
        X_train_outer, y_train_outer = X.iloc[train_idx], y[train_idx]
        X_test_outer, y_test_outer = X.iloc[test_idx], y[test_idx]

        quick_model = XGBRegressor(random_state=random_state)
        quick_model.fit(X_train_outer, y_train_outer)
        ranked_idx = shap_rank_features(quick_model, X_train_outer)

        inner_kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        inner_results = []
        for n_features in range(1, max_features + 1):
            top_idx = ranked_idx[:n_features]
            X_train_top = X_train_outer.iloc[:, top_idx]

            grid_search = GridSearchCV(
                estimator=XGBRegressor(random_state=random_state),
                param_grid=XGB_PARAM_GRID,
                cv=inner_kf,
                scoring="neg_mean_squared_error",
                n_jobs=-1,
            )
            try:
                grid_search.fit(X_train_top, y_train_outer)
            except ValueError:
                continue

            best_model = grid_search.best_estimator_
            inner_results.append({
                "n_features": n_features,
                "params": grid_search.best_params_,
                "rmse": np.sqrt(-grid_search.best_score_),
                "r2": r2_score(y_train_outer, best_model.predict(X_train_top)),
                "top_idx": top_idx,
            })

        if not inner_results:
            continue

        best_r2 = max(r["r2"] for r in inner_results)
        candidates = [r for r in inner_results if abs(r["r2"] - best_r2) < 1e-6]
        best = min(candidates, key=lambda r: r["rmse"])

        X_train_best = X_train_outer.iloc[:, best["top_idx"]]
        X_test_best = X_test_outer.iloc[:, best["top_idx"]]
        final_model = XGBRegressor(**best["params"])
        final_model.fit(X_train_best, y_train_outer)
        y_pred = final_model.predict(X_test_best)

        outer_results.append({
            "fold": fold,
            "n_features": best["n_features"],
            "params": best["params"],
            "top_idx": best["top_idx"],
            "r2": r2_score(y_test_outer, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test_outer, y_pred)),
            "mae": mean_absolute_error(y_test_outer, y_pred),
        })

    return pd.DataFrame(outer_results)


def train_final_model(X: pd.DataFrame, y: np.ndarray, outer_results: pd.DataFrame,
                       test_size: float = config.TEST_SIZE,
                       random_state: int = config.RANDOM_STATE) -> dict:
    """Refit XGBoost on a fresh train/hold-out split, using the feature
    count and hyperparameters from the best-scoring outer fold."""
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    best_fold = outer_results.loc[outer_results["r2"].idxmax()]
    feature_cols = X.columns[best_fold["top_idx"]]

    model = XGBRegressor(**best_fold["params"])
    model.fit(X_train[feature_cols], y_train)

    return {
        "model": model,
        "feature_cols": feature_cols,
        "X_train": X_train[feature_cols],
        "X_holdout": X_holdout[feature_cols],
        "y_train": y_train,
        "y_holdout": y_holdout,
    }
