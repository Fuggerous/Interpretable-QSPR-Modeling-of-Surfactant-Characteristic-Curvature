"""SHAP and partial-dependence interpretation of the final model, plus the
actual-vs-predicted parity plot."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from . import config


def compute_shap_values(model, X: pd.DataFrame):
    explainer = shap.TreeExplainer(model)
    return explainer, explainer.shap_values(X)


def plot_shap_summary(shap_values, X: pd.DataFrame, out_dir=config.OUTPUT_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)

    shap.summary_plot(shap_values, features=X, feature_names=X.columns, show=False)
    plt.title("SHAP Beeswarm - Final Model")
    plt.savefig(out_dir / "shap_summary_beeswarm.png", bbox_inches="tight")
    plt.close()

    shap.summary_plot(shap_values, features=X, feature_names=X.columns, plot_type="bar", show=False)
    plt.title("SHAP Bar Plot - Final Model")
    plt.savefig(out_dir / "shap_summary_bar.png", bbox_inches="tight")
    plt.close()


def plot_partial_dependence(model, X: pd.DataFrame, shap_values, top_k: int = 8,
                             out_dir=config.OUTPUT_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    rank = np.argsort(np.abs(shap_values).mean(axis=0))[::-1]
    top_features = list(X.columns[rank[:top_k]])

    fig, axes = plt.subplots(nrows=len(top_features), ncols=1, figsize=(6, 3.0 * len(top_features)))
    axes = np.atleast_1d(axes).ravel()
    for ax, feat in zip(axes, top_features):
        PartialDependenceDisplay.from_estimator(
            model, X, features=[feat], kind="average", grid_resolution=50, ax=ax
        )
        ax.set_title(f"PDP: {feat}")
        ax.grid(True)

    fig.tight_layout()
    fig.savefig(out_dir / "pdp_topk_by_shap.png", bbox_inches="tight")
    plt.close(fig)


def plot_parity(y_train, y_train_pred, y_holdout, y_holdout_pred, out_dir=config.OUTPUT_DIR):
    """Actual-vs-predicted scatter for train + hold-out, and the metrics
    used to annotate it (in whatever scale the y arrays are in)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    train_r2 = r2_score(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    holdout_r2 = r2_score(y_holdout, y_holdout_pred)
    holdout_rmse = np.sqrt(mean_squared_error(y_holdout, y_holdout_pred))
    holdout_mae = mean_absolute_error(y_holdout, y_holdout_pred)

    plt.figure(figsize=(8, 6))
    plt.scatter(y_train, y_train_pred, alpha=0.7, color="blue",
                label=f"Train\nRMSE={train_rmse:.2f}, R2={train_r2:.2f}, MAE={train_mae:.2f}")
    plt.scatter(y_holdout, y_holdout_pred, alpha=0.7, color="green",
                label=f"Hold-out\nRMSE={holdout_rmse:.2f}, R2={holdout_r2:.2f}, MAE={holdout_mae:.2f}")

    all_vals = np.concatenate([
        np.ravel(y_train), np.ravel(y_holdout), np.ravel(y_train_pred), np.ravel(y_holdout_pred)
    ])
    lims = [all_vals.min(), all_vals.max()]
    plt.plot(lims, lims, "r--", label="Perfect prediction")
    plt.xlim(lims)
    plt.ylim(lims)
    plt.xlabel("Actual Cc")
    plt.ylabel("Predicted Cc")
    plt.title("Actual vs Predicted (Final Model)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "parity_plot.png")
    plt.close()

    return {
        "train_r2": train_r2, "train_rmse": train_rmse, "train_mae": train_mae,
        "holdout_r2": holdout_r2, "holdout_rmse": holdout_rmse, "holdout_mae": holdout_mae,
    }
