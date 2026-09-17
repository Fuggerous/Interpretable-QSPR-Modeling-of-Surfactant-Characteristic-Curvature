"""Constant/correlation filtering and the scaling + target transform used
to go from the full RDKit descriptor matrix to the modeling-ready inputs."""
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, RobustScaler

from . import config


def remove_constant_features(df: pd.DataFrame):
    """Drop columns with a single unique value (zero variance)."""
    constant_cols = [c for c in df.columns if df[c].nunique() <= 1]
    return df.drop(columns=constant_cols), constant_cols


def remove_correlated_features(df: pd.DataFrame, threshold: float = config.CORRELATION_THRESHOLD):
    """Drop one column from every pair with |correlation| above threshold."""
    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > threshold)]
    return df.drop(columns=to_drop), to_drop


def build_clean_dataset(descriptors_df: pd.DataFrame) -> pd.DataFrame:
    """Reproduce Case4_1_clean.xlsx: Cc + the descriptors that survive
    constant-feature removal and correlation filtering."""
    X = descriptors_df.drop(columns=config.NON_FEATURE_COLUMNS)
    y = descriptors_df[config.TARGET_COLUMN]

    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.dropna(axis=1, how="all")
    X, _ = remove_constant_features(X)
    X, _ = remove_correlated_features(X)

    combined = pd.concat([y, X], axis=1)
    return combined.loc[combined[config.TARGET_COLUMN].notna()].reset_index(drop=True)


def scale_features(X: pd.DataFrame) -> pd.DataFrame:
    """RobustScaler(10th-90th percentile) followed by MinMaxScaler."""
    scaler = Pipeline([
        ("robust", RobustScaler(quantile_range=(10, 90))),
        ("minmax", MinMaxScaler()),
    ])
    scaled = scaler.fit_transform(X)
    return pd.DataFrame(scaled, columns=X.columns, index=X.index)


def transform_target(y: pd.Series):
    """Yeo-Johnson transform of Cc. Returns the transformed array (1D) and
    the fitted PowerTransformer so predictions can be mapped back to Cc
    via `pt.inverse_transform`."""
    pt = PowerTransformer(method="yeo-johnson")
    y_transformed = pt.fit_transform(y.to_frame())
    return y_transformed.ravel(), pt
