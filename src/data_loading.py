"""Read the raw SMILES dataset and the descriptor datasets derived from it."""
import pandas as pd

from . import config


def load_raw_dataset(path=config.RAW_SMILES_FILE) -> pd.DataFrame:
    """result/Mixed_exclude.xlsx: SMILES, EO, PO, Cc for the 63 surfactants."""
    return pd.read_excel(path)


def load_descriptor_dataset(path=config.DESCRIPTORS_FILE) -> pd.DataFrame:
    """Case4_1.xlsx: raw columns + every RDKit descriptor."""
    return pd.read_excel(path)


def load_clean_dataset(path=config.CLEAN_FILE) -> pd.DataFrame:
    """Case4_1_clean.xlsx: Cc + the descriptors that survive filtering."""
    return pd.read_excel(path)


def split_features_target(df: pd.DataFrame, target: str = config.TARGET_COLUMN):
    X = df.drop(columns=[target])
    y = df[target]
    return X, y
