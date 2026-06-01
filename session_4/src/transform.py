import pandas as pd
from session_4.metadata import (
    DROP_COLUMNS,
    BINARY_FEATURES,
    CATEGORICAL_COLUMNS,
    TARGET_COLUMN,
)


class Transformer:
    def __init__(self):
        self.drop_columns = DROP_COLUMNS
        self.binary_features = BINARY_FEATURES
        self.categorical_columns = CATEGORICAL_COLUMNS

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all transformations to the dataframe."""
        df = df.copy()
        df = self._drop_columns(df)
        df = self._cast_binary_to_int(df)
        df = self._one_hot_encoding(df)
        return df

    def _drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop identifier columns that are not useful for the model."""
        return df.drop(columns=self.drop_columns, errors="ignore")

    def _cast_binary_to_int(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cast binary columns to int, filling NaN with 0 first."""
        for col in self.binary_features:
            if col in df.columns:
                df[col] = df[col].fillna(0).astype(int)
        return df

    def _one_hot_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply one-hot encoding to categorical columns."""
        df = pd.get_dummies(df, columns=self.categorical_columns, drop_first=False)
        return df


def split_features_target(df: pd.DataFrame):
    """Split a dataframe into features (X) and target (y)."""
    y = df[TARGET_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN])
    return X, y
