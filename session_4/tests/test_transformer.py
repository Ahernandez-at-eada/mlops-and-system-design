import pandas as pd
import pytest
from session_4.src.transform import Transformer, split_features_target


@pytest.fixture
def sample_df():
    """Provide a small sample dataframe similar to the real dataset."""
    return pd.DataFrame(
        {
            "RowNumber": [1, 2, 3],
            "CustomerId": [100, 200, 300],
            "Surname": ["A", "B", "C"],
            "CreditScore": [650, 700, 600],
            "Geography": ["France", "Spain", "Germany"],
            "Gender": ["Male", "Female", "Male"],
            "Age": [40, 35, 50],
            "Tenure": [5, 3, 8],
            "Balance": [50000.0, 0.0, 120000.0],
            "NumOfProducts": [1, 2, 1],
            "HasCrCard": [1.0, 0.0, 1.0],
            "IsActiveMember": [1.0, 1.0, 0.0],
            "EstimatedSalary": [80000.0, 60000.0, 90000.0],
            "Exited": [0, 1, 0],
        }
    )


def test_transform_drops_identifier_columns(sample_df):
    """The transformer should drop RowNumber, CustomerId and Surname."""
    transformer = Transformer()
    result = transformer.transform(sample_df)

    assert "RowNumber" not in result.columns
    assert "CustomerId" not in result.columns
    assert "Surname" not in result.columns


def test_transform_creates_one_hot_columns(sample_df):
    """The transformer should create one-hot columns for Geography and Gender."""
    transformer = Transformer()
    result = transformer.transform(sample_df)

    # Original categorical columns should not be present
    assert "Geography" not in result.columns
    assert "Gender" not in result.columns

    # New one-hot columns should be present
    assert "Geography_France" in result.columns
    assert "Gender_Male" in result.columns


def test_transform_binary_columns_are_int(sample_df):
    """HasCrCard and IsActiveMember should be converted to int."""
    transformer = Transformer()
    result = transformer.transform(sample_df)

    assert result["HasCrCard"].dtype == int
    assert result["IsActiveMember"].dtype == int


def test_transform_handles_nan_in_binary_columns():
    """Binary columns with NaN values should be filled with 0 and cast to int."""
    df_with_nan = pd.DataFrame(
        {
            "RowNumber": [1, 2],
            "CustomerId": [100, 200],
            "Surname": ["A", "B"],
            "CreditScore": [650, 700],
            "Geography": ["France", "Spain"],
            "Gender": ["Male", "Female"],
            "Age": [40, 35],
            "Tenure": [5, 3],
            "Balance": [50000.0, 0.0],
            "NumOfProducts": [1, 2],
            "HasCrCard": [1.0, None],  # NaN here
            "IsActiveMember": [None, 1.0],  # NaN here
            "EstimatedSalary": [80000.0, 60000.0],
            "Exited": [0, 1],
        }
    )

    transformer = Transformer()
    result = transformer.transform(df_with_nan)

    # Should not raise errors and NaN should become 0
    assert result["HasCrCard"].tolist() == [1, 0]
    assert result["IsActiveMember"].tolist() == [0, 1]


def test_split_features_target_separates_correctly(sample_df):
    """split_features_target should separate Exited from the rest."""
    transformer = Transformer()
    df_transformed = transformer.transform(sample_df)
    X, y = split_features_target(df_transformed)

    assert "Exited" not in X.columns
    assert y.name == "Exited"
    assert len(X) == len(y)
