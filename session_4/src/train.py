import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from session_4.metadata import MODEL_PARAMS, TEST_SIZE, RANDOM_STATE


def train_model(X: pd.DataFrame, y: pd.Series) -> DecisionTreeClassifier:
    """
    Train a Decision Tree model on the given features and target.

    Args:
        X: Features dataframe.
        y: Target series.

    Returns:
        A trained DecisionTreeClassifier model.
    """
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Fill any remaining NaN values with the median (calculated only on train)
    medians = X_train.median()
    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)

    # Create and train the model
    model = DecisionTreeClassifier(**MODEL_PARAMS)
    model.fit(X_train, y_train)

    # Print accuracy on test set (just for feedback)
    accuracy = model.score(X_test, y_test)
    print(f"Model accuracy on test set: {accuracy:.4f}")

    return model
