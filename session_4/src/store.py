import os
import joblib
from datetime import datetime
from sklearn.base import BaseEstimator
from session_4.metadata import MODELS_PATH, AUTHOR_NAME


def save_model(model: BaseEstimator) -> str:
    """
    Save a trained model to the models folder with a timestamped filename.

    Args:
        model: A trained scikit-learn model.

    Returns:
        The path where the model was saved.
    """
    # Create the models folder if it does not exist
    os.makedirs(MODELS_PATH, exist_ok=True)

    # Build the filename: class_model-{name}-{timestamp}.joblib
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    filename = f"class_model-{AUTHOR_NAME}-{timestamp}.joblib"
    full_path = os.path.join(MODELS_PATH, filename)

    # Save the model
    joblib.dump(model, full_path)
    print(f"Model saved to: {full_path}")

    return full_path