import pandas as pd
from session_4.metadata import DATASET_PATH


def load_data():
    df = pd.read_csv(DATASET_PATH)
    return df