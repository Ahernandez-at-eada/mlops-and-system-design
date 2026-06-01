# Paths
DATASET_PATH = "session_4/datasets/Churn_Modelling_train_test.csv"
MODELS_PATH = "session_4/models/"


AUTHOR_NAME = "ahernandez"

TARGET_COLUMN = "Exited"


DROP_COLUMNS = ["CustomerId", "Surname", "RowNumber"]

CATEGORICAL_COLUMNS = ["Geography", "Gender"]

BINARY_FEATURES = ["HasCrCard", "IsActiveMember"]

TEST_SIZE = 0.2
RANDOM_STATE = 42

MODEL_PARAMS = {
    "max_depth": 5,
    "min_samples_split": 10,
    "random_state": 42,
}