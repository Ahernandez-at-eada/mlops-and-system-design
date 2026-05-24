"""
Flask inference server for the churn model.
Endpoint: POST /invocations
Input JSON format:
    {"dataframe_split": {"columns": [...], "data": [[...]]}}
Output JSON format:
    {"predictions": [...]}

Run from the 'Class Exercise' folder:
    python server.py
"""
import json
import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load model once at startup so every request reuses the same object
model = joblib.load("model_utils/lr_model.pkl")
print("Model loaded from model_utils/lr_model.pkl")


@app.route("/invocations", methods=["POST"])
def invocations():
    """Receive a dataframe_split payload and return class predictions."""
    payload = request.get_json(force=True)
    df_split = payload["dataframe_split"]
    df = pd.DataFrame(data=df_split["data"], columns=df_split["columns"])
    predictions = model.predict(df).tolist()
    return jsonify({"predictions": predictions})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
