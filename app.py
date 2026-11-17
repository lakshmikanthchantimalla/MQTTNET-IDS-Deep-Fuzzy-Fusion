"""Minimal Flask upload interface based on the report's web-app section.

This is a demonstration endpoint. It expects an uploaded CSV whose final
column is the target label, matching preprocess_data() in the report.
For deployment, train and persist the preprocessing objects and models,
then load them here instead of retraining on every request.
"""
from flask import Flask, render_template, request
import pandas as pd

from preprocess import preprocess_data
from autoencoder import extract_deep_features
from feature_fusion import fuse_features
from main import run_training

app = Flask(__name__)

# Train models once when the server starts. Requires mqtt_dataset.csv.
xgb_model, lgb_model = run_training()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    uploaded_file = request.files.get("dataset")

    if uploaded_file is None or uploaded_file.filename == "":
        return "Please select a CSV dataset.", 400

    if not uploaded_file.filename.lower().endswith(".csv"):
        return "Only CSV files are supported.", 400

    try:
        data = pd.read_csv(uploaded_file)
        X, _ = preprocess_data(data)
        deep_features = extract_deep_features(X)
        fused = fuse_features(X, X, deep_features)

        predictions = xgb_model.predict(fused)
        return render_template(
            "result.html",
            result=str(predictions[:10])
        )
    except Exception as exc:
        return f"Could not process the uploaded dataset: {exc}", 400


if __name__ == "__main__":
    app.run(debug=True)
