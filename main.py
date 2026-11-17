"""Training and evaluation entry point for MQTTNET-IDS."""
from sklearn.model_selection import train_test_split

from preprocess import load_dataset, preprocess_data
from autoencoder import extract_deep_features
from feature_fusion import fuse_features
from models import train_xgboost, train_lightgbm, evaluate
from ensemble import ensemble_predict


def run_training(dataset_path="mqtt_dataset.csv"):
    data = load_dataset(dataset_path)
    X, y = preprocess_data(data)

    deep_features = extract_deep_features(X)

    # Matches the report: X is passed as both raw and statistical features.
    fused_features = fuse_features(X, X, deep_features)

    X_train, X_test, y_train, y_test = train_test_split(
        fused_features,
        y,
        test_size=0.2,
        random_state=42
    )

    xgb_model = train_xgboost(X_train, y_train)
    lgb_model = train_lightgbm(X_train, y_train)

    print("XGBoost Results")
    evaluate(xgb_model, X_test, y_test)

    print("LightGBM Results")
    evaluate(lgb_model, X_test, y_test)

    final_predictions = ensemble_predict(xgb_model, lgb_model, X_test)
    print("Ensemble predictions (first 10):", final_predictions[:10])

    return xgb_model, lgb_model


if __name__ == "__main__":
    run_training()
