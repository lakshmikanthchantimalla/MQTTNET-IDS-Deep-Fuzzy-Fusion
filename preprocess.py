"""Dataset loading and preprocessing module."""
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_dataset(path):
    """Load an MQTT traffic dataset from a CSV file."""
    return pd.read_csv(path)


def preprocess_data(data):
    """
    Drop rows with missing values, encode categorical columns,
    and standardize input features.

    Assumption from the project report: the final CSV column is the target label.
    Returns X_scaled, y.
    """
    data = data.dropna().copy()

    if data.shape[1] < 2:
        raise ValueError("Dataset must contain at least one feature column and one target column.")

    for col in data.select_dtypes(include=["object", "category"]).columns:
        encoder = LabelEncoder()
        data[col] = encoder.fit_transform(data[col].astype(str))

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y
