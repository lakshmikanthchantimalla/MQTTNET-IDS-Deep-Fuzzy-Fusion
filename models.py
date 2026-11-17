"""XGBoost and LightGBM classification models."""
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report


def train_xgboost(X_train, y_train):
    model = XGBClassifier()
    model.fit(X_train, y_train)
    return model


def train_lightgbm(X_train, y_train):
    model = LGBMClassifier()
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print("Accuracy:", accuracy)
    print(classification_report(y_test, predictions))

    return accuracy
