"""Ensemble prediction module."""
import numpy as np


def ensemble_predict(xgb_model, lgb_model, X_test):
    """
    Combine XGBoost and LightGBM predictions.

    The report keeps the XGBoost prediction when the two models disagree.
    This is an XGBoost tie-break rule, not a true majority vote.
    """
    pred_xgb = xgb_model.predict(X_test)
    pred_lgb = lgb_model.predict(X_test)

    final_pred = []
    for i in range(len(pred_xgb)):
        if pred_xgb[i] == pred_lgb[i]:
            final_pred.append(pred_xgb[i])
        else:
            final_pred.append(pred_xgb[i])

    return np.asarray(final_pred)
