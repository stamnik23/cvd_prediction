import pandas as pd
import numpy as np

def predict(model, scaler, user_input, feature_order=None):
    if feature_order:
        new_data = pd.DataFrame([user_input], columns=feature_order)
    else:
        new_data = pd.DataFrame([user_input])
    
    if hasattr(model, "predict_proba") and not hasattr(model, "fit"):
        # scikit-learn models
        new_data_scaled = scaler.transform(new_data)
        pred = model.predict(new_data_scaled)[0]
        proba = model.predict_proba(new_data_scaled)[0][1]
    else:
        # TabNet or models expecting raw arrays
        X_array = new_data.values if isinstance(new_data, pd.DataFrame) else new_data
        pred = model.predict(X_array)
        if isinstance(pred, np.ndarray) and pred.ndim > 1:
            pred = pred[0][0]
        proba = model.predict_proba(X_array)
        if isinstance(proba, np.ndarray) and proba.ndim > 1:
            proba = proba[:, 1][0]
    return pred, proba
