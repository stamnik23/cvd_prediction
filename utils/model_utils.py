from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd
import numpy as np

def train_model(X_train, y_train, model_choice):
    if isinstance(X_train, pd.DataFrame):
        X_train_array = X_train.values
    else:
        X_train_array = X_train
    if isinstance(y_train, pd.Series):
        y_train_array = y_train.values
    else:
        y_train_array = y_train

    if model_choice == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_array, y_train_array)
    elif model_choice == "Logistic Regression":
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train_array, y_train_array)
    elif model_choice == "Gradient Boosting":
        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_array, y_train_array)
    elif model_choice == "XGBoost":
        model = xgb.XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', random_state=42)
        model.fit(X_train_array, y_train_array)
    elif model_choice == "LightGBM":
        model = lgb.LGBMClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_array, y_train_array)
    elif model_choice == "TabNet":
        model = TabNetClassifier()
        model.fit(X_train_array, y_train_array, max_epochs=50, patience=10, batch_size=32, virtual_batch_size=16)
    return model

def evaluate_model(model, X_test, y_test):
    if isinstance(X_test, pd.DataFrame):
        X_test_array = X_test.values
    else:
        X_test_array = X_test
    if isinstance(y_test, pd.Series):
        y_test_array = y_test.values
    else:
        y_test_array = y_test

    if isinstance(model, TabNetClassifier):
        y_pred = model.predict(X_test_array).reshape(-1)
        y_proba = model.predict_proba(X_test_array)[:,1]
    else:
        y_pred = model.predict(X_test_array)
        y_proba = model.predict_proba(X_test_array)[:,1]

    metrics = {
        "accuracy": accuracy_score(y_test_array, y_pred),
        "precision": precision_score(y_test_array, y_pred),
        "recall": recall_score(y_test_array, y_pred),
        "f1": f1_score(y_test_array, y_pred),
        "roc_auc": roc_auc_score(y_test_array, y_proba)
    }
    return metrics

