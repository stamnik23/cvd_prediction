import streamlit as st
import os
import joblib
import json
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from utils.data_utils import load_dataset, preprocess_data
from utils.model_utils import train_model, evaluate_model
from utils.predict_utils import predict

st.set_page_config(page_title="Cardiovascular Disease - Predictor", layout="wide")
st.title("🫀 Cardiovascular Disease Predictor - Train & Predict")

tab_train, tab_predict = st.tabs(["Train Model", "Predict CVD"])

with tab_train:
    input_file = st.file_uploader("Upload CSV/XLS/XLSX dataset:", type=["csv","xls","xlsx"])
    
    if input_file:
        df = load_dataset(input_file)
        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        target_column = st.selectbox("Select target column:", options=df.columns)
        exclude_columns = st.multiselect(
            "Select columns to exclude from features:",
            options=[c for c in df.columns if c != target_column]
        )

        model_choice = st.selectbox("Select model:", ["Random Forest", "Logistic Regression", "Gradient Boosting", "XGBoost", "LightGBM", "TabNet"])
        model_name = st.text_input("Model name:", value="cvd_model")

        if st.button("Train Model") and model_name:
            features = [c for c in df.columns if c not in exclude_columns + [target_column]]
            X, y, final_features = preprocess_data(df, features, target_column)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model = train_model(X_train_scaled, y_train, model_choice)
            metrics = evaluate_model(model, X_test_scaled, y_test)

            st.subheader("Model Evaluation")
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Accuracy", f"{metrics['accuracy']:.2f}")
            col2.metric("Precision", f"{metrics['precision']:.2f}")
            col3.metric("Recall", f"{metrics['recall']:.2f}")
            col4.metric("F1 Score", f"{metrics['f1']:.2f}")
            col5.metric("ROC-AUC", f"{metrics['roc_auc']:.2f}")

            os.makedirs("models", exist_ok=True)
            model_file = f"models/{model_name}.joblib"
            scaler_file = f"models/{model_name}_scaler.joblib"
            features_file = f"models/{model_name}_metadata.json"

            joblib.dump(model, model_file)
            joblib.dump(scaler, scaler_file)
            with open(features_file, "w") as f:
                json.dump({"features": final_features, "target": target_column}, f)

            st.success(f"Model, scaler, features, and target saved in 'models/' folder")

with tab_predict:
    model_files = [f for f in os.listdir("models") if f.endswith(".joblib") and not f.endswith("_scaler.joblib")]
    if not model_files:
        st.warning("No models found. Upload a dataset to train a model first.")
    else:
        model_file = st.selectbox("Choose a model:", model_files)
        if model_file:
            scaler_file = model_file.replace(".joblib", "_scaler.joblib")
            metadata_file = model_file.replace(".joblib", "_metadata.json")

            model = joblib.load(f"models/{model_file}")
            scaler = joblib.load(f"models/{scaler_file}")
            with open(f"models/{metadata_file}", "r") as f:
                metadata = json.load(f)
            features = metadata["features"]
            target_column = metadata["target"]

            st.success(f"Loaded model '{model_file}' with target '{target_column}'")

            user_input = {}
            st.subheader("Enter Patient Values")
            for feature in features:
                if feature in ["sex", "menopause_im", "obese", "smoking_ever", "HTN", "HTN5", "DM", "DM5", "DM10"]:
                    user_input[feature] = st.selectbox(feature, [0,1], key=feature)
                else:
                    user_input[feature] = st.number_input(feature, value=0.0, key=feature)

            if st.button("Predict CVD"):
                pred, proba = predict(model, scaler, user_input)
                if pred == 0:
                    st.success(f"{target_column} = 0 → No event (Probability: {proba:.2f})")
                else:
                    st.error(f"{target_column} = 1 → Event likely (Probability: {proba:.2f})")
