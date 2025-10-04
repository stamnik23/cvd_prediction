import streamlit as st
import os
import joblib
import json
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from utils.data_utils import load_dataset, preprocess_data
from utils.model_utils import train_model, evaluate_model
from utils.predict_utils import predict

st.set_page_config(page_title="CVD Risk Predictor", layout="wide", page_icon="💉")

st.markdown(
    """
    <style>
    .main {background-color: #f8fafc;}
    h1 {color: #0f172a !important; text-align: center;}
    h2, h3, h4 {color: #1e293b !important;}
    .stButton>button {
        background-color: #2563eb;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        border: none;
        font-weight: 600;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #1e40af;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Cardiovascular Disease Risk Predictor")

tab_train, tab_predict = st.tabs(["Train Model", "Predict Risk"])

with tab_train:
    dataset_source = st.radio("Select dataset source", ["Upload dataset", "Use sample dataset"])
    df = None
    if dataset_source == "Upload dataset":
        input_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xls", "xlsx"])
        if input_file:
            df = load_dataset(input_file)
    else:
        sample_path = "sampledata.csv"
        if os.path.exists(sample_path):
            df = pd.read_csv(sample_path)
            st.success("Sample dataset loaded.")
        else:
            st.error("Sample dataset not found. Please upload a file instead.")

    if df is not None:
        st.dataframe(df.head())
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            target_column = st.selectbox("Target column", options=df.columns)
        with col2:
            exclude_columns = st.multiselect(
                "Exclude columns",
                options=[c for c in df.columns if c != target_column]
            )

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            model_choice = st.selectbox(
                "Model",
                ["Random Forest", "Logistic Regression", "Gradient Boosting", "XGBoost", "LightGBM", "TabNet"]
            )
        with col2:
            model_name = st.text_input("Model name", value="cvd_model")

        if st.button("Train Model"):
            features = [c for c in df.columns if c not in exclude_columns + [target_column]]
            X, y, final_features = preprocess_data(df, features, target_column)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model = train_model(X_train_scaled, y_train, model_choice)
            metrics = evaluate_model(model, X_test_scaled, y_test)

            st.subheader("Model Evaluation")
            cols = st.columns(5)
            cols[0].metric("Accuracy", f"{metrics['accuracy']:.2f}")
            cols[1].metric("Precision", f"{metrics['precision']:.2f}")
            cols[2].metric("Recall", f"{metrics['recall']:.2f}")
            cols[3].metric("F1", f"{metrics['f1']:.2f}")
            cols[4].metric("ROC-AUC", f"{metrics['roc_auc']:.2f}")

            os.makedirs("models", exist_ok=True)
            joblib.dump(model, f"models/{model_name}.joblib")
            joblib.dump(scaler, f"models/{model_name}_scaler.joblib")
            with open(f"models/{model_name}_metadata.json", "w") as f:
                json.dump({"features": final_features, "target": target_column}, f)

            st.success("Model, scaler, and metadata saved.")

with tab_predict:
    os.makedirs("models", exist_ok=True)
    model_files = [f for f in os.listdir("models") if f.endswith(".joblib") and not f.endswith("_scaler.joblib")]

    if not model_files:
        st.warning("No trained models found.")
    else:
        model_file = st.selectbox("Select model", model_files)
        if model_file:
            scaler_file = model_file.replace(".joblib", "_scaler.joblib")
            metadata_file = model_file.replace(".joblib", "_metadata.json")

            model = joblib.load(f"models/{model_file}")
            scaler = joblib.load(f"models/{scaler_file}")
            with open(f"models/{metadata_file}", "r") as f:
                metadata = json.load(f)
            features = metadata["features"]
            target_column = metadata["target"]

            st.info(f"Model loaded: {model_file}")

            user_input = {}
            cols = st.columns(2)
            for i, feature in enumerate(features):
                with cols[i % 2]:
                    if feature in ["sex", "menopause_im", "obese", "smoking_ever", "HTN", "HTN5", "DM", "DM5", "DM10"]:
                        user_input[feature] = st.selectbox(feature, [0, 1], key=feature)
                    else:
                        user_input[feature] = st.number_input(feature, value=0.0, key=feature)

            if st.button("Predict"):
                pred, proba = predict(model, scaler, user_input)
                if pred == 0:
                    st.success(f"{target_column} = 0 → No event (Probability: {proba:.2f})")
                else:
                    st.error(f"{target_column} = 1 → Event likely (Probability: {proba:.2f})")
