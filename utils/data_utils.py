import pandas as pd

def load_dataset(file):
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    return df

def preprocess_data(df, features, target, exclude_cols=None):
    if exclude_cols is None:
        exclude_cols = []
    df = df.dropna(subset=[target])
    available_features = [f for f in features if f in df.columns and f not in exclude_cols]
    for col in available_features:
        if df[col].dtype in ["float64", "int64"]:
            df[col] = df[col].fillna(df[col].mean())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])
    X = df[available_features]
    y = df[target]
    return X, y, available_features

