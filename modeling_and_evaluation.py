"""SWYNEX Task 3: baseline model for the UCI Adult Income dataset.

Run:
    python modeling_and_evaluation.py

The script downloads the public UCI dataset, trains a logistic-regression pipeline,
prints evaluation metrics, and writes outputs to reports/ and models/.
"""
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
           "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
           "hours_per_week", "native_country", "income"]
BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/"

def main():
    train = pd.read_csv(BASE_URL + "adult.data", header=None, names=COLUMNS, na_values="?",
                        skipinitialspace=True, comment="|")
    test = pd.read_csv(BASE_URL + "adult.test", header=None, names=COLUMNS, na_values="?",
                       skipinitialspace=True, comment="|", skiprows=1)
    df = pd.concat([train, test], ignore_index=True)
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()
    df["income"] = df["income"].str.replace(".", "", regex=False)
    df = df.dropna(subset=["income"]).drop_duplicates().reset_index(drop=True)

    X = df.drop(columns="income")
    y = (df["income"] == ">50K").astype(int)
    numeric = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical = X.select_dtypes(exclude=["int64", "float64"]).columns.tolist()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    prep = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")),
                              ("scaler", StandardScaler())]), numeric),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                  ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical)
    ])
    model = Pipeline([("preprocessor", prep),
                      ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced",
                                                        random_state=RANDOM_STATE))])
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    model.fit(X_train, y_train)

    rows = []
    for name, fitted in [("Majority-class dummy", dummy), ("Logistic regression", model)]:
        pred = fitted.predict(X_test)
        rows.append({
            "model": name,
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, fitted.predict_proba(X_test)[:, 1])
        })
    results = pd.DataFrame(rows).set_index("model")
    print("Dataset shape:", df.shape)
    print("\\nEvaluation metrics:")
    print(results.round(4))
    print("\\nClassification report:")
    print(classification_report(y_test, model.predict(X_test),
                                target_names=["<=50K", ">50K"], zero_division=0))

    Path("reports").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    results.to_csv("reports/model_evaluation_metrics.csv")
    joblib.dump(model, "models/adult_income_logistic_regression.joblib")
    print("\\nSaved reports/model_evaluation_metrics.csv")
    print("Saved models/adult_income_logistic_regression.joblib")

if __name__ == "__main__":
    main()
