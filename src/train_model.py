import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


INPUT = "data/processed/cleaned_jobs.csv"

MODEL_DIR = "models"


df = pd.read_csv(INPUT)


# --------------------------------------------------
# Clean salary
# --------------------------------------------------

df["salary_numeric"] = pd.to_numeric(
    df["salary_numeric"],
    errors="coerce"
)


# Keep valid salaries

df = df[
    df["salary_numeric"].notna()
]


# Remove unrealistic values

df = df[
    (df["salary_numeric"] > 10000)
    &
    (df["salary_numeric"] < 100000000)
]


# Need enough data

if len(df) < 20:

    raise ValueError(
        "Not enough salary records for ML training."
    )


# --------------------------------------------------
# Prepare columns
# --------------------------------------------------

for column in [
    "job_title",
    "location",
    "experience",
    "skills_raw",
    "description"
]:

    df[column] = (
        df[column]
        .fillna("")
        .astype(str)
    )


# --------------------------------------------------
# Combine text
# --------------------------------------------------

df["combined_text"] = (
    df["job_title"]
    + " "
    + df["skills_raw"]
    + " "
    + df["description"]
)


X = df[
    [
        "job_title",
        "location",
        "experience",
        "combined_text"
    ]
]


y = df["salary_numeric"]


# --------------------------------------------------
# Preprocessing
# --------------------------------------------------

preprocessor = ColumnTransformer(

    transformers=[

        (
            "job_title",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            ["job_title"]
        ),

        (
            "location",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            ["location"]
        ),

        (
            "experience",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            ["experience"]
        ),

        (
            "text",
            TfidfVectorizer(
                max_features=1000,
                stop_words="english"
            ),
            "combined_text"
        )

    ],

    remainder="drop"
)


# --------------------------------------------------
# Model
# --------------------------------------------------

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)


pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# --------------------------------------------------
# Train/Test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )
)


# --------------------------------------------------
# Train
# --------------------------------------------------

pipeline.fit(
    X_train,
    y_train
)


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

predictions = pipeline.predict(
    X_test
)


mae = mean_absolute_error(
    y_test,
    predictions
)


r2 = r2_score(
    y_test,
    predictions
)


print("\n" + "=" * 60)

print("MODEL PERFORMANCE")

print("=" * 60)

print(
    f"MAE: ₹{mae:,.0f}"
)

print(
    f"R² Score: {r2:.3f}"
)


# --------------------------------------------------
# Save
# --------------------------------------------------

import os

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


joblib.dump(
    pipeline,
    f"{MODEL_DIR}/salary_model.pkl"
)


print(
    "\n✅ Salary model saved"
)