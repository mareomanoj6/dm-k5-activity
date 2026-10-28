import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_kddcup99
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, ConfusionMatrixDisplay)

kdd = fetch_kddcup99(percent10=True, as_frame=True)
df = kdd.frame.copy()
print("Raw shape:", df.shape)

for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].apply(lambda x: x.decode() if isinstance(x, bytes) else x)

categorical_cols = ["protocol_type", "service", "flag"]
numeric_cols = [c for c in df.columns if c not in categorical_cols + ["labels"]]

df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

df["target"] = (df["labels"] != "normal.").astype(int)

print("Number of features:", len(categorical_cols) + len(numeric_cols))
print("Target distribution:\n", df["target"].value_counts())
print("Original attack types:", df["labels"].nunique())

before = len(df)
df = df.drop_duplicates()
print(f"Removed {before - len(df)} duplicate rows. New shape: {df.shape}")

print("Missing values:", df.isnull().sum().sum())
df = df.dropna()

X = df[categorical_cols + numeric_cols]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print("Train:", X_train.shape, " Test:", X_test.shape)

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
])

models = {
    "Logistic Regression": Pipeline([
        ("prep", preprocessor),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ]),
    "Random Forest": Pipeline([
        ("prep", preprocessor),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
    ]),
}

results = []
conf_matrices = {}

for name, model in models.items():
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    y_pred = model.predict(X_test)

    results.append({
        "Algorithm": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred),
        "Training time (s)": train_time,
    })
    conf_matrices[name] = confusion_matrix(y_test, y_pred)

results_df = pd.DataFrame(results).set_index("Algorithm").round(4)
print("\n=== Comparison of Evaluation Metrics ===")
print(results_df)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, (name, cm) in zip(axes, conf_matrices.items()):
    disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Attack"])
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title(name)
plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150)
plt.show()

print("\n=== Confusion Matrix Breakdown ===")
for name, cm in conf_matrices.items():
    tn, fp, fn, tp = cm.ravel()
    total = cm.sum()
    print(f"\n{name}")
    print(f"  True Negatives  (normal correctly identified): {tn}")
    print(f"  True Positives  (attacks correctly detected) : {tp}")
    print(f"  False Positives (normal flagged as attack)   : {fp}")
    print(f"  False Negatives (attacks MISSED)             : {fn}")
    print(f"  Correct: {tn + tp} | Incorrect: {fp + fn} | Total: {total}")

best_recall = results_df["Recall"].idxmax()
best_f1 = results_df["F1-score"].idxmax()
print(f"\nHighest recall (fewest missed attacks): {best_recall}")
print(f"Highest F1-score: {best_f1}")
print("Use these numbers, plus the FN/FP counts above, to write your conclusion.")
