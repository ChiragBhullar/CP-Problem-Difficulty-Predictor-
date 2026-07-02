"""
train_models.py
----------------
Trains and compares multiple classifiers to predict competitive programming
problem difficulty tier (Easy / Medium / Hard / Expert) from metadata features
engineered in prepare_data.py.

Models compared:
    - Logistic Regression   (simple linear baseline)
    - Decision Tree         (interpretable, non-linear baseline)
    - Random Forest         (ensemble, usually strong on tabular data)
    - Gradient Boosting     (sequential ensemble, often best on tabular data)

Outputs:
    - Console: accuracy / F1 / confusion matrix per model
    - models/feature_importance.png : bar chart from the best model
    - models/model_comparison.png   : accuracy comparison chart
    - models/best_model.joblib      : the best-performing trained model
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

DATA_PATH = "data/problems_features.csv"
MODELS_DIR = "models"

import os
os.makedirs(MODELS_DIR, exist_ok=True)


def load_dataset():
    df = pd.read_csv(DATA_PATH)

    # Columns that are identifiers / leak the target / raw text -> excluded from X
    drop_cols = [
        "contestId", "index", "name", "type", "tags",
        "rating", "difficulty_bucket", "index_letter",
    ]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].copy()

    # One-hot encode the index_letter (categorical, low cardinality)
    index_dummies = pd.get_dummies(df["index_letter"], prefix="idx")
    X = pd.concat([X, index_dummies], axis=1)

    y = df["difficulty_bucket"]

    return X, y, feature_cols


def main():
    X, y, feature_cols = load_dataset()
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target classes: {sorted(y.unique())}\n")

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    # Scale features for the linear model; tree-based models don't need it
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=42),
    }

    results = {}
    trained = {}

    for name, model in models.items():
        if name == "Logistic Regression":
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
            cv_scores = cross_val_score(model, scaler.fit_transform(X), y_enc, cv=5)
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            cv_scores = cross_val_score(model, X, y_enc, cv=5)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")

        results[name] = {
            "accuracy": acc,
            "f1_weighted": f1,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
        }
        trained[name] = model

        print(f"=== {name} ===")
        print(f"Test Accuracy       : {acc:.3f}")
        print(f"Test F1 (weighted)  : {f1:.3f}")
        print(f"5-fold CV Accuracy  : {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
        print(classification_report(y_test, preds, target_names=le.classes_, zero_division=0))
        print()

    # --- Pick the best model by weighted F1 on the held-out test set ---
    best_name = max(results, key=lambda n: results[n]["f1_weighted"])
    best_model = trained[best_name]
    print(f"Best model: {best_name} (F1 weighted = {results[best_name]['f1_weighted']:.3f})")

    # Confusion matrix for the best model
    if best_name == "Logistic Regression":
        best_preds = best_model.predict(X_test_scaled)
    else:
        best_preds = best_model.predict(X_test)

    cm = confusion_matrix(y_test, best_preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(le.classes_)))
    ax.set_yticks(range(len(le.classes_)))
    ax.set_xticklabels(le.classes_)
    ax.set_yticklabels(le.classes_)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {best_name}")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(f"{MODELS_DIR}/confusion_matrix.png", dpi=150)
    plt.close(fig)

    # Model comparison chart
    fig, ax = plt.subplots(figsize=(7, 5))
    names = list(results.keys())
    accs = [results[n]["accuracy"] for n in names]
    f1s = [results[n]["f1_weighted"] for n in names]
    x = np.arange(len(names))
    width = 0.35
    ax.bar(x - width/2, accs, width, label="Accuracy")
    ax.bar(x + width/2, f1s, width, label="Weighted F1")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title("Model Comparison — CP Difficulty Predictor")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{MODELS_DIR}/model_comparison.png", dpi=150)
    plt.close(fig)

    # Feature importance (only meaningful for tree-based models)
    if hasattr(best_model, "feature_importances_"):
        importances = pd.Series(best_model.feature_importances_, index=X.columns)
        top15 = importances.sort_values(ascending=False).head(15)

        fig, ax = plt.subplots(figsize=(8, 6))
        top15.sort_values().plot(kind="barh", ax=ax, color="#4C72B0")
        ax.set_title(f"Top 15 Feature Importances — {best_name}")
        ax.set_xlabel("Importance")
        fig.tight_layout()
        fig.savefig(f"{MODELS_DIR}/feature_importance.png", dpi=150)
        plt.close(fig)

        print("\nTop 10 most important features:")
        print(top15.sort_values(ascending=False).head(10))

    # Save the best model + supporting objects for inference
    joblib.dump(
        {
            "model": best_model,
            "label_encoder": le,
            "scaler": scaler if best_name == "Logistic Regression" else None,
            "feature_cols": list(X.columns),
            "model_name": best_name,
        },
        f"{MODELS_DIR}/best_model.joblib",
    )
    print(f"\nSaved best model to {MODELS_DIR}/best_model.joblib")

    # Save a small results summary
    pd.DataFrame(results).T.to_csv(f"{MODELS_DIR}/results_summary.csv")
    print(f"Saved results summary to {MODELS_DIR}/results_summary.csv")


if __name__ == "__main__":
    main()
