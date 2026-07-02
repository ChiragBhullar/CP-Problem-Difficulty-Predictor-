"""
predict.py
----------
Loads the trained model and predicts the difficulty tier for a new,
hand-described problem. Demonstrates how the model would be used in practice
(e.g. behind a small API or CLI tool).
"""

import joblib
import pandas as pd

MODEL_PATH = "models/best_model.joblib"


def build_feature_row(feature_cols, points, num_tags, name_length,
                       is_versioned, has_special_tag, index_letter, tag_list):
    """Construct a single-row DataFrame matching the training feature schema."""
    row = {col: 0 for col in feature_cols}

    row["points"] = points
    row["num_tags"] = num_tags
    row["name_length"] = name_length
    row["is_versioned"] = int(is_versioned)
    row["has_special_tag"] = int(has_special_tag)

    idx_col = f"idx_{index_letter}"
    if idx_col in row:
        row[idx_col] = 1

    for tag in tag_list:
        tag_col = f"tag_{tag.replace(' ', '_')}"
        if tag_col in row:
            row[tag_col] = 1

    return pd.DataFrame([row])[feature_cols]


def predict_difficulty(**kwargs):
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    le = bundle["label_encoder"]
    scaler = bundle["scaler"]
    feature_cols = bundle["feature_cols"]

    X_new = build_feature_row(feature_cols, **kwargs)

    if scaler is not None:
        X_new_input = scaler.transform(X_new)
    else:
        X_new_input = X_new

    pred_idx = model.predict(X_new_input)[0]
    pred_label = le.inverse_transform([pred_idx])[0]

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_new_input)[0]
        proba_dict = dict(zip(le.classes_, proba.round(3)))
    else:
        proba_dict = None

    return pred_label, proba_dict


if __name__ == "__main__":
    # Example: a mid-contest problem, "C" slot, uses DP + greedy, mid-length name
    label, proba = predict_difficulty(
        points=1500,
        num_tags=2,
        name_length=20,
        is_versioned=False,
        has_special_tag=False,
        index_letter="C",
        tag_list=["dp", "greedy"],
    )
    print(f"Predicted difficulty: {label}")
    if proba:
        print("Class probabilities:")
        for cls, p in sorted(proba.items(), key=lambda x: -x[1]):
            print(f"  {cls:8s}: {p}")
