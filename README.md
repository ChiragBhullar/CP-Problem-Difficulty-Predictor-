# Competitive Programming Difficulty Predictor

A machine learning system that predicts the difficulty tier of a competitive
programming problem (Codeforces) using problem metadata — tags, contest
points, problem index, and name characteristics — instead of the problem
statement text itself.

Real problem data is pulled live from the free, public **Codeforces API**
(`problemset.problems` endpoint) — no API key required.

## Problem Framing

Codeforces assigns each rated problem a numeric difficulty rating
(e.g. 800, 1400, 2100...). This project buckets that rating into four
human-readable tiers and trains classifiers to predict the tier from
metadata alone:

| Rating range | Tier   |
|---------------|--------|
| < 1200        | Easy   |
| 1200 – 1899   | Medium |
| 1900 – 2399   | Hard   |
| ≥ 2400        | Expert |

## Pipeline

```
data/raw_codeforces_response.json   -> raw API pull
        |
src/prepare_data.py                 -> feature engineering
        |
data/problems_features.csv          -> clean, model-ready dataset
        |
src/train_models.py                 -> train + compare 4 classifiers
        |
models/best_model.joblib            -> saved best model
        |
src/predict.py                      -> inference on a new problem
```

## Features Engineered

- **Tags** — multi-hot encoded (each CF tag like `dp`, `graphs`, `greedy` becomes a binary column)
- **num_tags** — how many tags a problem has (harder problems tend to combine more techniques)
- **points** — contest scoring weight (proxy for perceived difficulty at contest time)
- **index_letter** — the problem's slot in the contest (A, B, C... — problems are ordered roughly easiest to hardest)
- **is_versioned** — whether it's an "Easy/Hard Version" pair (these often skew harder)
- **name_length** — rough proxy for statement/flavor-text complexity
- **has_special_tag** — flags CF's `*special` marker problems

## Models Compared

| Model               | Test Accuracy | Weighted F1 | 5-fold CV Accuracy |
|----------------------|:---:|:---:|:---:|
| Logistic Regression   | 68.9% | 0.685 | 65.5% |
| Decision Tree         | 58.9% | 0.582 | 66.8% |
| **Random Forest**     | **72.2%** | **0.715** | **70.1%** |
| Gradient Boosting     | 66.7% | 0.669 | 68.6% |

**Random Forest** was selected as the final model based on held-out weighted F1.

Full per-class precision/recall and a confusion matrix are generated in
`models/confusion_matrix.png`.

## Most Important Features

Contest **points**, **number of tags**, and **problem name length** were the
strongest predictors — followed closely by the problem's **index letter**
(A/B/C...), confirming that CF's own contest ordering carries real difficulty
signal. See `models/feature_importance.png`.

## Setup

```bash
pip install -r requirements.txt

# 1. (Already done) Raw data lives in data/raw_codeforces_response.json
#    To refresh with the latest problems, replace this file with a fresh
#    pull from https://codeforces.com/api/problemset.problems

# 2. Engineer features
python src/prepare_data.py

# 3. Train and compare models
python src/train_models.py

# 4. Run inference on a new problem
python src/predict.py
```

## Honest Limitations

- Dataset uses only problems with a public numeric rating (449 of ~459 pulled);
  unrated/interactive-only problems were excluded.
- Metadata-only features cap achievable accuracy — the actual problem
  statement text (not used here) would likely improve results further via NLP.
- Class imbalance exists (Easy/Medium problems outnumber Expert-tier ones),
  which is reflected in lower recall on the Hard/Expert classes.

## Data Source

Codeforces API — free, public, no authentication required for this data:
https://codeforces.com/apiHelp
