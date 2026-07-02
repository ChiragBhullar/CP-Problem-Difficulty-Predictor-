"""
prepare_data.py
----------------
Loads raw problem data pulled from the Codeforces public API
(problemset.problems endpoint) and turns it into a clean, feature-engineered
CSV ready for model training.

Data source: https://codeforces.com/api/problemset.problems (free, public, no API key)
"""

import json
import re
import pandas as pd

RAW_PATH = "data/raw_codeforces_response.json"
OUT_PATH = "data/problems_features.csv"


def difficulty_bucket(rating: int) -> str:
    """Map a numeric CF rating to a coarse difficulty tier."""
    if rating < 1200:
        return "Easy"
    elif rating < 1900:
        return "Medium"
    elif rating < 2400:
        return "Hard"
    else:
        return "Expert"


def index_letter(index: str) -> str:
    """Problem index like 'C2' or 'F1' -> base letter 'C' / 'F'."""
    match = re.match(r"([A-Za-z]+)", index)
    return match.group(1) if match else index


def load_raw(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    problems = payload["result"]["problems"]
    df = pd.DataFrame(problems)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    # Drop problems with no official rating -- we can't train/evaluate on them
    df = df.dropna(subset=["rating"]).copy()
    df["rating"] = df["rating"].astype(int)

    # Target variable
    df["difficulty_bucket"] = df["rating"].apply(difficulty_bucket)

    # Feature: number of tags (more tags often = more complex problem)
    df["num_tags"] = df["tags"].apply(len)

    # Feature: base index letter (A, B, C... roughly correlates with difficulty
    # within a contest, since problems are ordered easiest to hardest)
    df["index_letter"] = df["index"].apply(index_letter)

    # Feature: whether this is a "version" problem (Easy/Hard Version pairs
    # tend to skew harder due to added constraints)
    df["is_versioned"] = df["name"].str.contains(
        r"\(.*version.*\)", case=False, regex=True
    ).astype(int)

    # Feature: points awarded (proxy for perceived difficulty at contest time)
    df["points"] = df["points"].fillna(df["points"].median())

    # Feature: name length (rough proxy for statement complexity / flavor text)
    df["name_length"] = df["name"].str.len()

    # Feature: does the problem tag list include "*special" (used for div1-only /
    # unrated-style problems on CF, often signals unusual difficulty)
    df["has_special_tag"] = df["tags"].apply(lambda t: int("*special" in t))

    # Multi-hot encode tags (excluding the *special marker, handled separately)
    all_tags = sorted({tag for tags in df["tags"] for tag in tags if tag != "*special"})
    for tag in all_tags:
        df[f"tag_{tag.replace(' ', '_')}"] = df["tags"].apply(lambda t, tag=tag: int(tag in t))

    return df, all_tags


def main():
    df = load_raw(RAW_PATH)
    print(f"Loaded {len(df)} raw problems")

    df, all_tags = engineer_features(df)
    print(f"After dropping unrated problems: {len(df)} rows")
    print(f"Engineered {len(all_tags)} tag features")
    print("\nDifficulty bucket distribution:")
    print(df["difficulty_bucket"].value_counts())

    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved feature dataset to {OUT_PATH}")


if __name__ == "__main__":
    main()
