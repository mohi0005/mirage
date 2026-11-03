"""
prove.py — "Prove it" live experiment.

Trains two scikit-learn pipelines on a built-in dataset with fixed random_state:
  (A) LEAKY:   feature selection fit on ALL data before train_test_split
  (B) CORRECT: split first, feature selection fit on train only

Returns both test scores and the gap, demonstrating that data leakage
inflates apparent performance.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline


def run_prove_experiment() -> dict:
    """Run the leaky vs. correct experiment.

    Returns a dict with leaky_score, correct_score, gap, and description.
    Runtime target: under 5 seconds.
    """
    # Generate a dataset with many noise features to make leakage visible.
    # 200 samples, 50 features, but only 5 are actually informative.
    # SelectKBest on ALL data will "cheat" by finding features that correlate
    # with the target by chance in the test set.
    X, y = make_classification(
        n_samples=200,
        n_features=50,
        n_informative=5,
        n_redundant=0,
        n_classes=2,
        random_state=42,
        shuffle=False,
    )

    # ─── (A) LEAKY PIPELINE: SelectKBest on ALL data before split ───
    selector_leaky = SelectKBest(f_classif, k=10)
    X_selected_all = selector_leaky.fit_transform(X, y)  # fit on ALL data!
    X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(
        X_selected_all, y, test_size=0.3, random_state=42
    )
    model_a = LogisticRegression(random_state=42, max_iter=1000)
    model_a.fit(X_train_a, y_train_a)
    leaky_score = accuracy_score(y_test_a, model_a.predict(X_test_a))

    # ─── (B) CORRECT PIPELINE: split first, SelectKBest on train only ───
    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    selector_correct = SelectKBest(f_classif, k=10)
    X_train_b_sel = selector_correct.fit_transform(X_train_b, y_train_b)  # fit on TRAIN only
    X_test_b_sel = selector_correct.transform(X_test_b)  # transform test
    model_b = LogisticRegression(random_state=42, max_iter=1000)
    model_b.fit(X_train_b_sel, y_train_b)
    correct_score = accuracy_score(y_test_b, model_b.predict(X_test_b_sel))

    gap = leaky_score - correct_score

    return {
        "leaky_score": round(float(leaky_score), 4),
        "correct_score": round(float(correct_score), 4),
        "gap": round(float(gap), 4),
        "gap_percent": round(float(gap) * 100, 1),
        "description": (
            "Same data, same model. The only difference is the leak. "
            f"The leaky pipeline scored {leaky_score:.1%} vs. the correct "
            f"pipeline's {correct_score:.1%} — a {gap:.1%} gap caused by "
            f"fitting feature selection on data that includes the test set."
        ),
        "method": (
            "200 samples, 50 features (only 5 informative), SelectKBest(k=10), "
            "LogisticRegression. Leaky: SelectKBest fit on all data before split. "
            "Correct: split first, SelectKBest fit on training data only."
        ),
    }
