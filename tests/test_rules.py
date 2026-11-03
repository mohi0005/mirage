"""
test_rules.py — Unit tests for each detection rule.

Each rule has at least one positive case (should trigger) and one negative
case (should NOT trigger) to catch false positives.
"""

from __future__ import annotations

import ast
import pytest
from analyzer.parser import parse_source
from analyzer.flow import FlowTracker
from analyzer.rules import (
    check_lg001, check_lg002, check_lg003,
    check_lg004, check_lg005, check_lg006,
    check_lg007, check_lg008,
    run_all_rules,
)


def _analyze(code: str):
    """Helper: parse code, run flow tracker, return (tree, lines, tracker)."""
    source, cell_map, tree = parse_source(code)
    assert tree is not None, f"Failed to parse:\n{code}"
    tracker = FlowTracker()
    tracker.analyze(tree)
    lines = source.splitlines()
    return tree, lines, tracker


# ─────────────────────────────────────────────────────────────────────
# LG001: Preprocessing fit before split
# ─────────────────────────────────────────────────────────────────────

class TestLG001:
    def test_positive_fit_transform_before_split(self):
        code = '''
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)
'''
        tree, lines, tracker = _analyze(code)
        findings = check_lg001(tree, lines, tracker)
        assert len(findings) >= 1
        assert findings[0].rule_id == "LG001"

    def test_negative_fit_on_train_only(self):
        """Correct pattern: fit on X_train after split — must NOT flag."""
        code = '''
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
'''
        tree, lines, tracker = _analyze(code)
        findings = check_lg001(tree, lines, tracker)
        assert len(findings) == 0

    def test_negative_pipeline_not_flagged(self):
        """Pipeline pattern should NOT be flagged."""
        code = '''
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
scores = cross_val_score(pipe, X, y, cv=5)
'''
        tree, lines, tracker = _analyze(code)
        findings = check_lg001(tree, lines, tracker)
        assert len(findings) == 0


# ─────────────────────────────────────────────────────────────────────
# LG002: Fit on test data
# ─────────────────────────────────────────────────────────────────────

class TestLG002:
    def test_positive_fit_on_xtest(self):
        code = '''
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)
scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(X_test)
'''
        tree, lines, tracker = _analyze(code)
        findings = check_lg002(tree, lines, tracker)
        assert len(findings) >= 1
        assert findings[0].rule_id == "LG002"

    def test_negative_transform_on_test(self):
        """transform() on test data is correct — must NOT flag."""
        code = '''
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)
scaler = StandardScaler()
scaler.fit(X_train)
X_test_scaled = scaler.transform(X_test)
'''
        tree, lines, tracker = _analyze(code)
        findings = check_lg002(tree, lines, tracker)
        assert len(findings) == 0


# ─────────────────────────────────────────────────────────────────────
# LG003: Resampling before split
# ─────────────────────────────────────────────────────────────────────

class TestLG003:
    def test_positive_smote_before_split(self):
        code = '''
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
smote = SMOTE()
X_res, y_res = smote.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res)
'''
        tree, lines, tracker = _analyze(code)
        findings = check_lg003(tree, lines, tracker)
        assert len(findings) >= 1
        assert findings[0].rule_id == "LG003"

    def test_negative_smote_after_split(self):
        """SMOTE on train data after split is correct."""
        code = '''
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)
smote = SMOTE()
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
'''
        tree, lines, tracker = _analyze(code)
        findings = check_lg003(tree, lines, tracker)
        assert len(findings) == 0


# ─────────────────────────────────────────────────────────────────────
# LG004: Target leakage
# ─────────────────────────────────────────────────────────────────────

class TestLG004:
    def test_positive_target_not_dropped(self):
        code = '''
import pandas as pd
df = pd.read_csv("data.csv")
y = df["target"]
X = df
'''
        tree, lines, tracker = _analyze(code)
        findings = check_lg004(tree, lines, tracker)
        assert len(findings) >= 1
        assert findings[0].rule_id == "LG004"

    def test_negative_target_dropped(self):
        """Target properly dropped from features — must NOT flag."""
        code = '''
import pandas as pd
df = pd.read_csv("data.csv")
y = df["target"]
X = df.drop("target", axis=1)
'''
        tree, lines, tracker = _analyze(code)
        findings = check_lg004(tree, lines, tracker)
        assert len(findings) == 0


# ─────────────────────────────────────────────────────────────────────
# LG005: Test set used for tuning
# ─────────────────────────────────────────────────────────────────────

class TestLG005:
    def test_positive_gridsearch_on_test(self):
        code = '''
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
X_train, X_test, y_train, y_test = train_test_split(X, y)
grid = GridSearchCV(SVC(), {"C": [1, 10]})
grid.fit(X_test, y_test)
'''
        tree, lines, tracker = _analyze(code)
        findings = check_lg005(tree, lines, tracker)
        assert len(findings) >= 1
        assert findings[0].rule_id == "LG005"

    def test_negative_gridsearch_on_train(self):
        """GridSearchCV on train data is correct."""
        code = '''
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
X_train, X_test, y_train, y_test = train_test_split(X, y)
grid = GridSearchCV(SVC(), {"C": [1, 10]})
grid.fit(X_train, y_train)
'''
        tree, lines, tracker = _analyze(code)
        findings = check_lg005(tree, lines, tracker)
        assert len(findings) == 0


# ─────────────────────────────────────────────────────────────────────
# LG006: Shuffled split on time series
# ─────────────────────────────────────────────────────────────────────

class TestLG006:
    def test_positive_shuffle_on_timeseries(self):
        code = '''
import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv("data.csv", parse_dates=["date"])
X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y)
'''
        tree, lines, tracker = _analyze(code)
        findings = check_lg006(tree, lines, tracker)
        assert len(findings) >= 1
        assert findings[0].rule_id == "LG006"

    def test_negative_shuffle_false(self):
        """shuffle=False on time series is correct."""
        code = '''
import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv("data.csv", parse_dates=["date"])
X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)
'''
        tree, lines, tracker = _analyze(code)
        findings = check_lg006(tree, lines, tracker)
        assert len(findings) == 0


# ─────────────────────────────────────────────────────────────────────
# LG007: Missing random_state
# ─────────────────────────────────────────────────────────────────────

class TestLG007:
    def test_positive_no_random_state(self):
        code = '''
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)
'''
        tree, lines, tracker = _analyze(code)
        findings = check_lg007(tree, lines, tracker)
        assert len(findings) >= 1
        assert findings[0].rule_id == "LG007"

    def test_negative_has_random_state(self):
        code = '''
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
'''
        tree, lines, tracker = _analyze(code)
        findings = check_lg007(tree, lines, tracker)
        assert len(findings) == 0


# ─────────────────────────────────────────────────────────────────────
# LG008: Accuracy-only evaluation
# ─────────────────────────────────────────────────────────────────────

class TestLG008:
    def test_positive_accuracy_only(self):
        code = '''
from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, y_pred)
'''
        tree, lines, tracker = _analyze(code)
        findings = check_lg008(tree, lines, tracker)
        assert len(findings) >= 1
        assert findings[0].rule_id == "LG008"

    def test_negative_multiple_metrics(self):
        code = '''
from sklearn.metrics import accuracy_score, f1_score
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
'''
        tree, lines, tracker = _analyze(code)
        findings = check_lg008(tree, lines, tracker)
        assert len(findings) == 0


# ─────────────────────────────────────────────────────────────────────
# Notebook parsing
# ─────────────────────────────────────────────────────────────────────

class TestNotebookParsing:
    def test_parse_notebook(self):
        import json
        nb = {
            "cells": [
                {"cell_type": "code", "source": ["import pandas as pd\n", "x = 1\n"]},
                {"cell_type": "markdown", "source": ["# Title"]},
                {"cell_type": "code", "source": ["y = 2"]},
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        text = json.dumps(nb)
        source, cell_map, tree = parse_source(text, "test.ipynb")
        assert tree is not None
        assert 1 in cell_map  # first line maps to cell 1
        assert "import pandas" in source


# ─────────────────────────────────────────────────────────────────────
# Malformed input
# ─────────────────────────────────────────────────────────────────────

class TestMalformedInput:
    def test_syntax_error_does_not_crash(self):
        code = "def broken(:\n  pass"
        source, cell_map, tree = parse_source(code)
        assert tree is None  # Should return None, not crash

    def test_empty_code(self):
        source, cell_map, tree = parse_source("")
        assert tree is not None  # Empty code is valid Python
        tracker = FlowTracker()
        tracker.analyze(tree)
        findings = run_all_rules(tree, "", tracker)
        assert findings == []
