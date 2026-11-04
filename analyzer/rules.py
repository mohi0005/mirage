"""
rules.py — Deterministic detection rules for ML pipeline issues.

Each rule is a function that takes (tree, source, tracker) and returns a list
of Finding objects. The LLM never decides whether a bug exists — these rules
are purely AST-based.

Rules:
  LG001: Preprocessing fit before split
  LG002: Fit on test data
  LG003: Resampling before split
  LG004: Target leakage
  LG005: Test set used for tuning
  LG006: Shuffled split on time series
  LG007: Missing random_state
  LG008: Accuracy-only evaluation
"""

from __future__ import annotations

import ast
from typing import Optional

from .flow import (
    FlowTracker, PREPROCESSING_CLASSES, RESAMPLING_CLASSES,
    RANDOMISED_CLASSES, HP_SEARCH_CLASSES,
)
from .report import Finding


def run_all_rules(
    tree: ast.Module,
    source: str,
    tracker: FlowTracker,
    cell_map: Optional[dict[int, int]] = None,
) -> list[Finding]:
    """Run all detection rules and return a list of findings.

    Args:
        tree: Parsed AST module.
        source: Original source code string.
        tracker: Populated FlowTracker.
        cell_map: Optional line→cell mapping for notebooks.

    Returns:
        List of Finding objects.
    """
    if tree is None:
        return []

    rules = [
        check_lg001, check_lg002, check_lg003,
        check_lg004, check_lg005, check_lg006,
        check_lg007, check_lg008,
    ]

    findings: list[Finding] = []
    source_lines = source.splitlines()

    for rule_fn in rules:
        try:
            results = rule_fn(tree, source_lines, tracker)
            if cell_map:
                for f in results:
                    if f.cell is None and f.line in cell_map:
                        f.cell = cell_map[f.line]
            findings.extend(results)
        except Exception:
            # Never crash — skip broken rule and continue
            continue

    return findings


def _get_source_line(source_lines: list[str], line: int) -> str:
    """Safely get a source line (1-indexed)."""
    if 1 <= line <= len(source_lines):
        return source_lines[line - 1].rstrip()
    return ""


def _get_snippet(source_lines: list[str], line: int, context: int = 1) -> str:
    """Get a snippet with context lines around the target line."""
    start = max(0, line - 1 - context)
    end = min(len(source_lines), line + context)
    return "\n".join(source_lines[start:end])


# ─────────────────────────────────────────────────────────────────────
# LG001: Preprocessing fit before split
# ─────────────────────────────────────────────────────────────────────

def check_lg001(tree: ast.Module, source_lines: list[str], tracker: FlowTracker) -> list[Finding]:
    """Detect preprocessors fit on pre-split data before train_test_split.

    Pattern: scaler.fit(X) or scaler.fit_transform(X) where X is pre-split,
    and train_test_split is called later.
    """
    findings: list[Finding] = []

    if not tracker.has_split:
        return findings

    # Walk AST to find fit/fit_transform calls on preprocessors BEFORE split
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in ("fit", "fit_transform"):
            continue

        obj_name = _get_attr_name(node.func.value)
        if not obj_name:
            continue

        # Check if this object is a preprocessor
        var_info = tracker.vars.get(obj_name)
        if not var_info or not var_info.is_preprocessor:
            # Also check if it's a pipeline — pipelines are OK
            if var_info and var_info.is_pipeline:
                continue
            # Check if the object name looks like a preprocessor constructor in the same call
            call_name = _get_call_name_from_node(node)
            if call_name not in PREPROCESSING_CLASSES:
                continue

        # Skip if the object is part of a Pipeline
        if var_info and var_info.is_pipeline:
            continue

        line = node.lineno
        # This fit/fit_transform must be BEFORE the split
        if line >= tracker.split_line:
            continue

        # Check what data is being fit on
        if node.args:
            arg_name = _get_attr_name(node.args[0])
            if arg_name:
                arg_state = tracker.get_var_state(arg_name)
                # If it's already train-derived, it's fine
                if arg_state == "train-derived":
                    continue

        class_name = var_info.preprocessor_class if var_info else "preprocessor"
        findings.append(Finding(
            rule_id="LG001",
            title="Preprocessing fit before split",
            severity="HIGH",
            line=line,
            snippet=_get_snippet(source_lines, line),
            why=(f"{class_name}.{node.func.attr}() is called on data before "
                 f"train_test_split (line {tracker.split_line}). This means the "
                 f"preprocessing learns statistics from the test set, causing data leakage."),
            fix_hint=(f"Move {class_name}.{node.func.attr}() to after the split and "
                      f"fit only on the training data. Use .transform() on the test data."),
        ))

    return findings


# ─────────────────────────────────────────────────────────────────────
# LG002: Fit on test data
# ─────────────────────────────────────────────────────────────────────

def check_lg002(tree: ast.Module, source_lines: list[str], tracker: FlowTracker) -> list[Finding]:
    """Detect fit or fit_transform called on test-derived data."""
    findings: list[Finding] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in ("fit", "fit_transform"):
            continue

        # Check what the call is fit on
        if not node.args:
            continue

        arg_name = _get_attr_name(node.args[0])
        if not arg_name:
            continue

        arg_state = tracker.get_var_state(arg_name)
        if arg_state != "test-derived":
            continue

        obj_name = _get_attr_name(node.func.value) or "preprocessor"

        # Skip HP search objects — they are caught by LG005
        var_info = tracker.vars.get(obj_name)
        if var_info and var_info.is_hp_search:
            continue

        # Skip model .fit() calls (not preprocessing) — only flag preprocessors
        # A model.fit(X_test) is unusual but not "fit on test data" leakage
        # in the same sense as a scaler. However, fitting a preprocessor on
        # test data IS leakage, so flag it.
        if var_info and not var_info.is_preprocessor:
            continue

        # Also check if the object name matches a known preprocessor class
        # even if not tracked (e.g. inline constructor)
        if not var_info:
            call_name = _get_call_name_from_node(node)
            if call_name not in PREPROCESSING_CLASSES:
                continue

        line = node.lineno

        findings.append(Finding(
            rule_id="LG002",
            title="Fit on test data",
            severity="HIGH",
            line=line,
            snippet=_get_snippet(source_lines, line),
            why=(f"{obj_name}.{node.func.attr}() is called on {arg_name}, which is "
                 f"test-derived data. Fitting on test data leaks test information "
                 f"into the model or preprocessing."),
            fix_hint=(f"Fit the {obj_name} on training data only, then use "
                      f".transform({arg_name}) on the test data."),
        ))

    return findings


# ─────────────────────────────────────────────────────────────────────
# LG003: Resampling before split
# ─────────────────────────────────────────────────────────────────────

def check_lg003(tree: ast.Module, source_lines: list[str], tracker: FlowTracker) -> list[Finding]:
    """Detect SMOTE or other resampling applied before train_test_split."""
    findings: list[Finding] = []

    if not tracker.has_split:
        return findings

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in ("fit_resample", "fit_sample"):
            continue

        obj_name = _get_attr_name(node.func.value)
        if not obj_name:
            continue

        # Check if this object is a resampler
        var_info = tracker.vars.get(obj_name)
        if not var_info or not var_info.is_resampler:
            continue

        line = node.lineno
        # Must be before the split
        if line >= tracker.split_line:
            continue

        class_name = var_info.resampler_class if var_info else "resampler"
        findings.append(Finding(
            rule_id="LG003",
            title="Resampling before split",
            severity="HIGH",
            line=line,
            snippet=_get_snippet(source_lines, line),
            why=(f"{class_name}.fit_resample() is called before train_test_split "
                 f"(line {tracker.split_line}). Synthetic samples created from test data "
                 f"will leak into the training set, inflating performance."),
            fix_hint=(f"Split first, then apply {class_name} only on the training data."),
        ))

    return findings


# ─────────────────────────────────────────────────────────────────────
# LG004: Target leakage
# ─────────────────────────────────────────────────────────────────────

def check_lg004(tree: ast.Module, source_lines: list[str], tracker: FlowTracker) -> list[Finding]:
    """Detect target column included in feature matrix.

    Looks for patterns like:
        y = df["target"]
        X = df  # without dropping target
    or:
        X = df.drop("other_col", axis=1)  # target still in X
    """
    findings: list[Finding] = []
    target_col: Optional[str] = None
    target_line: int = 0
    feature_var: Optional[str] = None
    feature_line: int = 0
    dropped_cols: set[str] = set()

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not node.targets:
            continue

        target_name = _get_attr_name(node.targets[0])
        if not target_name:
            continue

        # Detect y = df["target_col"]
        if target_name.lower() in ("y", "target", "labels", "label", "y_train", "y_test"):
            if isinstance(node.value, ast.Subscript):
                col = _get_subscript_key(node.value)
                if col:
                    target_col = col
                    target_line = node.lineno

        # Detect X = df or X = df.drop(...)
        if target_name.lower() in ("x", "features", "x_data", "feature_matrix",
                                     "x_train", "x_test"):
            feature_var = target_name
            feature_line = node.lineno

            # Check if it's a .drop() call
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute):
                if node.value.func.attr == "drop":
                    dropped = _extract_drop_columns_from_call(node.value)
                    dropped_cols.update(dropped)

    # If we found both target and feature assignments, check if target was dropped
    if target_col and feature_var and feature_line > 0:
        if target_col not in dropped_cols:
            # Check if there's a drop call that includes the target column
            for _, _, cols in tracker.drop_calls:
                dropped_cols.update(cols)

            if target_col not in dropped_cols:
                findings.append(Finding(
                    rule_id="LG004",
                    title="Target leakage — target column in features",
                    severity="HIGH",
                    line=feature_line,
                    snippet=_get_snippet(source_lines, feature_line),
                    why=(f"The target column '{target_col}' (assigned at line {target_line}) "
                         f"may still be in the feature matrix '{feature_var}'. The model will "
                         f"learn to just read the answer."),
                    fix_hint=(f"Drop the target column before training: "
                              f"X = df.drop('{target_col}', axis=1)"),
                ))

    return findings


# ─────────────────────────────────────────────────────────────────────
# LG005: Test set used for tuning
# ─────────────────────────────────────────────────────────────────────

def check_lg005(tree: ast.Module, source_lines: list[str], tracker: FlowTracker) -> list[Finding]:
    """Detect hyperparameter search scored on test data."""
    findings: list[Finding] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute):
            continue

        method = node.func.attr
        obj_name = _get_attr_name(node.func.value)

        if not obj_name:
            continue

        # Check if it's an HP search object calling .fit() with test data
        var_info = tracker.vars.get(obj_name)
        if var_info and var_info.is_hp_search:
            if method == "fit":
                # GridSearchCV.fit(X_test, y_test) is wrong
                if node.args:
                    arg_name = _get_attr_name(node.args[0])
                    if arg_name and tracker.get_var_state(arg_name) == "test-derived":
                        findings.append(Finding(
                            rule_id="LG005",
                            title="Test set used for hyperparameter tuning",
                            severity="HIGH",
                            line=node.lineno,
                            snippet=_get_snippet(source_lines, node.lineno),
                            why=(f"{obj_name}.fit() is called on test-derived data "
                                 f"({arg_name}). Hyperparameter search should use only "
                                 f"training data with cross-validation."),
                            fix_hint=(f"Use {obj_name}.fit(X_train, y_train) instead. "
                                      f"GridSearchCV already does internal cross-validation."),
                        ))

        # Check for .score() on test data inside loops (manual HP tuning)
        if method == "score":
            if node.args:
                arg_name = _get_attr_name(node.args[0])
                if arg_name and tracker.get_var_state(arg_name) == "test-derived":
                    # Check if we're inside a for loop
                    if _is_inside_loop(tree, node):
                        findings.append(Finding(
                            rule_id="LG005",
                            title="Test set used for tuning (manual loop)",
                            severity="HIGH",
                            line=node.lineno,
                            snippet=_get_snippet(source_lines, node.lineno),
                            why=(f"model.score() is called on test data ({arg_name}) "
                                 f"inside a loop, suggesting manual hyperparameter search "
                                 f"using the test set."),
                            fix_hint=("Use cross-validation on training data for hyperparameter "
                                      "tuning. Keep the test set for final evaluation only."),
                        ))

        # Check for .predict(X_test) inside a loop (manual HP tuning pattern)
        if method == "predict":
            if node.args:
                arg_name = _get_attr_name(node.args[0])
                if arg_name and tracker.get_var_state(arg_name) == "test-derived":
                    if _is_inside_loop(tree, node):
                        findings.append(Finding(
                            rule_id="LG005",
                            title="Test set used for tuning (manual loop)",
                            severity="HIGH",
                            line=node.lineno,
                            snippet=_get_snippet(source_lines, node.lineno),
                            why=(f"model.predict() is called on test data ({arg_name}) "
                                 f"inside a loop, suggesting manual hyperparameter search "
                                 f"using the test set for model selection."),
                            fix_hint=("Use cross-validation on training data for hyperparameter "
                                      "tuning. Keep the test set for final evaluation only."),
                        ))

    return findings


# ─────────────────────────────────────────────────────────────────────
# LG006: Shuffled split on time series
# ─────────────────────────────────────────────────────────────────────

def check_lg006(tree: ast.Module, source_lines: list[str], tracker: FlowTracker) -> list[Finding]:
    """Detect train_test_split with shuffle on time series data."""
    findings: list[Finding] = []

    if not tracker.has_split or not tracker.has_time_column:
        return findings

    # Check if train_test_split has shuffle=False set
    for line, func_name, args_info in tracker.func_calls:
        if func_name != "train_test_split":
            continue

        # If shuffle is explicitly set to False, it's fine
        if args_info.get("has_shuffle") and args_info.get("shuffle_value") is False:
            continue

        findings.append(Finding(
            rule_id="LG006",
            title="Shuffled split on time series data",
            severity="MEDIUM",
            line=line,
            snippet=_get_snippet(source_lines, line),
            why=(f"train_test_split uses shuffle=True (default) but the data appears "
                 f"to have time-based features (detected near line {tracker.time_evidence_line}). "
                 f"Shuffling time series data causes future data to leak into training."),
            fix_hint=("Use train_test_split(..., shuffle=False) or a time-based split "
                      "to preserve temporal order."),
        ))

    return findings


# ─────────────────────────────────────────────────────────────────────
# LG007: Missing random_state
# ─────────────────────────────────────────────────────────────────────

def check_lg007(tree: ast.Module, source_lines: list[str], tracker: FlowTracker) -> list[Finding]:
    """Detect train_test_split or estimators missing random_state."""
    findings: list[Finding] = []

    for line, func_name, args_info in tracker.func_calls:
        if func_name == "train_test_split":
            if not args_info.get("has_random_state"):
                findings.append(Finding(
                    rule_id="LG007",
                    title="Missing random_state in train_test_split",
                    severity="LOW",
                    line=line,
                    snippet=_get_source_line(source_lines, line),
                    why=("train_test_split without random_state produces different splits "
                         "each run, making results non-reproducible."),
                    fix_hint="Add random_state=42 (or any fixed integer) to train_test_split().",
                ))

        elif func_name in RANDOMISED_CLASSES:
            if not args_info.get("has_random_state"):
                findings.append(Finding(
                    rule_id="LG007",
                    title=f"Missing random_state in {func_name}",
                    severity="LOW",
                    line=line,
                    snippet=_get_source_line(source_lines, line),
                    why=(f"{func_name} without random_state produces different results "
                         f"each run, making the pipeline non-reproducible."),
                    fix_hint=f"Add random_state=42 to {func_name}().",
                ))

    return findings


# ─────────────────────────────────────────────────────────────────────
# LG008: Accuracy-only evaluation
# ─────────────────────────────────────────────────────────────────────

def check_lg008(tree: ast.Module, source_lines: list[str], tracker: FlowTracker) -> list[Finding]:
    """Detect when accuracy_score is the only metric used.

    Walks ALL Call nodes in the AST (including nested ones like
    print(accuracy_score(...))) to find metric function calls.
    """
    findings: list[Finding] = []

    metric_funcs = {"accuracy_score", "precision_score", "recall_score",
                    "f1_score", "roc_auc_score", "classification_report",
                    "confusion_matrix", "mean_squared_error", "r2_score",
                    "mean_absolute_error", "log_loss", "balanced_accuracy_score"}

    metrics_found: set[str] = set()
    acc_line = 0

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _get_call_name_from_node(node)
        if name in metric_funcs:
            metrics_found.add(name)
            if name == "accuracy_score" and acc_line == 0:
                acc_line = node.lineno

    if "accuracy_score" not in metrics_found:
        return findings

    other_metrics = metrics_found - {"accuracy_score"}
    if other_metrics:
        return findings  # Has other metrics, fine

    if acc_line:
        findings.append(Finding(
            rule_id="LG008",
            title="Accuracy-only evaluation",
            severity="MEDIUM",
            line=acc_line,
            snippet=_get_source_line(source_lines, acc_line),
            why=("accuracy_score is the only evaluation metric. On imbalanced datasets, "
                 "a model predicting only the majority class can score high accuracy. "
                 "This hides poor performance on the minority class."),
            fix_hint=("Add precision_score, recall_score, f1_score, or "
                      "classification_report for a fuller picture."),
        ))

    return findings


# ─────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────

def _get_attr_name(node: ast.AST) -> Optional[str]:
    """Extract a simple name from Name or Attribute nodes."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return _get_attr_name(node.value)
    return None


def _get_call_name_from_node(node: ast.Call) -> str:
    """Get the class/function name from a Call node."""
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return ""


def _get_subscript_key(node: ast.Subscript) -> Optional[str]:
    """Extract a string key from df["key"]."""
    if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
        return node.slice.value
    return None


def _extract_drop_columns_from_call(call: ast.Call) -> list[str]:
    """Extract column names from a .drop() call."""
    cols: list[str] = []
    if call.args:
        first = call.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            cols.append(first.value)
        elif isinstance(first, ast.List):
            for elt in first.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    cols.append(elt.value)
    for kw in call.keywords:
        if kw.arg == "columns":
            if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                cols.append(kw.value.value)
            elif isinstance(kw.value, ast.List):
                for elt in kw.value.elts:
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                        cols.append(elt.value)
    return cols


def _is_inside_loop(tree: ast.Module, target_node: ast.AST) -> bool:
    """Check if a node is inside a for/while loop (simple heuristic)."""
    class LoopFinder(ast.NodeVisitor):
        def __init__(self) -> None:
            self.found = False
            self._in_loop = False

        def visit_For(self, node: ast.AST) -> None:
            old = self._in_loop
            self._in_loop = True
            self.generic_visit(node)
            self._in_loop = old

        def visit_While(self, node: ast.AST) -> None:
            old = self._in_loop
            self._in_loop = True
            self.generic_visit(node)
            self._in_loop = old

        def visit_Call(self, node: ast.AST) -> None:
            if node is target_node and self._in_loop:
                self.found = True
            self.generic_visit(node)

    finder = LoopFinder()
    finder.visit(tree)
    return finder.found
