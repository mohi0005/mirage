"""
flow.py — Lightweight variable-flow tracker for ML pipeline analysis.

Tracks variable states through the AST in source order:
  - "pre-split"     : data that exists before train_test_split
  - "train-derived" : derived from the training split (X_train, y_train, etc.)
  - "test-derived"  : derived from the test split (X_test, y_test, etc.)
  - "target"        : the target variable (y, target, labels)
  - "features"      : the feature matrix (X, features)

Heuristic-based: handles common sklearn/pandas patterns well rather than
building a full dataflow engine. Wraps everything in try/except to never
crash on unusual code.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Optional


# Common variable names for train/test splits
_TRAIN_NAMES = {"X_train", "x_train", "y_train", "train_X", "train_y", "train_x",
                "X_tr", "x_tr", "y_tr", "train_data", "train_features", "train_labels"}
_TEST_NAMES = {"X_test", "x_test", "y_test", "test_X", "test_y", "test_x",
               "X_te", "x_te", "y_te", "test_data", "test_features", "test_labels",
               "X_val", "x_val", "y_val", "val_X", "val_y"}
_TARGET_NAMES = {"y", "target", "labels", "y_col", "label"}
_FEATURE_NAMES = {"X", "features", "X_data", "feature_matrix"}

# Preprocessing classes that cause leakage if fit on full data
PREPROCESSING_CLASSES = {
    "StandardScaler", "MinMaxScaler", "RobustScaler", "MaxAbsScaler",
    "Normalizer", "QuantileTransformer", "PowerTransformer",
    "SimpleImputer", "KNNImputer", "IterativeImputer",
    "OneHotEncoder", "OrdinalEncoder", "LabelEncoder", "LabelBinarizer",
    "PCA", "TruncatedSVD", "KernelPCA", "IncrementalPCA",
    "SelectKBest", "SelectPercentile", "SelectFromModel",
    "RFE", "RFECV", "VarianceThreshold", "GenericUnivariateSelect",
    "TfidfVectorizer", "CountVectorizer", "HashingVectorizer",
    "TfidfTransformer",
    "PolynomialFeatures",
}

# Resampling classes (imbalanced-learn)
RESAMPLING_CLASSES = {
    "SMOTE", "ADASYN", "BorderlineSMOTE", "SVMSMOTE", "KMeansSMOTE",
    "RandomOverSampler", "RandomUnderSampler", "SMOTEENN", "SMOTETomek",
    "EditedNearestNeighbours", "TomekLinks", "NearMiss",
    "CondensedNearestNeighbour", "OneSidedSelection",
    "NeighbourhoodCleaningRule",
}

# Randomised estimators / splitters that should have random_state
RANDOMISED_CLASSES = {
    "RandomForestClassifier", "RandomForestRegressor",
    "GradientBoostingClassifier", "GradientBoostingRegressor",
    "ExtraTreesClassifier", "ExtraTreesRegressor",
    "AdaBoostClassifier", "AdaBoostRegressor",
    "BaggingClassifier", "BaggingRegressor",
    "DecisionTreeClassifier", "DecisionTreeRegressor",
    "SGDClassifier", "SGDRegressor",
    "KMeans", "MiniBatchKMeans",
    "LogisticRegression", "MLPClassifier", "MLPRegressor",
    "XGBClassifier", "XGBRegressor",
    "LGBMClassifier", "LGBMRegressor",
    "CatBoostClassifier", "CatBoostRegressor",
    "KFold", "StratifiedKFold", "ShuffleSplit", "StratifiedShuffleSplit",
}

# Hyperparameter search classes
HP_SEARCH_CLASSES = {
    "GridSearchCV", "RandomizedSearchCV", "BayesSearchCV",
    "HalvingGridSearchCV", "HalvingRandomSearchCV",
}


@dataclass
class VarInfo:
    """Metadata about a tracked variable."""
    state: str  # "pre-split", "train-derived", "test-derived", "target", "features", "unknown"
    line: int = 0
    is_preprocessor: bool = False
    preprocessor_class: str = ""
    is_resampler: bool = False
    resampler_class: str = ""
    is_fitted: bool = False
    fitted_on: str = ""  # state of data it was fit on
    is_pipeline: bool = False
    is_hp_search: bool = False


@dataclass
class FlowTracker:
    """Track variable states through an AST.

    Usage:
        tracker = FlowTracker()
        tracker.analyze(tree)
        print(tracker.vars)
    """
    vars: dict[str, VarInfo] = field(default_factory=dict)
    split_line: Optional[int] = -1  # line where train_test_split is called, -1 if not found
    has_split: bool = False
    has_time_column: bool = False
    time_evidence_line: int = 0
    target_column: Optional[str] = None  # name of the target column if detected
    imports: dict[str, str] = field(default_factory=dict)  # alias → real name
    # Track method calls: list of (line, obj_name, method_name, args_info)
    method_calls: list[tuple[int, str, str, dict]] = field(default_factory=list)
    # Track function calls: list of (line, func_name, args_info)
    func_calls: list[tuple[int, str, dict]] = field(default_factory=list)
    # Track metrics used
    metrics_used: set[str] = field(default_factory=set)
    # Track df.drop() calls to know if target was dropped
    drop_calls: list[tuple[int, str, list[str]]] = field(default_factory=list)
    # Pipeline objects
    pipeline_vars: set[str] = field(default_factory=set)
    # Variables used in cross_val_score / cross_validate
    cross_val_data: set[str] = field(default_factory=set)

    def analyze(self, tree: ast.Module) -> None:
        """Walk the AST in source order and track variable states."""
        try:
            self._walk(tree)
        except Exception:
            # Never crash on unusual code
            pass

    def _walk(self, node: ast.AST) -> None:
        """Process AST nodes in source order."""
        for child in ast.iter_child_nodes(node):
            try:
                if isinstance(child, (ast.Import, ast.ImportFrom)):
                    self._handle_import(child)
                elif isinstance(child, ast.Assign):
                    self._handle_assign(child)
                elif isinstance(child, ast.Expr):
                    if isinstance(child.value, ast.Call):
                        self._handle_call_expr(child.value, child.lineno)
                elif isinstance(child, (ast.For, ast.While, ast.If, ast.With)):
                    self._walk(child)
                elif isinstance(child, ast.FunctionDef):
                    self._walk(child)
            except Exception:
                continue

    def _handle_import(self, node: ast.AST) -> None:
        """Record import aliases."""
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name
                self.imports[name] = alias.name
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name
                self.imports[name] = f"{module}.{alias.name}" if module else alias.name

    def _handle_assign(self, node: ast.Assign) -> None:
        """Process assignment statements."""
        line = node.lineno
        value = node.value

        # Check for train_test_split unpacking:
        # X_train, X_test, y_train, y_test = train_test_split(...)
        if isinstance(value, ast.Call) and self._is_call_to(value, "train_test_split"):
            self._handle_split(node, line)
            return

        # Check for DataFrame operations that reveal time columns
        self._check_time_ops(node, line)

        # Single target assignment
        for target in node.targets:
            self._assign_value(target, value, line)

    def _handle_split(self, node: ast.Assign, line: int) -> None:
        """Handle train_test_split call and mark resulting variables."""
        self.has_split = True
        self.split_line = line

        call = node.value
        # Record args info for LG007 check
        args_info = self._extract_call_info(call)
        self.func_calls.append((line, "train_test_split", args_info))

        # Handle tuple unpacking
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Tuple):
            elts = node.targets[0].elts
            for i, elt in enumerate(elts):
                if isinstance(elt, ast.Name):
                    name = elt.id
                    # Even indices are X (train, test), odd are y (train, test)
                    # Common pattern: X_train, X_test, y_train, y_test
                    if name.lower() in {n.lower() for n in _TRAIN_NAMES}:
                        self.vars[name] = VarInfo(state="train-derived", line=line)
                    elif name.lower() in {n.lower() for n in _TEST_NAMES}:
                        self.vars[name] = VarInfo(state="test-derived", line=line)
                    else:
                        # Guess based on position: 0=train, 1=test, 2=train, 3=test
                        state = "train-derived" if i % 2 == 0 else "test-derived"
                        self.vars[name] = VarInfo(state=state, line=line)
        elif len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            # result = train_test_split(...)  — single variable, probably unpacked later
            self.vars[node.targets[0].id] = VarInfo(state="pre-split", line=line)

    def _assign_value(self, target: ast.AST, value: ast.AST, line: int) -> None:
        """Assign a value to a target, inferring state."""
        if isinstance(target, ast.Name):
            name = target.id
            info = self._infer_value_info(value, line)
            self.vars[name] = info

            # Track method calls on this value
            if isinstance(value, ast.Call):
                self._record_call(value, line, assign_target=name)
        elif isinstance(target, ast.Tuple):
            # Tuple unpacking (not train_test_split)
            for elt in target.elts:
                if isinstance(elt, ast.Name):
                    self.vars[elt.id] = VarInfo(state="unknown", line=line)

    def _infer_value_info(self, value: ast.AST, line: int) -> VarInfo:
        """Infer VarInfo from an AST value node."""
        # Constructor call: SomeClass(...)
        if isinstance(value, ast.Call):
            class_name = self._get_call_name(value)

            # Pipeline detection
            if class_name in ("Pipeline", "make_pipeline"):
                return VarInfo(state="unknown", line=line, is_pipeline=True)

            # HP search detection
            if class_name in HP_SEARCH_CLASSES:
                return VarInfo(state="unknown", line=line, is_hp_search=True)

            # Preprocessing class
            if class_name in PREPROCESSING_CLASSES:
                return VarInfo(state="unknown", line=line,
                               is_preprocessor=True, preprocessor_class=class_name)

            # Resampling class
            if class_name in RESAMPLING_CLASSES:
                return VarInfo(state="unknown", line=line,
                               is_resampler=True, resampler_class=class_name)

            # Randomised estimator
            if class_name in RANDOMISED_CLASSES:
                args_info = self._extract_call_info(value)
                info = VarInfo(state="unknown", line=line)
                self.func_calls.append((line, class_name, args_info))
                return info

            # Method call on a known variable: var.method(...)
            if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute):
                obj_name = self._get_name(value.func.value)
                method_name = value.func.attr
                if obj_name and obj_name in self.vars:
                    parent_state = self.vars[obj_name].state
                    return VarInfo(state=parent_state, line=line)

            # Function call that takes a known variable as first arg
            if value.args:
                first_arg = self._get_name(value.args[0])
                if first_arg and first_arg in self.vars:
                    return VarInfo(state=self.vars[first_arg].state, line=line)

        # Variable reference
        if isinstance(value, ast.Name):
            if value.id in self.vars:
                src = self.vars[value.id]
                return VarInfo(state=src.state, line=line)

        # Subscript: df["col"] or df[cols]
        if isinstance(value, ast.Subscript):
            obj_name = self._get_name(value.value)
            if obj_name and obj_name in self.vars:
                return VarInfo(state=self.vars[obj_name].state, line=line)

        # Attribute access: obj.attr
        if isinstance(value, ast.Attribute):
            obj_name = self._get_name(value.value)
            if obj_name and obj_name in self.vars:
                return VarInfo(state=self.vars[obj_name].state, line=line)

        return VarInfo(state="pre-split", line=line)

    def _record_call(self, call: ast.Call, line: int, assign_target: str = "") -> None:
        """Record a function or method call for rule checking."""
        if isinstance(call.func, ast.Attribute):
            obj_name = self._get_name(call.func.value)
            method = call.func.attr
            args_info = self._extract_call_info(call)
            if obj_name:
                self.method_calls.append((line, obj_name, method, args_info))

                # Track fit/fit_transform calls
                if method in ("fit", "fit_transform"):
                    if obj_name in self.vars:
                        self.vars[obj_name].is_fitted = True
                        # What was it fit on?
                        if call.args:
                            arg_name = self._get_name(call.args[0])
                            if arg_name and arg_name in self.vars:
                                self.vars[obj_name].fitted_on = self.vars[arg_name].state

                # Track .drop() calls for target leakage detection
                if method == "drop":
                    dropped = self._extract_drop_columns(call)
                    self.drop_calls.append((line, obj_name, dropped))

                # Track resampling fit_resample / fit_sample
                if method in ("fit_resample", "fit_sample"):
                    self.method_calls.append((line, obj_name, method, args_info))

                # Track metrics
                if method in ("score",):
                    pass  # handled in rules

        elif isinstance(call.func, ast.Name):
            func_name = call.func.id
            args_info = self._extract_call_info(call)
            self.func_calls.append((line, func_name, args_info))

            # Track metric function calls
            metric_funcs = {"accuracy_score", "precision_score", "recall_score",
                            "f1_score", "roc_auc_score", "classification_report",
                            "confusion_matrix", "mean_squared_error", "r2_score",
                            "mean_absolute_error", "log_loss", "balanced_accuracy_score"}
            if func_name in metric_funcs:
                self.metrics_used.add(func_name)

            # Track cross_val_score / cross_validate
            if func_name in ("cross_val_score", "cross_validate"):
                self.cross_val_data.update(
                    self._get_name(a) for a in call.args if self._get_name(a)
                )

    def _handle_call_expr(self, call: ast.Call, line: int) -> None:
        """Handle standalone call expressions (not assigned to anything)."""
        self._record_call(call, line)

    def _check_time_ops(self, node: ast.Assign, line: int) -> None:
        """Check for time-related operations indicating time series data."""
        source_segment = ast.dump(node)
        time_indicators = ["parse_dates", "to_datetime", "DatetimeIndex",
                           "date", "time", "timestamp", "datetime"]
        for indicator in time_indicators:
            if indicator.lower() in source_segment.lower():
                self.has_time_column = True
                self.time_evidence_line = line
                break

    def _is_call_to(self, call: ast.Call, func_name: str) -> bool:
        """Check if a Call node calls the given function name."""
        return self._get_call_name(call) == func_name

    def _get_call_name(self, call: ast.Call) -> str:
        """Get the simple name of a function being called."""
        if isinstance(call.func, ast.Name):
            return call.func.id
        if isinstance(call.func, ast.Attribute):
            return call.func.attr
        return ""

    def _get_name(self, node: ast.AST) -> Optional[str]:
        """Extract a simple name from a Name node."""
        if isinstance(node, ast.Name):
            return node.id
        return None

    def _extract_call_info(self, call: ast.Call) -> dict:
        """Extract argument info from a function call."""
        info: dict = {"arg_names": [], "kwarg_names": []}
        for arg in call.args:
            name = self._get_name(arg)
            info["arg_names"].append(name)
        for kw in call.keywords:
            info["kwarg_names"].append(kw.arg)
            if kw.arg == "random_state":
                info["has_random_state"] = True
            if kw.arg == "shuffle":
                info["has_shuffle"] = True
                # Try to get the value
                if isinstance(kw.value, ast.Constant):
                    info["shuffle_value"] = kw.value.value
        return info

    def _extract_drop_columns(self, call: ast.Call) -> list[str]:
        """Extract column names from a .drop() call."""
        cols: list[str] = []
        # df.drop("col", ...) or df.drop(["col1", "col2"], ...)
        if call.args:
            first = call.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                cols.append(first.value)
            elif isinstance(first, ast.List):
                for elt in first.elts:
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                        cols.append(elt.value)
        # df.drop(columns="col") or df.drop(columns=["col"])
        for kw in call.keywords:
            if kw.arg == "columns":
                if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                    cols.append(kw.value.value)
                elif isinstance(kw.value, ast.List):
                    for elt in kw.value.elts:
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                            cols.append(elt.value)
        return cols

    def get_var_state(self, name: str) -> str:
        """Get the state of a variable, defaulting to 'unknown'."""
        if name in self.vars:
            return self.vars[name].state
        # Check common name patterns
        if name.lower() in {n.lower() for n in _TRAIN_NAMES}:
            return "train-derived"
        if name.lower() in {n.lower() for n in _TEST_NAMES}:
            return "test-derived"
        return "unknown"
