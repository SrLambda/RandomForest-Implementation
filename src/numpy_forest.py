"""From-scratch NumPy/Pandas forest for binary service scoring."""

import json
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from numbers import Real
from pathlib import Path

import numpy as np
import pandas as pd


# Una hoja conserva la fracción de positivos; una división añade el umbral y dos ramas.
@dataclass(frozen=True)
class _TreeNode:
    positive_fraction: float
    feature: str | None = None
    threshold: float | None = None
    left: '_TreeNode | None' = None
    right: '_TreeNode | None' = None


class RandomForest:
    """Train binary CART trees for scoring candidate root causes."""

    _EXPORT_VERSION = 1
    _MAX_DEPTH = 8

    def __init__(self, *, n_trees: int = 10, seed: int = 42) -> None:
        if type(n_trees) is not int or n_trees < 1:
            raise ValueError("n_trees must be a positive integer.")
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a nonnegative integer.")
        self.n_trees, self.seed = n_trees, seed

    def _prepare_tree_samples(
        self, X_train: pd.DataFrame, Y_train: pd.DataFrame, *, n_trees: int = 10, seed: int = 42
    ) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        """Draw bootstrap rows and a fixed feature subset for each tree."""
        if n_trees < 1 or X_train.empty or Y_train.empty or not X_train.index.equals(Y_train.index):
            raise ValueError("Provide at least one tree and nonempty, aligned training tables.")

        # Cada árbol recibe filas con reemplazo y un subconjunto fijo de variables sin reemplazo.
        rng = np.random.default_rng(seed)
        n_rows, k = X_train.shape
        n_features_per_tree = int(np.sqrt(k))
        samples = []
        for _ in range(n_trees):
            rows = rng.choice(n_rows, size=n_rows, replace=True)
            features = rng.choice(k, size=n_features_per_tree, replace=False)
            samples.append((X_train.iloc[rows, features].copy(), Y_train.iloc[rows].copy()))
        return samples

    def _build_tree(self, X_tree: pd.DataFrame, y_tree: pd.DataFrame) -> _TreeNode:
        """Grow one binary CART tree using only its assigned features."""
        values = X_tree.to_numpy(dtype=float)
        if y_tree.shape[1] != 1 or not y_tree.isin([0, 1]).to_numpy().all():
            raise ValueError("Provide exactly one binary target column.")
        if not np.isfinite(values).all():
            raise ValueError("Tree features must be finite; impute missing values before fitting.")
        labels = y_tree.iloc[:, 0].to_numpy(dtype=np.int64)
        max_depth, min_leaf = self._MAX_DEPTH, 2

        def grow(rows: np.ndarray, depth: int) -> _TreeNode:
            size = len(rows)
            positives = labels[rows].sum()
            fraction = float(positives / size)
            leaf = _TreeNode(fraction)
            if depth >= max_depth or size < 2 * min_leaf or positives in (0, size):
                return leaf

            # Comparar cortes entre valores distintos mediante la impureza de Gini ponderada.
            best_impurity = 2 * fraction * (1 - fraction)
            best_split = None
            for feature in range(values.shape[1]):
                ordered_rows = rows[np.argsort(values[rows, feature], kind="stable")]
                ordered_values = values[ordered_rows, feature]
                cuts = np.flatnonzero(ordered_values[:-1] < ordered_values[1:]) + 1
                cuts = cuts[(cuts >= min_leaf) & (size - cuts >= min_leaf)]
                if cuts.size == 0:
                    continue

                left_size, right_size = cuts, size - cuts
                left_pos = np.cumsum(labels[ordered_rows])[cuts - 1]
                right_pos = positives - left_pos
                left_rate, right_rate = left_pos / left_size, right_pos / right_size
                impurities = 2 * (
                    left_size * left_rate * (1 - left_rate)
                    + right_size * right_rate * (1 - right_rate)
                ) / size
                candidate = int(np.argmin(impurities))
                if impurities[candidate] < best_impurity - 1e-12:
                    best_impurity = impurities[candidate]
                    best_split = feature, ordered_rows, int(cuts[candidate])

            if best_split is None:
                return leaf
            feature, ordered_rows, cut = best_split
            lower, upper = values[ordered_rows[[cut - 1, cut]], feature]
            threshold = lower / 2 + upper / 2
            # Mantener el corte válido si el punto medio se redondea al valor superior.
            if not lower <= threshold < upper:
                threshold = lower
            return _TreeNode(
                fraction, X_tree.columns[feature], float(threshold),
                grow(ordered_rows[:cut], depth + 1),
                grow(ordered_rows[cut:], depth + 1),
            )

        return grow(np.arange(len(labels)), depth=0)

    def fit(self, X_train: pd.DataFrame, Y_train: pd.DataFrame) -> "RandomForest":
        """Prepare aligned bootstrap samples and build their trees concurrently."""
        schema = self._validate_schema(list(X_train.columns))
        # Comprobar todas las filas antes del muestreo, sin descartar ni imputar valores.
        if any(dtype.kind not in "iuf" for dtype in X_train.dtypes) or not np.isfinite(
            X_train.to_numpy(dtype=float, na_value=np.nan)
        ).all():
            raise ValueError("Training features must be finite real numbers.")
        if Y_train.shape[1] != 1 or not Y_train.isin([0, 1]).to_numpy().all():
            raise ValueError("Provide exactly one binary target column.")
        samples = self._prepare_tree_samples(X_train, Y_train, n_trees=self.n_trees, seed=self.seed)
        with ThreadPoolExecutor() as executor:
            trees = list(executor.map(self._build_tree, *zip(*samples)))
        self._training_features = tuple(tuple(X.columns) for X, _ in samples)
        self._trees, self._feature_schema = trees, schema
        return self

    @property
    def training_features(self) -> tuple[tuple[str, ...], ...]:
        """Features sampled for each tree after fitting (not stored in inference JSON)."""
        self._require_fitted()
        if not hasattr(self, "_training_features"):
            raise RuntimeError("Sampled feature sets are available only on a directly fitted model.")
        return self._training_features

    @staticmethod
    def _validate_schema(names: list) -> tuple[str, ...]:
        if type(names) is not list or not names or any(type(name) is not str or not name for name in names):
            raise ValueError("Feature schema must be a nonempty list of nonempty strings.")
        if len(set(names)) != len(names):
            raise ValueError("Feature names must be unique.")
        return tuple(names)

    def _require_fitted(self) -> None:
        if not getattr(self, "_trees", None):
            raise RuntimeError("Fit or load a forest before using it.")

    @staticmethod
    def _finite_number(value: Real) -> float:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError("Expected a finite real number.")
        try:
            number = float(value)
        except OverflowError as exc:
            raise ValueError("Expected a finite real number.") from exc
        if not math.isfinite(number):
            raise ValueError("Expected a finite real number.")
        return number

    @staticmethod
    def _predict_tree(node: _TreeNode, values: dict[str, float]) -> float:
        # Usar la misma comparación del entrenamiento y devolver únicamente el valor de la hoja.
        while node.feature is not None:
            node = node.left if values[node.feature] <= node.threshold else node.right
        return node.positive_fraction

    def predict_tree_scores(self, x: pd.Series) -> tuple[float, ...]:
        """Return the leaf score reached in each tree, in forest order."""
        self._require_fitted()
        if not isinstance(x, pd.Series):
            raise TypeError("Provide one feature row as a pandas Series.")
        # Validar el conjunto completo de nombres; el orden de la fila no afecta el recorrido.
        if set(self._validate_schema(list(x.index))) != set(self._feature_schema):
            raise ValueError("Feature names must match the fitted schema exactly.")
        values = {name: self._finite_number(value) for name, value in x.items()}
        return tuple(self._predict_tree(tree, values) for tree in self._trees)

    def predict(self, x: pd.Series) -> float:
        """Return the root-cause score for one service feature row."""
        scores = self.predict_tree_scores(x)
        return math.fsum(scores) / len(scores)

    @classmethod
    def _encode_tree(cls, node: _TreeNode) -> dict:
        if node.feature is None:
            return {"score": node.positive_fraction}
        return {"feature": node.feature, "threshold": node.threshold,
                "left": cls._encode_tree(node.left), "right": cls._encode_tree(node.right)}

    def export(self, path: str | Path) -> None:
        """Write inference-only JSON to a new file."""
        self._require_fitted()
        # Guardar solo lo necesario para inferencia; el modo exclusivo evita sobrescrituras.
        payload = {"version": self._EXPORT_VERSION, "features": list(self._feature_schema),
                   "trees": [self._encode_tree(tree) for tree in self._trees]}
        text = json.dumps(payload, allow_nan=False, indent=2)
        with Path(path).open("x", encoding="utf-8") as stream:
            stream.write(text + "\n")

    @classmethod
    def _decode_tree(cls, node: dict, features: set[str], depth: int = 0) -> _TreeNode:
        # Distinguir hojas de divisiones completas y validar cada rama antes de crear el modelo.
        if type(node) is not dict or depth > cls._MAX_DEPTH:
            raise ValueError("Invalid node or tree depth.")
        if node.keys() == {"score"}:
            score = cls._finite_number(node["score"])
            if not 0 <= score <= 1:
                raise ValueError("Leaf score must be in [0, 1].")
            return _TreeNode(score)
        if node.keys() != {"feature", "threshold", "left", "right"}:
            raise ValueError("Invalid split node fields.")
        feature = node["feature"]
        if type(feature) is not str or feature not in features:
            raise ValueError("Split feature is not in the inference schema.")
        threshold = cls._finite_number(node["threshold"])
        return _TreeNode(0.0, feature, threshold,
                         cls._decode_tree(node["left"], features, depth + 1),
                         cls._decode_tree(node["right"], features, depth + 1))

    @classmethod
    def from_export(cls, path: str | Path) -> "RandomForest":
        """Construct a forest from a model saved with export."""
        def unique_keys(pairs):
            fields = dict(pairs)
            if len(fields) != len(pairs):
                raise ValueError("Duplicate JSON fields are not allowed.")
            return fields

        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique_keys)
        except RecursionError as exc:
            raise ValueError("Model JSON is nested too deeply.") from exc
        if type(payload) is not dict or payload.keys() != {"version", "features", "trees"}:
            raise ValueError("Invalid forest export fields.")
        if type(payload["version"]) is not int or payload["version"] != cls._EXPORT_VERSION:
            raise ValueError("Unsupported forest export version.")
        schema = cls._validate_schema(payload["features"])
        if type(payload["trees"]) is not list or not payload["trees"]:
            raise ValueError("Forest export must contain at least one tree.")
        features = set(schema)
        model = cls(n_trees=len(payload["trees"]))
        model._trees = [cls._decode_tree(node, features) for node in payload["trees"]]
        model._feature_schema = schema
        return model
