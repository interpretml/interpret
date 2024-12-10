# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

import logging
from abc import abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.base import ClassifierMixin, RegressorMixin, is_classifier
from sklearn.tree import DecisionTreeClassifier as SKDT
from sklearn.tree import DecisionTreeRegressor as SKRT
from sklearn.tree import _tree
from sklearn.utils.validation import check_is_fitted

from ..api.base import ExplainerMixin, ExplanationMixin
from ..utils._clean_simple import clean_dimensions, typify_classification
from ..utils._clean_x import preclean_X
from ..utils._explanation import (
    gen_global_selector,
    gen_local_selector,
    gen_name_from_class,
    gen_perf_dicts,
)
from ..utils._unify_data import unify_data

_log = logging.getLogger(__name__)

COLORS = ["#1f77b4", "#ff7f0e", "#808080", "#3a729b", "#ff420e"]


class TreeExplanation(ExplanationMixin):
    """Explanation object specific to trees."""

    explanation_type = None

    def __init__(
        self,
        explanation_type,
        internal_obj,
        feature_names=None,
        feature_types=None,
        name=None,
        selector=None,
    ):
        """Initializes class.

        Args:
            explanation_type:  Type of explanation.
            internal_obj: A jsonable object that backs the explanation.
            feature_names: List of feature names.
            feature_types: List of feature types.
            name: User-defined name of explanation.
            selector: A dataframe whose indices correspond to explanation entries.
        """

        self.explanation_type = explanation_type
        self._internal_obj = internal_obj
        self.feature_names = feature_names
        self.feature_types = feature_types
        self.name = name
        self.selector = selector

    def data(self, key=None):
        """Provides specific explanation data.

        Args:
            key: A number/string that references a specific data item.

        Returns:
            A serializable dictionary.
        """
        if key is None:
            return self._internal_obj["overall"]
        return self._internal_obj["specific"][key]

    def visualize(self, key=None):
        """Provides interactive visualizations.

        Args:
            key: Either a scalar or list
                that indexes the internal object for sub-plotting.
                If an overall visualization is requested, pass None.

        Returns:
            A Dash Cytoscape object.
        """
        import dash_cytoscape as cyto

        data_dict = self.data(key)
        if data_dict is None:
            return None

        stylesheet = [
            {
                "selector": "edge",
                "style": {
                    "label": "data(label)",
                    "line-color": COLORS[0],
                    "width": "data(edge_weight)",
                    "line-style": "dotted",
                },
            },
            {
                "selector": "node",
                "style": {
                    "label": "data(label)",
                    "text-wrap": "wrap",
                    "background-color": COLORS[3],
                    "font-size": 20,
                    "font-weight": 500,
                },
            },
            {
                "selector": "[weight > 1]",
                "style": {"line-color": COLORS[1], "background-color": COLORS[4]},
            },
        ]

        # Handle overall graphs
        if key is None:
            return cyto.Cytoscape(
                layout={"name": "breadthfirst", "roots": '[id = "1"]'},
                style={"width": "100%", "height": "390px"},
                # userZoomingEnabled=False,
                elements=data_dict["nodes"] + data_dict["edges"],
                stylesheet=stylesheet,
            )

        # Handle local instance graphs
        if self.explanation_type == "local":
            edges = data_dict["edges"]
            nodes = data_dict["nodes"]
            new_edges = self._weight_edges(edges, data_dict["decision"])
            new_nodes = self._weight_nodes_decision(nodes, data_dict["decision"])
            return cyto.Cytoscape(
                layout={"name": "breadthfirst", "roots": '[id = "1"]'},
                style={"width": "100%", "height": "390px"},
                elements=new_nodes + new_edges,
                stylesheet=stylesheet,
            )
        # Handle global feature graphs
        if self.explanation_type == "global":
            feature = self.feature_names[key]
            nodes = data_dict["nodes"]

            feature_present = np.any(
                [feature == node["data"]["feature"] for node in nodes]
            )
            if not feature_present:
                figure = r"""
                        <style>
                        .center {{
                            position: absolute;
                            left: 50%;
                            top: 50%;
                            -webkit-transform: translate(-50%, -50%);
                            transform: translate(-50%, -50%);
                        }}
                        </style>
                        <div class='center'><h1>"{0}" is not used by this tree.</h1></div>
                    """
                return figure.format(feature)

            new_nodes = self._weight_nodes_feature(nodes, feature)
            elements = new_nodes + data_dict["edges"]
            return cyto.Cytoscape(
                layout={"name": "breadthfirst", "roots": '[id = "1"]'},
                style={"width": "100%", "height": "390px"},
                elements=elements,
                stylesheet=stylesheet,
            )
        # pragma: no cover
        msg = f"Cannot handle type {self.explanation_type}"
        _log.error(msg)
        raise Exception(msg)

    def _weight_edges(self, edges, decision_nodes):
        edges = deepcopy(edges)

        new_edges = []
        for edge in edges:
            source = int(edge["data"]["source"])
            target = int(edge["data"]["target"])
            if source in decision_nodes and target in decision_nodes:
                edge["data"]["weight"] = 2
            else:
                edge["data"]["weight"] = 1
            new_edges.append(edge)

        return new_edges

    def _weight_nodes_decision(self, nodes, decision_nodes):
        nodes = deepcopy(nodes)

        new_nodes = []
        for node in nodes:
            node_id = int(node["data"]["id"])
            if node_id in decision_nodes:
                node["data"]["weight"] = 2
            else:
                node["data"]["weight"] = 1
            new_nodes.append(node)

        return new_nodes

    # TODO: Consider removing later, potentially dead code.
    def _weight_nodes_feature(self, nodes, feature_name):
        nodes = deepcopy(nodes)

        new_nodes = []
        for node in nodes:
            feature = node["data"]["feature"]
            if feature == feature_name:
                node["data"]["weight"] = 2
            else:
                node["data"]["weight"] = 1
            new_nodes.append(node)

        return new_nodes


@dataclass
class TreeInputTags:
    one_d_array: bool = False
    two_d_array: bool = True
    three_d_array: bool = False
    sparse: bool = True
    categorical: bool = False
    string: bool = True
    dict: bool = True
    positive_only: bool = False
    allow_nan: bool = True
    pairwise: bool = False


@dataclass
class TreeTargetTags:
    required: bool = True
    one_d_labels: bool = True
    two_d_labels: bool = False
    positive_only: bool = False
    multi_output: bool = False
    single_output: bool = True


@dataclass
class TreeClassifierTags:
    poor_score: bool = False
    multi_class: bool = True
    multi_label: bool = False


@dataclass
class TreeRegressorTags:
    poor_score: bool = False


@dataclass
class TreeTags:
    estimator_type: Optional[str] = None
    target_tags: TreeTargetTags = field(default_factory=TreeTargetTags)
    transformer_tags: None = None
    classifier_tags: Optional[TreeClassifierTags] = None
    regressor_tags: Optional[TreeRegressorTags] = None
    array_api_support: bool = True
    no_validation: bool = False
    non_deterministic: bool = False
    requires_fit: bool = True
    _skip_test: bool = False
    input_tags: TreeInputTags = field(default_factory=TreeInputTags)


class BaseShallowDecisionTree(ExplainerMixin):
    """Shallow Decision Tree (low depth).

    Currently wrapper around DecisionTreeClassifier or DecisionTreeRegressor in scikit-learn.
    To keep the tree shallow, max depth is defaulted to 3.

    https://github.com/scikit-learn/scikit-learn

    """

    available_explanations = ["global", "local"]
    explainer_type = "model"

    def __init__(self, feature_names=None, feature_types=None, max_depth=3, **kwargs):
        """Initializes tree with low depth.

        Args:
            feature_names: List of feature names.
            feature_types: List of feature types.
            max_depth: Max depth of tree.
            **kwargs: Kwargs sent to __init__() method of tree.
        """
        self.feature_names = feature_names
        self.feature_types = feature_types
        self.max_depth = max_depth
        self.kwargs = kwargs

    @abstractmethod
    def _model(self):
        # This method should be overriden
        return None

    def fit(self, X, y, sample_weight=None, check_input=True):
        """Fits model to provided instances.

        Args:
            X: Numpy array for training instances.
            y: Numpy array as training labels.
            sample_weight (optional[np.ndarray]): (n_samples,) Sample weights. If None (default), then samples are equally weighted. Splits that would create child nodes with net zero or negative weight are ignored while searching for a split in each node.
            check_input (bool): default=True. Allow to bypass several input checking. Don't use this parameter unless you know what you're doing.

        Returns:
            Itself.
        """

        y = clean_dimensions(y, "y")
        if y.ndim != 1:
            msg = "y must be 1 dimensional"
            raise ValueError(msg)
        if len(y) == 0:
            msg = "y cannot have 0 samples"
            raise ValueError(msg)

        if is_classifier(self):
            y = typify_classification(y)
        else:
            y = y.astype(np.float64, copy=False)

        X, n_samples = preclean_X(X, self.feature_names, self.feature_types, len(y))

        X, self.feature_names_in_, self.feature_types_in_ = unify_data(
            X, n_samples, self.feature_names, self.feature_types, True, 0
        )

        model = self._model()
        model.fit(X, y, sample_weight=sample_weight, check_input=check_input)

        unique_val_counts = np.zeros(len(self.feature_names_in_), dtype=np.int64)
        for col_idx in range(len(self.feature_names_in_)):
            X_col = X[:, col_idx]
            unique_val_counts[col_idx] = len(np.unique(X_col))

        feat_imp = model.feature_importances_
        self.global_selector_ = gen_global_selector(
            len(self.feature_names_in_),
            self.feature_names_in_,
            self.feature_types_in_,
            unique_val_counts,
            feat_imp,
        )
        self.n_samples_ = n_samples

        self.n_features_in_ = len(self.feature_names_in_)
        if is_classifier(self):
            self.classes_ = model.classes_

        self.has_fitted_ = True

        return self

    def predict(self, X):
        """Predicts on provided instances.

        Args:
            X: Numpy array for instances.

        Returns:
            Predicted class label per instance.
        """

        check_is_fitted(self, "has_fitted_")

        X, n_samples = preclean_X(X, self.feature_names_in_, self.feature_types_in_)
        X, _, _ = unify_data(
            X, n_samples, self.feature_names_in_, self.feature_types_in_, False, 0
        )

        return self._model().predict(X)

    def explain_global(self, name=None):
        """Provides global explanation for model.

        Args:
            name: User-defined explanation name.

        Returns:
            An explanation object,
            visualizing feature-value pairs as horizontal bar chart.
        """

        check_is_fitted(self, "has_fitted_")

        if name is None:
            name = gen_name_from_class(self)

        # Extract decision tree structure
        nodes, edges = self._graph_from_tree(
            self._model(), self.feature_names_in_, max_depth=self.max_depth
        )
        overall_data_dict = {
            "type": "tree",
            "features": self.feature_names_in_,
            "nodes": nodes,
            "edges": edges,
        }
        data_dicts = [
            {
                "type": "tree",
                "features": self.feature_names_in_,
                "nodes": nodes,
                "edges": edges,
            }
            for _ in self.feature_names_in_
        ]

        internal_obj = {"overall": overall_data_dict, "specific": data_dicts}

        return TreeExplanation(
            "global",
            internal_obj,
            feature_names=self.feature_names_in_,
            feature_types=self.feature_types_in_,
            name=name,
            selector=self.global_selector_,
        )

    def explain_local(self, X, y=None, name=None):
        """Provides local explanations for provided instances.

        Args:
            X: Numpy array for X to explain.
            y: Numpy vector for y to explain.
            name: User-defined explanation name.

        Returns:
            An explanation object.
        """

        check_is_fitted(self, "has_fitted_")

        if name is None:
            name = gen_name_from_class(self)

        n_samples = None
        if y is not None:
            y = clean_dimensions(y, "y")
            if y.ndim != 1:
                msg = "y must be 1 dimensional"
                raise ValueError(msg)
            n_samples = len(y)

            if is_classifier(self):
                y = typify_classification(y)
            else:
                y = y.astype(np.float64, copy=False)

        X, n_samples = preclean_X(
            X, self.feature_names_in_, self.feature_types_in_, n_samples
        )

        if n_samples == 0:
            # TODO: we could probably handle this case
            msg = "X has zero samples"
            raise ValueError(msg)

        # Extract decision tree structure
        nodes, edges = self._graph_from_tree(self._model(), self.feature_names_in_)

        X, _, _ = unify_data(
            X, n_samples, self.feature_names_in_, self.feature_types_in_, False, 0
        )

        decisions = [
            self._model().decision_path(instance.reshape(1, -1)).nonzero()[1] + 1
            for instance in X
        ]

        classes = None
        is_classification = is_classifier(self)
        if is_classification:
            classes = self.classes_
            predictions = self.predict_proba(X)
            if len(self.classes_) == 2:
                predictions = predictions[:, 1]
        else:
            predictions = self.predict(X)

        perf_dicts = gen_perf_dicts(predictions, y, is_classification, classes)
        data_dicts = [
            {
                "type": "tree",
                "features": self.feature_names_in_,
                "nodes": nodes,
                "edges": edges,
                "decision": decision,
                "perf": None if perf_dicts is None else perf_dicts[i],
            }
            for i, decision in enumerate(decisions)
        ]

        internal_obj = {"overall": None, "specific": data_dicts}

        selector = gen_local_selector(data_dicts, is_classification=is_classification)
        return TreeExplanation(
            "local",
            internal_obj,
            feature_names=self.feature_names_in_,
            feature_types=self.feature_types_in_,
            name=name,
            selector=selector,
        )

    def _graph_from_tree(self, tree, feature_names=None, max_depth=None):
        """Adapted from:
        https://github.com/scikit-learn/scikit-learn/blob/79bdc8f711d0af225ed6be9fdb708cea9f98a910/sklearn/tree/export.py
        """
        tree_ = tree.tree_
        nodes = []
        edges = []
        max_samples = self.n_samples_
        counter = {"node": 0}

        # i is the element in the tree_ to create a dict for
        def recur(i, depth=0):
            if max_depth is not None and depth > max_depth:
                return None
            if i == _tree.TREE_LEAF:
                return None

            feature = int(tree_.feature[i])
            threshold = float(tree_.threshold[i])

            if feature == _tree.TREE_UNDEFINED:
                feature = None
                threshold = None
                value = [list(map(int, x)) for x in tree_.value[i].tolist()]
            else:
                value = [list(map(int, x)) for x in tree_.value[i].tolist()]
                if feature_names is not None:
                    feature = feature_names[feature]

            counter["node"] += 1
            node_id = str(counter["node"])
            value_str = "# Obs: " if is_classifier(self) else "E[Y]: "

            if feature is not None and threshold is not None:
                value_str += ", ".join([str(v) for v in value[0]])
                label_str = f"{feature} <= {threshold:.2f}\n{value_str}"
            else:
                value_str += ", ".join([str(v) for v in value[0]])
                label_str = f"Impurity: {tree_.impurity[i]:.2f}\n{value_str}"

            nodes.append(
                {"data": {"id": node_id, "label": label_str, "feature": feature}}
            )
            left = recur(tree_.children_left[i], depth + 1)
            right = recur(tree_.children_right[i], depth + 1)
            if left is not None:
                data_left = {
                    "data": {
                        "source": node_id,
                        "target": left["node_id"],
                        "edge_weight": left["n_node_samples"] / max_samples * 15,
                    }
                }
                edges.append(data_left)

            if right is not None:
                data_right = {
                    "data": {
                        "source": node_id,
                        "target": right["node_id"],
                        "edge_weight": right["n_node_samples"] / max_samples * 15,
                    }
                }
                edges.append(data_right)

            return {
                "node_id": node_id,
                "feature": feature,
                "threshold": threshold,
                "impurity": float(tree_.impurity[i]),
                "n_node_samples": int(tree_.n_node_samples[i]),
                "left": left,
                "right": right,
                "value": value,
            }

        recur(0)
        return nodes, edges

    def __sklearn_tags__(self):
        return TreeTags()


class RegressionTree(RegressorMixin, BaseShallowDecisionTree):
    """Regression tree with shallow depth."""

    def __init__(self, feature_names=None, feature_types=None, max_depth=3, **kwargs):
        """Initializes tree with low depth.

        Args:
            feature_names: List of feature names.
            feature_types: List of feature types.
            max_depth: Max depth of tree.
            **kwargs: Kwargs sent to __init__() method of tree.
        """
        super().__init__(
            feature_names=feature_names,
            feature_types=feature_types,
            max_depth=max_depth,
            **kwargs,
        )

    def _model(self):
        return self.sk_model_

    def fit(self, X, y, sample_weight=None, check_input=True):
        """Fits model to provided instances.

        Args:
            X: Numpy array for training instances.
            y: Numpy array as training labels.
            sample_weight (optional[np.ndarray]): (n_samples,) Sample weights. If None (default), then samples are equally weighted. Splits that would create child nodes with net zero or negative weight are ignored while searching for a split in each node.
            check_input (bool): default=True. Allow to bypass several input checking. Don't use this parameter unless you know what you're doing.

        Returns:
            Itself.
        """
        self.sk_model_ = SKRT(max_depth=self.max_depth, **self.kwargs)
        return super().fit(
            X,
            y,
            sample_weight=sample_weight,
            check_input=check_input,
        )

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "regressor"
        tags.regressor_tags = TreeRegressorTags()
        return tags


class ClassificationTree(ClassifierMixin, BaseShallowDecisionTree):
    """Classification tree with shallow depth."""

    def __init__(self, feature_names=None, feature_types=None, max_depth=3, **kwargs):
        """Initializes tree with low depth.

        Args:
            feature_names: List of feature names.
            feature_types: List of feature types.
            max_depth: Max depth of tree.
            **kwargs: Kwargs sent to __init__() method of tree.
        """
        super().__init__(
            feature_names=feature_names,
            feature_types=feature_types,
            max_depth=max_depth,
            **kwargs,
        )

    def _model(self):
        return self.sk_model_

    def fit(self, X, y, sample_weight=None, check_input=True):
        """Fits model to provided instances.

        Args:
            X: Numpy array for training instances.
            y: Numpy array as training labels.
            sample_weight (optional[np.ndarray]): (n_samples,) Sample weights. If None (default), then samples are equally weighted. Splits that would create child nodes with net zero or negative weight are ignored while searching for a split in each node.
            check_input (bool): default=True. Allow to bypass several input checking. Don't use this parameter unless you know what you're doing.

        Returns:
            Itself.
        """
        self.sk_model_ = SKDT(max_depth=self.max_depth, **self.kwargs)
        return super().fit(
            X,
            y,
            sample_weight=sample_weight,
            check_input=check_input,
        )

    def predict_proba(self, X):
        """Probability estimates on provided instances.

        Args:
            X: Numpy array for instances.

        Returns:
            Probability estimate of instance for each class.
        """

        check_is_fitted(self, "has_fitted_")

        X, n_samples = preclean_X(X, self.feature_names_in_, self.feature_types_in_)
        X, _, _ = unify_data(
            X, n_samples, self.feature_names_in_, self.feature_types_in_, False, 0
        )

        return self._model().predict_proba(X)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        tags.classifier_tags = TreeClassifierTags()
        return tags
