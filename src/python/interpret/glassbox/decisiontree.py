# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

from ..api.base import ExplainerMixin, ExplanationMixin
from ..utils import unify_data
from ..utils import gen_name_from_class, gen_local_selector, gen_global_selector

from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.tree import DecisionTreeClassifier as SKDT
from sklearn.tree import DecisionTreeRegressor as SKRT
from sklearn.base import is_classifier
import numpy as np
from abc import abstractmethod
from sklearn.tree import _tree
from copy import deepcopy
import dash_cytoscape as cyto

import logging

log = logging.getLogger(__name__)

COLORS = ["#1f77b4", "#ff7f0e", "#808080", "#3a729b", "#ff420e"]


class TreeExplanation(ExplanationMixin):
    """
    """

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

        self.explanation_type = explanation_type
        self._internal_obj = internal_obj
        self.feature_names = feature_names
        self.feature_types = feature_types
        self.name = name
        self.selector = selector

    def data(self, key=None):
        if key is None:
            return self._internal_obj["overall"]
        return self._internal_obj["specific"][key]

    def visualize(self, key=None):
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
            component = cyto.Cytoscape(
                layout={"name": "breadthfirst", "roots": '[id = "1"]'},
                style={"width": "100%", "height": "390px"},
                # userZoomingEnabled=False,
                elements=data_dict["nodes"] + data_dict["edges"],
                stylesheet=stylesheet,
            )
            return component

        # Handle local instance graphs
        if self.explanation_type == "local":
            edges = data_dict["edges"]
            nodes = data_dict["nodes"]
            new_edges = self._weight_edges(edges, data_dict["decision"])
            new_nodes = self._weight_nodes_decision(nodes, data_dict["decision"])
            component = cyto.Cytoscape(
                layout={"name": "breadthfirst", "roots": '[id = "1"]'},
                style={"width": "100%", "height": "390px"},
                elements=new_nodes + new_edges,
                stylesheet=stylesheet,
            )
            return component
        # Handle global feature graphs
        elif self.explanation_type == "global":
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
                figure = figure.format(feature)
                return figure

            new_nodes = self._weight_nodes_feature(nodes, feature)
            elements = new_nodes + data_dict["edges"]
            component = cyto.Cytoscape(
                layout={"name": "breadthfirst", "roots": '[id = "1"]'},
                style={"width": "100%", "height": "390px"},
                elements=elements,
                stylesheet=stylesheet,
            )
            return component
        else:  # pragma: no cover
            msg = "Cannot handle type {0}".format(self.explanation_type)
            log.error(msg)
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


class BaseShallowDecisionTree:
    """ Shallow Decision Tree (low depth).

    Currently wrapper around DecisionTreeClassifier in scikit-learn.
    To keep the tree shallow, max depth is defaulted to 3.

    https://github.com/scikit-learn/scikit-learn

    """

    available_explanations = ["global", "local"]
    explainer_type = "model"

    def __init__(self, max_depth=3, feature_names=None, feature_types=None, **kwargs):
        """ Initializes decision tree with low depth.

        Args:
            **kwargs: Keyword arguments to be passed to DecisionTreeClassifier
                in scikit-learn.
        """
        self.feature_names = feature_names
        self.feature_types = feature_types
        self.max_depth = max_depth
        self.kwargs = kwargs

    @abstractmethod
    def _model(self):
        # This method should be overriden
        return None

    def fit(self, X, y):
        """ Fits model to provided instances.

        Args:
            X: Numpy array for training instances.
            y: Numpy array as training labels.

        Returns:
            Itself.
        """
        X, y, self.feature_names, self.feature_types = unify_data(
            X, y, self.feature_names, self.feature_types
        )
        sk_model_ = self._model()
        sk_model_.fit(X, y)

        feat_imp = sk_model_.feature_importances_
        self.global_selector = gen_global_selector(
            X, self.feature_names, self.feature_types, feat_imp
        )
        self.n_samples_ = X.shape[0]
        return self

    def predict(self, X):
        """ Predicts on provided instances.

        Args:
            X: Numpy array for instances.

        Returns:
            Predicted class label per instance.
        """
        X, _, _, _ = unify_data(X, None, self.feature_names, self.feature_types)
        return self._model().predict(X)

    def explain_global(self, name=None):
        """ Provides global explanation for model.

        Args:
            name: User-defined explanation name.

        Returns:
            An explanation object,
            visualizing feature-value pairs as horizontal bar chart.
        """
        if name is None:
            name = gen_name_from_class(self)

        # Extract decision tree structure
        nodes, edges = self._graph_from_tree(
            self._model(), self.feature_names, max_depth=self.max_depth
        )
        overall_data_dict = {
            "type": "tree",
            "features": self.feature_names,
            "nodes": nodes,
            "edges": edges,
        }
        data_dicts = [
            {
                "type": "tree",
                "features": self.feature_names,
                "nodes": nodes,
                "edges": edges,
            }
            for _ in self.feature_names
        ]

        internal_obj = {"overall": overall_data_dict, "specific": data_dicts}

        return TreeExplanation(
            "global",
            internal_obj,
            feature_names=self.feature_names,
            feature_types=self.feature_types,
            name=name,
            selector=self.global_selector,
        )

    def explain_local(self, X, y=None, name=None):
        if name is None:
            name = gen_name_from_class(self)

        X, y, _, _ = unify_data(X, y, self.feature_names, self.feature_types)

        # Extract decision tree structure
        nodes, edges = self._graph_from_tree(self._model(), self.feature_names)

        decisions = [
            self._model().decision_path(instance.reshape(1, -1)).nonzero()[1] + 1
            for instance in X
        ]
        data_dicts = [
            {
                "type": "tree",
                "features": self.feature_names,
                "nodes": nodes,
                "edges": edges,
                "decision": decision,
            }
            for decision in decisions
        ]

        internal_obj = {"overall": None, "specific": data_dicts}

        if is_classifier(self):
            scores = self.predict_proba(X)[:, 1]
        else:
            scores = self.predict(X)

        selector = gen_local_selector(X, y, scores)

        return TreeExplanation(
            "local",
            internal_obj,
            feature_names=self.feature_names,
            feature_types=self.feature_types,
            name=name,
            selector=selector,
        )

    def _graph_from_tree(self, tree, feature_names=None, max_depth=None):
        """ Adapted from:
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
                value = [list(map(int, l)) for l in tree_.value[i].tolist()]
            else:
                value = [list(map(int, l)) for l in tree_.value[i].tolist()]
                if feature_names is not None:
                    feature = feature_names[feature]

            counter["node"] += 1
            node_id = str(counter["node"])
            if is_classifier(self):
                value_str = "# Obs: "
            else:
                value_str = "E[Y]: "

            if feature is not None and threshold is not None:
                value_str += ", ".join([str(v) for v in value[0]])
                label_str = "{0} <= {1:.2f}\n{2}".format(feature, threshold, value_str)
            else:
                value_str += ", ".join([str(v) for v in value[0]])
                label_str = "Impurity: {0:.2f}\n{1}".format(
                    tree_.impurity[i], value_str
                )

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
        return (nodes, edges)


class RegressionTree(BaseShallowDecisionTree, RegressorMixin, ExplainerMixin):
    def __init__(self, max_depth=3, feature_names=None, feature_types=None, **kwargs):
        super().__init__(
            max_depth=max_depth,
            feature_names=feature_names,
            feature_types=feature_types,
            **kwargs
        )

    def _model(self):
        return self.sk_model_

    def fit(self, X, y):
        self.sk_model_ = SKRT(max_depth=self.max_depth, **self.kwargs)
        return super().fit(X, y)


class ClassificationTree(BaseShallowDecisionTree, ClassifierMixin, ExplainerMixin):
    def __init__(self, max_depth=3, feature_names=None, feature_types=None, **kwargs):
        super().__init__(
            max_depth=max_depth,
            feature_names=feature_names,
            feature_types=feature_types,
            **kwargs
        )

    def _model(self):
        return self.sk_model_

    def fit(self, X, y):
        self.sk_model_ = SKDT(max_depth=self.max_depth, **self.kwargs)
        return super().fit(X, y)

    def predict_proba(self, X):
        X, _, _, _ = unify_data(X, None, self.feature_names, self.feature_types)
        return self._model().predict_proba(X)
