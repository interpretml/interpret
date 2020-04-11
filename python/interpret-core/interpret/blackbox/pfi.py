# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

import scipy as sp
import numpy as np

from ..utils import gen_name_from_class, gen_local_selector

from interpret.api.base import ExplainerMixin
from interpret.api.templates import FeatureValueExplanation
from interpret.utils import unify_predict_fn, unify_data
from interpret.utils import gen_name_from_class, gen_global_selector
from interpret.blackbox.sensitivity import SamplerMixin  # Where should this live?
from sklearn.metrics import f1_score

from abc import ABC, abstractmethod


def _order_imp(summary):
    """Compute the ranking of feature importance values.

    :param summary: A 3D array of the feature importance values to be ranked.
    :type summary: numpy.ndarray
    :return: The rank of the feature importance values.
    :rtype: numpy.ndarray
    """
    return summary.argsort()[..., ::-1]


class ExplainParams:
    CLASSES = "classes"
    MODEL_TASK = "model_task"
    NUM_FEATURES = "num_features"
    EXPECTED_VALUES = "expected_values"
    CLASSIFICATION = "classification"
    GLOBAL_IMPORTANCE_VALUES = "global_importance_values"
    GLOBAL_IMPORTANCE_RANK = "global_importance_rank"
    FEATURES = "features"
    MODEL_TYPE = "model_type"


class SubsetSampler(SamplerMixin):
    def __init__(self, indices=None):
        self.indices = indicies

    def sample(self, data):
        return data[self.indices]


class Metric:
    def __init__(self, name, **kwargs):
        self.name = name
        self.kwargs = {"average": "micro"} if not kwargs else kwargs

    def get(self, *args):
        return f1_score(*args, **self.kwargs)

class PermutationImportanceClassification(ExplainerMixin):
    available_explanations = ["global"]
    explainer_type = "blackbox"

    def __init__(

        self, predict_fn, metric, sampler=None, feature_names=None, feature_types=None
    ):
        self._predict_fn_not_unified = predict_fn
        self.predict_fn = None
        self.feature_names = feature_names
        self.feature_types = feature_types
        self.metric = Metric(metric) if isinstance(metric, str) else metric
        self.sampler = sampler

    def _add_metric(self, predict_function, shuffled_dataset, true_labels,
                    base_metric, global_importance_values, idx):
        """Compute and add the metric to the global importance values array.

        :param predict_function: The prediction function.
        :type predict_function: function
        :param shuffled_dataset: The shuffled dataset to predict on.
        :type shuffled_dataset: scipy.csr or numpy.ndarray
        :param true_labels: The true labels.
        :type true_labels: numpy.ndarray
        :param base_metric: Base metric for unshuffled dataset.
        :type base_metric: float
        :param global_importance_values: Pre-allocated array of global importance values.
        :type global_importance_values: numpy.ndarray
        """
        shuffled_prediction = predict_function(shuffled_dataset)
        if sp.sparse.issparse(shuffled_prediction):
            shuffled_prediction = shuffled_prediction.toarray()
        metric = self.metric.get(true_labels, shuffled_prediction)
        importance_score = base_metric - metric
        global_importance_values[idx] = importance_score

    def _compute_sparse_metric(self, dataset, col_idx, random_indexes, shuffled_dataset,
                               predict_function, true_labels, base_metric, global_importance_values):
        """Shuffle a sparse dataset column and compute the feature importance metric.

        :param dataset: Dataset used as a reference point for getting column indexes per row.
        :type dataset: scipy.csc
        :param col_idx: The column index.
        :type col_idx: int
        :param random_indexes: Generated random indexes.
        :type random_indexes: numpy.ndarray
        :param shuffled_dataset: The dataset to shuffle.
        :type shuffled_dataset: scipy.csr
        :param predict_function: The prediction function.
        :type predict_function: function
        :param true_labels: The true labels.
        :type true_labels: numpy.ndarray
        :param base_metric: Base metric for unshuffled dataset.
        :type base_metric: float
        :param global_importance_values: Pre-allocated array of global importance values.
        :type global_importance_values: numpy.ndarray
        """
        # Get non zero column indexes
        indptr = dataset.indptr
        indices = dataset.indices
        col_nz_indices = indices[indptr[col_idx]:indptr[col_idx + 1]]
        # Sparse optimization: If all zeros, skip the column!  Shuffling won't make a difference to metric.
        if col_nz_indices.size == 0:
            return
        data = dataset.data
        # Replace non-zero indexes with shuffled indexes
        col_random_indexes = random_indexes[0:len(col_nz_indices)]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', SparseEfficiencyWarning)
            # Shuffle the sparse column indexes
            shuffled_dataset[col_random_indexes, col_idx] = shuffled_dataset[col_nz_indices, col_idx].T
            # Get set difference and zero-out indexes that had a value but now should be zero
            difference_nz_random = list(set(col_nz_indices).difference(set(col_random_indexes)))
            difference_random_nz = list(set(col_random_indexes).difference(set(col_nz_indices)))
            # Set values that should not be sparse explicitly to zeros
            shuffled_dataset[difference_nz_random, col_idx] = np.zeros((len(difference_nz_random)),
                                                                       dtype=data.dtype)
            idx = col_idx
            self._add_metric(predict_function, shuffled_dataset, true_labels,
                             base_metric, global_importance_values, idx)
            # Restore column back to previous state by undoing shuffle
            shuffled_dataset[col_nz_indices, col_idx] = shuffled_dataset[col_random_indexes, col_idx].T
            shuffled_dataset[difference_random_nz, col_idx] = np.zeros((len(difference_random_nz)),
                                                                       dtype=data.dtype)

    def _compute_dense_metric(self, dataset, col_idx, subset_idx, random_indexes,
                              predict_function, true_labels, base_metric, global_importance_values):
        """Shuffle a dense dataset column and compute the feature importance metric.

        :param dataset: Dataset used as a reference point for getting column indexes per row.
        :type dataset: numpy.ndarray
        :param col_idx: The column index.
        :type col_idx: int
        :param subset_idx: The subset index.
        :type subset_idx: int
        :param random_indexes: Generated random indexes.
        :type random_indexes: numpy.ndarray
        :param predict_function: The prediction function.
        :type predict_function: function
        :param true_labels: The true labels.
        :type true_labels: numpy.ndarray
        :param base_metric: Base metric for unshuffled dataset.
        :type base_metric: float
        :param global_importance_values: Pre-allocated array of global importance values.
        :type global_importance_values: numpy.ndarray
        """
        # Create a copy of the original dataset
        shuffled_dataset = np.array(dataset, copy=True)
        # Shuffle one of the columns in place
        shuffled_dataset[:, col_idx] = shuffled_dataset[random_indexes, col_idx]
        idx = col_idx
        self._add_metric(predict_function, shuffled_dataset, true_labels,
                         base_metric, global_importance_values, idx)

    def explain_global(self, X, y, name=None):

        data, labels, _, _ = unify_data(
            X, y, self.feature_names, self.feature_types
        )
        predict_fn = unify_predict_fn(self._predict_fn_not_unified, X)

        evaluation_examples = data
        true_labels = labels
        mli_dict = {ExplainParams.MODEL_TYPE: "pfi"}

        columns = getattr(y, "columns", None)
        mli_dict[ExplainParams.CLASSES] = [col for col in columns] if columns is not None else None
        mli_dict[ExplainParams.MODEL_TASK] = "classification"

        dataset = data

        mli_dict[ExplainParams.NUM_FEATURES] = len(data[0])


        predict_function = predict_fn
        # Score the model on the given dataset
        predictions = predict_function(dataset)
        # The scikit-learn metrics can't handle sparse arrays
        if sp.sparse.issparse(true_labels):
            true_labels = true_labels.toarray()
        if sp.sparse.issparse(predictions):
            predictions = prediction.toarray()
        # Evaluate the model with given metric on the dataset
        base_metric = self.metric.get(true_labels, predictions)
        column_indexes = range(dataset.shape[1])
        global_importance_values = np.zeros(dataset.shape[1])
        if sp.sparse.issparse(dataset):
            # Create a dataset for shuffling
            # Although lil matrix is better for changing sparsity structure, scikit-learn
            # converts matrixes back to csr for predictions which is much more expensive
            shuffled_dataset = dataset.tocsr(copy=True)
            # Convert to csc format if not already for faster column index access
            if not sp.sparse.isspmatrix_csc(dataset):
                dataset = dataset.tocsc()
            # Get max NNZ across all columns
            dataset_nnz = dataset.getnnz(axis=0)
            maxnnz = max(dataset_nnz)
            column_indexes = np.unique(np.intersect1d(dataset.nonzero()[1], column_indexes))
            # Choose random, shuffled n of k indexes
            random_indexes = np.random.choice(dataset.shape[0], maxnnz, replace=False)
            # Shuffle all sparse columns
            for subset_idx, col_idx in enumerate(column_indexes):
                self._compute_sparse_metric(dataset, col_idx, subset_idx, random_indexes, shuffled_dataset,
                                            predict_function, true_labels, base_metric, global_importance_values)
        else:
            num_rows = dataset.shape[0]
            random_indexes = np.random.choice(num_rows, num_rows, replace=False)
            for subset_idx, col_idx in enumerate(column_indexes):
                self._compute_dense_metric(dataset, col_idx, subset_idx, random_indexes, predict_function,
                                           true_labels, base_metric, global_importance_values)
        order = _order_imp(global_importance_values)
        mli_dict[ExplainParams.EXPECTED_VALUES] = None
        mli_dict[ExplainParams.CLASSIFICATION] = True
        mli_dict[ExplainParams.GLOBAL_IMPORTANCE_VALUES] = global_importance_values
        mli_dict[ExplainParams.GLOBAL_IMPORTANCE_RANK] = order
        mli_dict[ExplainParams.FEATURES] = self.feature_names

        internal_obj = {
            "overall": None,
            "specific": None,
            "mli": mli_dict
        }

        ys = [val[0] for val in y.values]
        global_selector = gen_local_selector(ys, predictions)
        name = gen_name_from_class(self) if name is None else name
        return FeatureValueExplanation(
            "global",
            internal_obj,
            feature_names=self.feature_names,
            feature_types=self.feature_types,
            name=name,
            selector=global_selector,
        )
