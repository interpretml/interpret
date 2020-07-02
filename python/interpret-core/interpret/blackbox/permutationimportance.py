# # Copyright (c) 2019 Microsoft Corporation
# # Distributed under the MIT software license
#
# import scipy as sp
# import numpy as np
#
# import sklearn.metrics
# from scipy.sparse import SparseEfficiencyWarning
#
# from ..utils import gen_name_from_class, gen_local_selector
#
# from interpret.api.base import ExplainerMixin
# from interpret.api.templates import FeatureValueExplanation
# from interpret.utils import unify_predict_fn, unify_data
# from interpret.utils import gen_name_from_class, gen_global_selector
#
# import warnings
#
#
# def _order_imp(summary):
#     return summary.argsort()[..., ::-1]
#
#
# class ExplainParams:
#     CLASSES = "classes"
#     MODEL_TASK = "model_task"
#     NUM_FEATURES = "num_features"
#     EXPECTED_VALUES = "expected_values"
#     CLASSIFICATION = "classification"
#     GLOBAL_IMPORTANCE_VALUES = "global_importance_values"
#     GLOBAL_IMPORTANCE_RANK = "global_importance_rank"
#     FEATURES = "features"
#     MODEL_TYPE = "model_type"
#
#
# VALID_SKLEARN_METRICS = {'mean_absolute_error',
#                          'explained_variance_score',
#                          'mean_squared_error',
#                          'mean_squared_log_error',
#                          'median_absolute_error',
#                          'r2_score',
#                          'average_precision_score',
#                          'f1_score',
#                          'fbeta_score',
#                          'precision_score',
#                          'recall_score'}
#
#
# class PermutationImportance(ExplainerMixin):
#     available_explanations = ["global"]
#     explainer_type = "blackbox"
#
#     def __init__(
#         self, predict_fn, data, labels, metric=None, sampler=None, feature_names=None, feature_types=None
#     ):
#         self.data, self.labels, self.feature_names, self.feature_types = unify_data(
#             data, labels, feature_names, feature_types
#         )
#         self.predict_fn = unify_predict_fn(predict_fn, data)
#         self.y_hat = self.predict_fn(data)
#         if metric is None:
#             metric_func = sklearn.metrics.mean_squared_error
#         elif isinstance(metric, str):
#             if metric not in VALID_SKLEARN_METRICS:
#                 raise Exception("Unsupported metric name {}, supported metric functions include {}. "
#                                 " Passing in the metric function such as sklearn.metrics.mean_squared_error "
#                                 "is also supported.".format(metric, VALID_SKLEARN_METRICS))
#             else:
#                 metric_func = getattr(sklearn.metrics, metric)
#         elif callable(metric):
#             metric_func = metric
#         else:
#             raise Exception("Unsupported metric input type {}.".format(type(metric)))
#
#         self.metric = metric_func
#         self.sampler = sampler
#
#     def _add_metric(self, predict_function, shuffled_dataset, true_labels,
#                     base_metric, global_importance_values, idx):
#         shuffled_prediction = predict_function(shuffled_dataset)
#         if sp.sparse.issparse(shuffled_prediction):
#             shuffled_prediction = shuffled_prediction.toarray()
#         metric = self.metric(self.labels, shuffled_prediction)
#         importance_score = base_metric - metric
#         global_importance_values[idx] = importance_score
#
#     def _compute_sparse_metric(self, dataset, col_idx, random_indexes, shuffled_dataset,
#                                predict_function, true_labels, base_metric, global_importance_values):
#         # Get non zero column indexes
#         indptr = dataset.indptr
#         indices = dataset.indices
#         col_nz_indices = indices[indptr[col_idx]:indptr[col_idx + 1]]
#         # Sparse optimization: If all zeros, skip the column!  Shuffling won't make a difference to metric.
#         if col_nz_indices.size == 0:
#             return
#         data = dataset.data
#         # Replace non-zero indexes with shuffled indexes
#         col_random_indexes = random_indexes[0:len(col_nz_indices)]
#         with warnings.catch_warnings():
#             warnings.simplefilter('ignore', SparseEfficiencyWarning)
#             # Shuffle the sparse column indexes
#             shuffled_dataset[col_random_indexes, col_idx] = shuffled_dataset[col_nz_indices, col_idx].T
#             # Get set difference and zero-out indexes that had a value but now should be zero
#             difference_nz_random = list(set(col_nz_indices).difference(set(col_random_indexes)))
#             difference_random_nz = list(set(col_random_indexes).difference(set(col_nz_indices)))
#             # Set values that should not be sparse explicitly to zeros
#             shuffled_dataset[difference_nz_random, col_idx] = np.zeros((len(difference_nz_random)),
#                                                                        dtype=data.dtype)
#             idx = col_idx
#             self._add_metric(predict_function, shuffled_dataset, true_labels,
#                              base_metric, global_importance_values, idx)
#             # Restore column back to previous state by undoing shuffle
#             shuffled_dataset[col_nz_indices, col_idx] = shuffled_dataset[col_random_indexes, col_idx].T
#             shuffled_dataset[difference_random_nz, col_idx] = np.zeros((len(difference_random_nz)),
#                                                                        dtype=data.dtype)
#
#     def _compute_dense_metric(self, dataset, col_idx, subset_idx, random_indexes,
#                               predict_function, true_labels, base_metric, global_importance_values):
#         # Create a copy of the original dataset
#         shuffled_dataset = np.array(dataset, copy=True)
#         # Shuffle one of the columns in place
#         shuffled_dataset[:, col_idx] = shuffled_dataset[random_indexes, col_idx]
#         idx = col_idx
#         self._add_metric(predict_function, shuffled_dataset, true_labels,
#                          base_metric, global_importance_values, idx)
#
#     def explain_global(self, name=None):
#         if hasattr(self, "_global_explanation"):
#             return self._global_explanation
#
#         true_labels = self.labels
#         mli_dict = {ExplainParams.MODEL_TYPE: "pfi"}
#
#         columns = getattr(self.y_hat, "columns", None)
#         mli_dict[ExplainParams.CLASSES] = [col for col in columns] if columns is not None else None
#         mli_dict[ExplainParams.MODEL_TASK] = "classification"
#
#         dataset = self.data
#
#         mli_dict[ExplainParams.NUM_FEATURES] = len(dataset[0])
#
#         predict_function = self.predict_fn
#         # Score the model on the given dataset
#         predictions = predict_function(dataset)
#         # The scikit-learn metrics can't handle sparse arrays
#         if sp.sparse.issparse(true_labels):
#             true_labels = true_labels.toarray()
#         if sp.sparse.issparse(predictions):
#             predictions = predictions.toarray()
#         # Evaluate the model with given metric on the dataset
#         base_metric = self.metric(true_labels, predictions)
#         column_indexes = range(dataset.shape[1])
#         global_importance_values = np.zeros(dataset.shape[1])
#         if sp.sparse.issparse(dataset):
#             # Create a dataset for shuffling
#             # Although lil matrix is better for changing sparsity structure, scikit-learn
#             # converts matrixes back to csr for predictions which is much more expensive
#             shuffled_dataset = dataset.tocsr(copy=True)
#             # Convert to csc format if not already for faster column index access
#             if not sp.sparse.isspmatrix_csc(dataset):
#                 dataset = dataset.tocsc()
#             # Get max NNZ across all columns
#             dataset_nnz = dataset.getnnz(axis=0)
#             maxnnz = max(dataset_nnz)
#             column_indexes = np.unique(np.intersect1d(dataset.nonzero()[1], column_indexes))
#             # Choose random, shuffled n of k indexes
#             random_indexes = np.random.choice(dataset.shape[0], maxnnz, replace=False)
#             # Shuffle all sparse columns
#             for subset_idx, col_idx in enumerate(column_indexes):
#                 self._compute_sparse_metric(dataset, col_idx, subset_idx, random_indexes, shuffled_dataset,
#                                             predict_function, true_labels, base_metric, global_importance_values)
#         else:
#             num_rows = dataset.shape[0]
#             random_indexes = np.random.choice(num_rows, num_rows, replace=False)
#             for subset_idx, col_idx in enumerate(column_indexes):
#                 self._compute_dense_metric(dataset, col_idx, subset_idx, random_indexes, predict_function,
#                                            true_labels, base_metric, global_importance_values)
#         order = _order_imp(global_importance_values)
#         mli_dict[ExplainParams.EXPECTED_VALUES] = None
#         mli_dict[ExplainParams.CLASSIFICATION] = True
#         mli_dict[ExplainParams.GLOBAL_IMPORTANCE_VALUES] = global_importance_values
#         mli_dict[ExplainParams.GLOBAL_IMPORTANCE_RANK] = order
#         mli_dict[ExplainParams.FEATURES] = self.feature_names
#         overall_dict = {
#             "names": self.feature_names,
#             "scores": global_importance_values
#         }
#
#         internal_obj = {
#             "overall": overall_dict,
#             "specific": None,
#             "mli": mli_dict
#         }
#
#         global_selector = gen_local_selector(self.labels, predictions)
#         name = gen_name_from_class(self) if name is None else name
#         global_explanation = FeatureValueExplanation(
#             "global",
#             internal_obj,
#             feature_names=self.feature_names,
#             feature_types=self.feature_types,
#             name=name,
#             selector=global_selector,
#         )
#         self._global_explanation = global_explanation  # Not threadsafe
#         return global_explanation
#
#     def visualize(self, **kwargs):
#         from interpret.glassbox.linear import LinearExplanation
#         # TODO Can the base vis be a util?
#         return LinearExplanation.visualize(self, **kwargs)
