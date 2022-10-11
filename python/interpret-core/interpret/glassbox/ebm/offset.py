import interpret
from interpret.glassbox.ebm.internal import Native
from typing import DefaultDict
from interpret.provider.visualize import PreserveProvider
from interpret.utils import gen_perf_dicts
from interpret.glassbox.ebm.utils import DPUtils #, EBMUtils
from interpret.glassbox.ebm.postprocessing import multiclass_postprocess
from interpret.utils import unify_data, autogen_schema, unify_vector
from interpret.api.base import ExplainerMixin
from interpret.api.templates import FeatureValueExplanation
from interpret.provider.compute import JobLibProvider
from interpret.utils import gen_name_from_class, gen_global_selector, gen_local_selector
import ctypes as ct
from multiprocessing.sharedctypes import RawArray
import numpy as np
from warnings import warn
from sklearn.base import is_classifier
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import log_loss, mean_squared_error
import heapq
from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
    ClassifierMixin,
    RegressorMixin,
)
from itertools import combinations

# Imports required in interpret.utils
from math import ceil
from interpret.glassbox.ebm.internal import Native, Booster, InteractionDetector
# from scipy.special import expit
from sklearn.utils.extmath import softmax
from sklearn.model_selection import train_test_split
from sklearn.base import is_classifier
import numbers
import numpy as np
import warnings
import copy
from scipy.stats import norm
from scipy.optimize import root_scalar, brentq
import logging
log = logging.getLogger(__name__)

from interpret.glassbox.ebm.ebm import DPExplainableBoostingClassifier, DPExplainableBoostingRegressor, DPUtils, EBMPreprocessor, EBMExplanation #, ExplainableBoostingClassifier


# Add offset to EBMUtils
class OffsetEBMUtils(interpret.glassbox.ebm.utils.EBMUtils):

    @staticmethod
    def ebm_train_test_split(
            X, Off, y, w, test_size, random_state, is_classification, is_train=True,
    ):

        if (X.shape[0] != len(y) or X.shape[0] != len(w)):
            raise Exception("Data, labels and weights should have the same number of rows.")

        sampling_result = None

        if test_size == 0:
            X_train, y_train, w_train, Off_train, = X, y, w, Off
            X_val = np.empty(shape=(0, X.shape[1]), dtype=X.dtype)
            y_val = np.empty(shape=(0,), dtype=y.dtype)
            w_val = np.empty(shape=(0,), dtype=w.dtype)
            Off_val = np.empty(shape=(0,), dtype=Off.dtype)

        elif test_size > 0:
            n_samples = X.shape[0]
            n_test_samples = 0

            if test_size >= 1:
                if test_size % 1:
                    raise Exception("If test_size >= 1, test_size should be a whole number.")
                n_test_samples = test_size
            else:
                n_test_samples = ceil(n_samples * test_size)

            n_train_samples = n_samples - n_test_samples
            native = Native.get_native_singleton()

            # Adapt test size if too small relative to number of classes
            if is_classification:
                y_uniq = len(set(y))
                if n_test_samples < y_uniq:  # pragma: no cover
                    warnings.warn(
                        "Too few samples per class, adapting test size to guarantee 1 sample per class."
                    )
                    n_test_samples = y_uniq
                    n_train_samples = n_samples - n_test_samples

                sampling_result = native.stratified_sampling_without_replacement(
                    random_state,
                    y_uniq,
                    n_train_samples,
                    n_test_samples,
                    y
                )
            else:
                sampling_result = native.sample_without_replacement(
                    random_state,
                    n_train_samples,
                    n_test_samples
                )

        else:  # pragma: no cover
            raise Exception("test_size must be a positive numeric value.")

        if sampling_result is not None:
            train_indices = np.where(sampling_result == 1)
            test_indices = np.where(sampling_result == -1)
            X_train = X[train_indices]
            X_val = X[test_indices]
            y_train = y[train_indices]
            y_val = y[test_indices]
            w_train = w[train_indices]
            w_val = w[test_indices]
            Off_train = Off[train_indices]
            Off_val = Off[test_indices]

        if not is_train:
            X_train, y_train, Off_train = None, None, None

        # TODO PK doing a fortran re-ordering here (and an extra copy) isn't the most efficient way
        #         push the re-ordering right to our first call to fit(..) AND stripe convert
        #         groups of rows at once and they process them in fortran order after that
        # change to Fortran ordering on our data, which is more efficient in terms of memory accesses
        # AND our C code expects it in that ordering
        if X_train is not None:
            X_train = np.ascontiguousarray(X_train.T)

        X_val = np.ascontiguousarray(X_val.T)

        return X_train, X_val, Off_train, Off_val, y_train, y_val, w_train, w_val

    @staticmethod
    def scores_by_feature_group(X, X_pair, feature_groups, model):

        for set_idx, feature_group in enumerate(feature_groups):
            tensor = model[set_idx]

            # Get the current column(s) to process
            feature_idxs = feature_group

            if X_pair is not None:
                sliced_X = X[feature_idxs, :] if len(feature_group) == 1 else X_pair[feature_idxs, :]
            else:
                sliced_X = X[feature_idxs, :]

            scores = tensor[tuple(sliced_X)]

            # Reset scores from unknown (not missing!) indexes to 0
            # this assumes all logits are zero weighted centered, and ideally tensors are purified
            unknowns = (sliced_X < 0).any(axis=0)
            scores[unknowns] = 0

            yield set_idx, feature_group, scores

    @staticmethod
    def decision_function(X, X_pair, Off, feature_groups, model, intercept):

        if X.ndim == 1:
            X = X.reshape(X.shape[0], 1)

        # Initialize empty vector for predictions
        if isinstance(intercept, numbers.Number) or len(intercept) == 1:
            score_vector = np.empty(X.shape[1])
        else:
            score_vector = np.empty((X.shape[1], len(intercept)))

        np.copyto(score_vector, intercept)

        # Adding offset to score_vector
        if Off is not None:
            score_vector += Off

        # Generate prediction scores
        scores_gen = EBMUtils.scores_by_feature_group(
            X, X_pair, feature_groups, model
        )

        for _, _, scores in scores_gen:
            score_vector += scores

        if not np.all(np.isfinite(score_vector)):  # pragma: no cover
            msg = "Non-finite values present in log odds vector."
            log.error(msg)
            raise Exception(msg)

        return score_vector


    @staticmethod
    def decision_function_and_explain(X, X_pair, Off, feature_groups, model, intercept):

        if X.ndim == 1:
            X = X.reshape(X.shape[0], 1)

        # Initialize empty vector for predictions and explanations
        if isinstance(intercept, numbers.Number) or len(intercept) == 1:
            score_vector = np.empty(X.shape[1])
        else:
            score_vector = np.empty((X.shape[1], len(intercept)))

        np.copyto(score_vector, intercept)

        # Add offset to score_vector
        if Off is not None:
            score_vector += np.array(Off)

        n_interactions = sum(len(fg) > 1 for fg in feature_groups)
        explanations = np.empty((X.shape[1], X.shape[0] + n_interactions))

        # Generate prediction scores
        scores_gen = EBMUtils.scores_by_feature_group(
            X, X_pair, feature_groups, model
        )
        for set_idx, _, scores in scores_gen:
            score_vector += scores
            explanations[:, set_idx] = scores

        if not np.all(np.isfinite(score_vector)):  # pragma: no cover
            msg = "Non-finite values present in log odds vector."
            log.error(msg)
            raise Exception(msg)

        return score_vector, explanations

    @staticmethod
    def classifier_predict_proba(X, X_pair, Off, feature_groups, model, intercept):

        log_odds_vector = EBMUtils.decision_function(
            X, X_pair, Off, feature_groups, model, intercept
        )

        # Handle binary classification case -- softmax only works with 0s appended
        if log_odds_vector.ndim == 1:
            log_odds_vector = np.c_[np.zeros(log_odds_vector.shape), log_odds_vector]

        return softmax(log_odds_vector)

    @staticmethod
    def classifier_predict(X, X_pair, Off, feature_groups, model, intercept, classes):

        log_odds_vector = EBMUtils.decision_function(
            X, X_pair, Off, feature_groups, model, intercept
        )
        if log_odds_vector.ndim == 1:
            log_odds_vector = np.c_[np.zeros(log_odds_vector.shape), log_odds_vector]

        return classes[np.argmax(log_odds_vector, axis=1)]

    @staticmethod
    def classifier_predict_and_contrib(X, X_pair, Off, feature_groups, model, intercept, classes, output='probabilities'):

        scores_vector, explanations = EBMUtils.decision_function_and_explain(
            X,
            X_pair,
            Off,
            feature_groups,
            model,
            intercept
        )

        if output == 'probabilities':
            if scores_vector.ndim == 1:
                scores_vector = np.c_[np.zeros(scores_vector.shape), scores_vector]
            return softmax(scores_vector), explanations
        elif output == 'labels':
            if scores_vector.ndim == 1:
                scores_vector = np.c_[np.zeros(scores_vector.shape), scores_vector]
            return classes[np.argmax(scores_vector, axis=1)], explanations
        else:
            return scores_vector, explanations

    @staticmethod
    def regressor_predict(X, X_pair, Off, feature_groups, model, intercept):
        scores = EBMUtils.decision_function(X, X_pair, Off, feature_groups, model, intercept)
        return scores

    @staticmethod
    def regressor_predict_and_contrib(X, X_pair, Off, feature_groups, model, intercept):

        scores, explanations = EBMUtils.decision_function_and_explain(
            X,
            X_pair,
            Off,
            feature_groups,
            model,
            intercept
        )
        return scores, explanations

    # NOT REQ?
    @staticmethod
    def gen_feature_group_name(feature_idxs, col_names):
        feature_group_name = []
        for feature_index in feature_idxs:
            col_name = col_names[feature_index]
            feature_group_name.append(
                f'feature_{col_name:04}'
                if isinstance(col_name, int)
                else str(col_name)
            )
        feature_group_name = " x ".join(feature_group_name)
        return feature_group_name

    # NOT REQ?
    @staticmethod
    def gen_feature_group_type(feature_idxs, col_types):
        if len(feature_idxs) == 1:
            return col_types[feature_idxs[0]]
        else:
            # TODO PK we should consider changing the feature type to the same " x " separator
            # style as gen_feature_name, for human understanability
            return "interaction"

    @staticmethod
    def cyclic_gradient_boost(
            model_type,
            n_classes,
            features_categorical,
            features_bin_count,
            feature_groups,
            X_train,
            y_train,
            w_train,
            scores_train,
            X_val,
            y_val,
            w_val,
            scores_val,
            n_inner_bags,
            generate_update_options,
            learning_rate,
            min_samples_leaf,
            max_leaves,
            early_stopping_rounds,
            early_stopping_tolerance,
            max_rounds,
            random_state,
            name,
            noise_scale,
            bin_counts,
            optional_temp_params=None,
    ):
        min_metric = np.inf
        episode_index = 0
        with Booster(
                model_type,
                n_classes,
                features_categorical,
                features_bin_count,
                feature_groups,
                X_train,
                y_train,
                w_train,
                scores_train,
                X_val,
                y_val,
                w_val,
                scores_val,
                n_inner_bags,
                random_state,
                optional_temp_params,
        ) as booster:
            no_change_run_length = 0
            bp_metric = np.inf
            log.info("Start boosting {0}".format(name))
            for episode_index in range(max_rounds):
                if episode_index % 10 == 0:
                    log.debug("Sweep Index for {0}: {1}".format(name, episode_index))
                    log.debug("Metric: {0}".format(min_metric))

                for feature_group_index in range(len(feature_groups)):
                    gain = booster.generate_model_update(
                        feature_group_index=feature_group_index,
                        generate_update_options=generate_update_options,
                        learning_rate=learning_rate,
                        min_samples_leaf=min_samples_leaf,
                        max_leaves=max_leaves,
                    )

                    if noise_scale:  # Differentially private updates
                        splits = booster.get_model_update_splits()[0]

                        model_update_tensor = booster.get_model_update_expanded()
                        noisy_update_tensor = model_update_tensor.copy()

                        splits_iter = [0] + list(splits + 1) + [
                            len(model_update_tensor)]  # Make splits iteration friendly
                        # Loop through all random splits and add noise before updating
                        for f, s in zip(splits_iter[:-1], splits_iter[1:]):
                            if s == 1:
                                continue  # Skip cuts that fall on 0th (missing value) bin -- missing values not supported in DP

                            noise = np.random.normal(0.0, noise_scale)
                            noisy_update_tensor[f:s] = model_update_tensor[f:s] + noise

                            # Native code will be returning sums of residuals in slices, not averages.
                            # Compute noisy average by dividing noisy sum by noisy histogram counts
                            instance_count = np.sum(bin_counts[feature_group_index][f:s])
                            noisy_update_tensor[f:s] = noisy_update_tensor[f:s] / instance_count

                        noisy_update_tensor = noisy_update_tensor * -1  # Invert gradients before updates
                        booster.set_model_update_expanded(feature_group_index, noisy_update_tensor)

                    curr_metric = booster.apply_model_update()

                    min_metric = min(curr_metric, min_metric)

                # TODO PK this early_stopping_tolerance is a little inconsistent
                #      since it triggers intermittently and only re-triggers if the
                #      threshold is re-passed, but not based on a smooth windowed set
                #      of checks.  We can do better by keeping a list of the last
                #      number of measurements to have a consistent window of values.
                #      If we only cared about the metric at the start and end of the epoch
                #      window a circular buffer would be best choice with O(1).
                if no_change_run_length == 0:
                    bp_metric = min_metric
                if min_metric + early_stopping_tolerance < bp_metric:
                    no_change_run_length = 0
                else:
                    no_change_run_length += 1

                if (
                        early_stopping_rounds >= 0
                        and no_change_run_length >= early_stopping_rounds
                ):
                    break

            log.info(
                "End boosting {0}, Best Metric: {1}, Num Rounds: {2}".format(
                    name, min_metric, episode_index
                )
            )

            # TODO: Add more ways to call alternative get_current_model
            # Use latest model if there are no instances in the (transposed) validation set
            # or if training with privacy
            if X_val.shape[1] == 0 or noise_scale is not None:
                model_update = booster.get_current_model()
            else:
                model_update = booster.get_best_model()

        return model_update, min_metric, episode_index

    # IMPORTANT!! NOT REQ? Not sure about this one... scores = Off maybe?
    @staticmethod
    def get_interactions(
            n_interactions,
            iter_feature_groups,
            model_type,
            n_classes,
            features_categorical,
            features_bin_count,
            X,
            y,
            w,
            scores,
            min_samples_leaf,
            optional_temp_params=None,
    ):
        interaction_scores = []
        with InteractionDetector(
                model_type, n_classes, features_categorical, features_bin_count, X, y, w, scores, optional_temp_params
        ) as interaction_detector:
            for feature_group in iter_feature_groups:
                score = interaction_detector.get_interaction_score(
                    feature_group, min_samples_leaf,
                )
                interaction_scores.append((feature_group, score))

        ranked_scores = list(
            sorted(interaction_scores, key=lambda x: x[1], reverse=True)
        )
        final_ranked_scores = ranked_scores

        final_indices = [x[0] for x in final_ranked_scores]
        final_scores = [x[1] for x in final_ranked_scores]

        return final_indices, final_scores


# Add offset to BaseCoreEBM. REQUIRES ATTENTION!!!
class OffsetBaseCoreEBM(interpret.glassbox.ebm.ebm.BaseCoreEBM):

    def fit_parallel(self, X, Off, y, w, X_pair, n_classes):

        self.n_classes_ = n_classes

        # Split data into train/val
        X_train, X_val, Off_train, Off_val, y_train, y_val, w_train, w_val = EBMUtils.ebm_train_test_split(
            X,
            Off,
            y,
            w,
            test_size=self.validation_size,
            random_state=self.random_state,
            is_classification=self.model_type == "classification",
            is_train=True
        )

        if X_pair is not None:
            X_pair_train, X_pair_val, Off_train, Off_val, y_train, y_val, w_train, w_val = EBMUtils.ebm_train_test_split(
                X_pair,
                Off,
                y,
                w,
                test_size=self.validation_size,
                random_state=self.random_state,
                is_classification=self.model_type == "classification",
            )
        else:
            X_pair_train, X_pair_val = None, None

        # Build EBM allocation code

        # scikit-learn returns an np.array for classification and
        # a single np.float64 for regression, so we do the same
        if self.model_type == "classification":
            self.intercept_ = np.zeros(
                Native.get_count_scores_c(self.n_classes_),
                dtype=np.float64,
                order="C",
            )
        else:
            self.intercept_ = np.float64(0)

        if isinstance(self.main_features, str) and self.main_features == "all":
            main_feature_indices = [[x] for x in range(X.shape[1])]
        elif isinstance(self.main_features, list) and all(
                isinstance(x, int) for x in self.main_features
        ):
            main_feature_indices = [[x] for x in self.main_features]
        else:  # pragma: no cover
            raise RuntimeError("Argument 'mains' has invalid value")

        self.feature_groups_ = []
        self.model_ = []

        # Train main effects
        self._fit_main(main_feature_indices, X_train, y_train, w_train, X_val, y_val, w_val, Off_train, Off_val)
        # Build interaction terms, if required
        self.inter_indices_, self.inter_scores_ = self._build_interactions(
            X_train, Off_train, y_train, w_train, X_pair_train
        )

        self.inter_episode_idx_ = 0
        return self

    def _fit_main(self, main_feature_groups, X_train, y_train, w_train, X_val, y_val, w_val, Off_train, Off_val):

        if self.noise_scale is not None:  # Differentially Private Training
            update = Native.GenerateUpdateOptions_GradientSums | Native.GenerateUpdateOptions_RandomSplits
        else:
            update = Native.GenerateUpdateOptions_Default
        log.info("Train main effects")
        (
            self.model_,
            self.current_metric_,
            self.main_episode_idx_,
        ) = EBMUtils.cyclic_gradient_boost(
            model_type=self.model_type,
            n_classes=self.n_classes_,
            features_categorical=self.features_categorical,
            features_bin_count=self.features_bin_count,
            feature_groups=main_feature_groups,
            X_train=X_train,
            y_train=y_train,
            w_train=w_train,
            scores_train=Off_train,
            X_val=X_val,
            y_val=y_val,
            w_val=w_val,
            scores_val=Off_val,
            n_inner_bags=self.inner_bags,
            generate_update_options=update,
            learning_rate=self.learning_rate,
            min_samples_leaf=self.min_samples_leaf,
            max_leaves=self.max_leaves,
            early_stopping_rounds=self.early_stopping_rounds,
            early_stopping_tolerance=self.early_stopping_tolerance,
            max_rounds=self.max_rounds,
            random_state=self.random_state,
            name="Main",
            noise_scale=self.noise_scale,
            bin_counts=self.bin_counts,
        )

        self.feature_groups_ = main_feature_groups

        return

    def _build_interactions(self, X_train, Off_train, y_train, w_train, X_pair):

        if isinstance(self.interactions, int) and self.interactions != 0:
            log.info("Estimating with FAST")

            scores_train = EBMUtils.decision_function(
                X_train, X_pair, Off_train, self.feature_groups_, self.model_, self.intercept_
            )

            iter_feature_groups = combinations(range(X_pair.shape[0]), 2)
            final_indices, final_scores = EBMUtils.get_interactions(
                n_interactions=self.interactions,
                iter_feature_groups=iter_feature_groups,
                model_type=self.model_type,
                n_classes=self.n_classes_,
                features_categorical = self.pair_features_categorical,
                features_bin_count = self.pair_features_bin_count,
                X=X_pair,
                y=y_train,
                w=w_train,
                scores=scores_train, # SO: maybe Off_train? I don't think so ..
                min_samples_leaf=self.min_samples_leaf,
            )
        elif isinstance(self.interactions, int) and self.interactions == 0:
            final_indices = []
            final_scores = []
        elif isinstance(self.interactions, list):
            final_indices = self.interactions
            final_scores = [None for _ in range(len(self.interactions))]
        else:  # pragma: no cover
            raise RuntimeError("Argument 'interaction' has invalid value")
        return final_indices, final_scores

    def _staged_fit_interactions(
            self, X_train, Off_train, y_train, w_train, X_val, Off_val, y_val, w_val, X_pair_train, X_pair_val, inter_indices=[]
    ):

        log.info("Training interactions")

        scores_train = EBMUtils.decision_function(
            X_train, X_pair_train, Off_train, self.feature_groups_, self.model_, self.intercept_
        )
        scores_val = EBMUtils.decision_function(
            X_val, X_pair_val, Off_val, self.feature_groups_, self.model_, self.intercept_
        )

        (
            model_update,
            self.current_metric_,
            self.inter_episode_idx_,
        ) = EBMUtils.cyclic_gradient_boost(
            model_type=self.model_type,
            n_classes=self.n_classes_,
            features_categorical=self.pair_features_categorical,
            features_bin_count=self.pair_features_bin_count,
            feature_groups=inter_indices,
            X_train=X_pair_train,
            y_train=y_train,
            w_train=w_train,
            scores_train=scores_train,
            X_val=X_pair_val,
            y_val=y_val,
            w_val=w_val,
            scores_val=scores_val,
            n_inner_bags=self.inner_bags,
            generate_update_options=Native.GenerateUpdateOptions_Default,
            learning_rate=self.learning_rate,
            min_samples_leaf=self.min_samples_leaf,
            max_leaves=self.max_leaves,
            early_stopping_rounds=self.early_stopping_rounds,
            early_stopping_tolerance=self.early_stopping_tolerance,
            max_rounds=self.max_rounds,
            random_state=self.random_state,
            name="Pair",
            noise_scale=self.noise_scale,
            bin_counts=self.bin_counts,
        )

        self.model_.extend(model_update)
        self.feature_groups_.extend(inter_indices)

        return

    def staged_fit_interactions_parallel(self, X, Off, y, w, X_pair, inter_indices=[]):

        log.info("Splitting train/test for interactions")

        # Split data into train/val
        # NOTE: ideally we would store the train/validation split in the
        #       remote processes, but joblib doesn't have a concept
        #       of keeping remote state, so we re-split our sets
        X_train, X_val, Off_train, Off_val, y_train, y_val, w_train, w_val = EBMUtils.ebm_train_test_split(
            X,
            Off,
            y,
            w,
            test_size=self.validation_size,
            random_state=self.random_state,
            is_classification=self.model_type == "classification",
        )

        # Check this out. NOT sure about offset and interaction terms
        X_pair_train, X_pair_val, _, _, y_train, y_val, w_train, w_val = EBMUtils.ebm_train_test_split(
            X_pair,
            Off,
            y,
            w,
            test_size=self.validation_size,
            random_state=self.random_state,
            is_classification=self.model_type == "classification",
        )

        self._staged_fit_interactions(X_train, Off_train, y_train, w_train, X_val, Off_val, y_val, w_val, X_pair_train, X_pair_val, inter_indices)
        return self


# Add offset to BaseEBM.
class OffsetBaseEBM(interpret.glassbox.ebm.ebm.BaseEBM):

    def fit(self, X, Off, y, sample_weight=None):  # noqa: C901
        """ Fits model to provided samples.

        Args:
            X: Numpy array for training samples.
            y: Numpy array as training labels.
            sample_weight: Optional array of weights per sample. Should be same length as X and y.

        Returns:
            Itself.
        """

        # NOTE: Generally, we want to keep parameters in the __init__ function, since scikit-learn
        #       doesn't like parameters in the fit function, other than ones like weights that have
        #       the same length as the number of samples.  See:
        #       https://scikit-learn.org/stable/developers/develop.html
        #       https://github.com/microsoft/LightGBM/issues/2628#issue-536116395
        #

        # TODO PK sanity check all our inputs from the __init__ function, and this fit fuction

        # TODO PK we shouldn't expose our internal state until we are 100% sure that we succeeded
        #         so move everything to local variables until the end when we assign them to self.*

        # TODO PK we should do some basic checks here that X and y have the same dimensions and that
        #      they are well formed (look for NaNs, etc)

        # TODO PK handle calls where X.dim == 1.  This could occur if there was only 1 feature, or if
        #     there was only 1 sample?  We can differentiate either condition via y.dim and reshape
        #     AND add some tests for the X.dim == 1 scenario

        # TODO PK write an efficient striping converter for X that replaces unify_data for EBMs
        # algorithm: grap N columns and convert them to rows then process those by sending them to C

        # TODO: PK don't overwrite self.feature_names here (scikit-learn rules), and it's also confusing to
        #       user to have their fields overwritten.  Use feature_names_out_ or something similar

        X, y, self.feature_names, _ = unify_data(
            X, y, self.feature_names, self.feature_types, missing_data_allowed=True
        )

        # NOTE: Temporary override -- replace before push
        w = sample_weight if sample_weight is not None else np.ones_like(y, dtype=np.float64)
        w = unify_vector(w).astype(np.float64, casting="unsafe", copy=False)

        # Privacy calculations
        if isinstance(self, (DPExplainableBoostingClassifier, DPExplainableBoostingRegressor)):
            DPUtils.validate_eps_delta(self.epsilon, self.delta)
            DPUtils.validate_DP_EBM(self)

            if self.privacy_schema is None:
                warn("Possible privacy violation: assuming min/max values per feature/target are public info."
                     "Pass a privacy schema with known public ranges to avoid this warning.")
                self.privacy_schema = DPUtils.build_privacy_schema(X, y)

            self.domain_size_ = self.privacy_schema['target'][1] - self.privacy_schema['target'][0]

            # Split epsilon, delta budget for binning and learning
            bin_eps_ = self.epsilon * self.bin_budget_frac
            training_eps_ = self.epsilon - bin_eps_
            bin_delta_ = self.delta / 2
            training_delta_ = self.delta / 2

            # [DP] Calculate how much noise will be applied to each iteration of the algorithm
            if self.composition == 'classic':
                self.noise_scale_ = DPUtils.calc_classic_noise_multi(
                    total_queries=self.max_rounds * X.shape[1] * self.outer_bags,
                    target_epsilon=training_eps_,
                    delta=training_delta_,
                    sensitivity=self.domain_size_ * self.learning_rate * np.max(w)
                )
            elif self.composition == 'gdp':
                self.noise_scale_ = DPUtils.calc_gdp_noise_multi(
                    total_queries=self.max_rounds * X.shape[1] * self.outer_bags,
                    target_epsilon=training_eps_,
                    delta=training_delta_
                )
                self.noise_scale_ = self.noise_scale_ * self.domain_size_ * self.learning_rate * np.max(
                    w)  # Alg Line 17
            else:
                raise NotImplementedError(
                    f"Unknown composition method provided: {self.composition}. Please use 'gdp' or 'classic'.")
        else:
            bin_eps_, bin_delta_ = None, None
            training_eps_, training_delta_ = None, None

        # Build preprocessor
        self.preprocessor_ = EBMPreprocessor(
            feature_names=self.feature_names,
            feature_types=self.feature_types,
            max_bins=self.max_bins,
            binning=self.binning,
            epsilon=bin_eps_,  # Only defined during private training
            delta=bin_delta_,
            privacy_schema=getattr(self, 'privacy_schema', None)
        )
        self.preprocessor_.fit(X)
        X_orig = X
        X = self.preprocessor_.transform(X_orig)

        features_categorical = np.array([x == "categorical" for x in self.preprocessor_.col_types_], dtype=ct.c_int64)
        features_bin_count = np.array([len(x) for x in self.preprocessor_.col_bin_counts_], dtype=ct.c_int64)

        # NOTE: [DP] Passthrough to lower level layers for noise addition
        bin_data_counts = {i: self.preprocessor_.col_bin_counts_[i] for i in range(X.shape[1])}

        if self.interactions != 0:
            self.pair_preprocessor_ = EBMPreprocessor(
                feature_names=self.feature_names,
                feature_types=self.feature_types,
                max_bins=self.max_interaction_bins,
                binning=self.binning,
            )
            self.pair_preprocessor_.fit(X_orig)
            X_pair = self.pair_preprocessor_.transform(X_orig)
            pair_features_categorical = np.array([x == "categorical" for x in self.pair_preprocessor_.col_types_],
                                                 dtype=ct.c_int64)
            pair_features_bin_count = np.array([len(x) for x in self.pair_preprocessor_.col_bin_counts_],
                                               dtype=ct.c_int64)
        else:
            self.pair_preprocessor_, X_pair, pair_features_categorical, pair_features_bin_count = None, None, None, None

        estimators = []
        seed = EBMUtils.normalize_initial_random_seed(self.random_state)

        native = Native.get_native_singleton()
        if is_classifier(self):
            self.classes_, y = np.unique(y, return_inverse=True)
            self._class_idx_ = {x: index for index, x in enumerate(self.classes_)}

            y = y.astype(np.int64, casting="unsafe", copy=False)
            n_classes = len(self.classes_)
            if n_classes > 2:  # pragma: no cover
                warn("Multiclass is still experimental. Subject to change per release.")
            if n_classes > 2 and self.interactions != 0:
                self.interactions = 0
                warn("Detected multiclass problem: forcing interactions to 0")
            for i in range(self.outer_bags):
                seed = native.generate_random_number(seed, 1416147523)
                estimator = BaseCoreEBM(
                    # Data
                    model_type="classification",
                    features_categorical=features_categorical,
                    features_bin_count=features_bin_count,
                    pair_features_categorical=pair_features_categorical,
                    pair_features_bin_count=pair_features_bin_count,
                    # Core
                    main_features=self.mains,
                    interactions=self.interactions,
                    validation_size=self.validation_size,
                    max_rounds=self.max_rounds,
                    early_stopping_tolerance=self.early_stopping_tolerance,
                    early_stopping_rounds=self.early_stopping_rounds,
                    # Native
                    inner_bags=self.inner_bags,
                    learning_rate=self.learning_rate,
                    min_samples_leaf=self.min_samples_leaf,
                    max_leaves=self.max_leaves,
                    # Overall
                    random_state=seed,
                    # Differential Privacy
                    noise_scale=getattr(self, 'noise_scale_', None),
                    bin_counts=bin_data_counts,
                )
                estimators.append(estimator)
        else:
            n_classes = -1
            y = y.astype(np.float64, casting="unsafe", copy=False)
            for i in range(self.outer_bags):
                seed = native.generate_random_number(seed, 1416147523)
                estimator = BaseCoreEBM(
                    # Data
                    model_type="regression",
                    features_categorical=features_categorical,
                    features_bin_count=features_bin_count,
                    pair_features_categorical=pair_features_categorical,
                    pair_features_bin_count=pair_features_bin_count,
                    # Core
                    main_features=self.mains,
                    interactions=self.interactions,
                    validation_size=self.validation_size,
                    max_rounds=self.max_rounds,
                    early_stopping_tolerance=self.early_stopping_tolerance,
                    early_stopping_rounds=self.early_stopping_rounds,
                    # Native
                    inner_bags=self.inner_bags,
                    learning_rate=self.learning_rate,
                    min_samples_leaf=self.min_samples_leaf,
                    max_leaves=self.max_leaves,
                    # Overall
                    random_state=seed,
                    # Differential Privacy
                    noise_scale=getattr(self, 'noise_scale_', None),
                    bin_counts=bin_data_counts,
                )
                estimators.append(estimator)

        # Train base models for main effects, pair detection.

        # scikit-learn returns an np.array for classification and
        # a single float64 for regression, so we do the same
        if is_classifier(self):
            self.intercept_ = np.zeros(
                Native.get_count_scores_c(n_classes), dtype=np.float64, order="C",
            )
        else:
            self.intercept_ = np.float64(0)

        provider = JobLibProvider(n_jobs=self.n_jobs)

        train_model_args_iter = (
            (estimators[i], X, Off, y, w, X_pair, n_classes) for i in range(self.outer_bags)
        )

        estimators = provider.parallel(BaseCoreEBM.fit_parallel, train_model_args_iter)

        def select_pairs_from_fast(estimators, n_interactions):
            # Average rank from estimators
            pair_ranks = {}

            for n, estimator in enumerate(estimators):
                for rank, indices in enumerate(estimator.inter_indices_):
                    old_mean = pair_ranks.get(indices, 0)
                    pair_ranks[indices] = old_mean + ((rank - old_mean) / (n + 1))

            final_ranks = []
            total_interactions = 0
            for indices in pair_ranks:
                heapq.heappush(final_ranks, (pair_ranks[indices], indices))
                total_interactions += 1

            n_interactions = min(n_interactions, total_interactions)
            top_pairs = [heapq.heappop(final_ranks)[1] for _ in range(n_interactions)]
            return top_pairs

        if isinstance(self.interactions, int) and self.interactions > 0:
            # Select merged pairs
            pair_indices = select_pairs_from_fast(estimators, self.interactions)

            for estimator in estimators:
                # Discard initial interactions
                new_model = []
                new_feature_groups = []
                for i, feature_group in enumerate(estimator.feature_groups_):
                    if len(feature_group) != 1:
                        continue
                    new_model.append(estimator.model_[i])
                    new_feature_groups.append(estimator.feature_groups_[i])
                estimator.model_ = new_model
                estimator.feature_groups_ = new_feature_groups
                estimator.inter_episode_idx_ = 0

            if len(pair_indices) != 0:
                # Retrain interactions for base models

                # CAREFUL HERE!! ATTENTION - VALIDATE
                staged_fit_args_iter = (
                    (estimators[i], X, Off, y, w, X_pair, pair_indices) for i in range(self.outer_bags)
                )

                estimators = provider.parallel(BaseCoreEBM.staged_fit_interactions_parallel, staged_fit_args_iter)
        elif isinstance(self.interactions, int) and self.interactions == 0:
            pair_indices = []
        elif isinstance(self.interactions, list):
            pair_indices = self.interactions
            if len(pair_indices) != 0:
                # Check and remove duplicate interaction terms
                existing_terms = set()
                unique_terms = []

                for i, term in enumerate(pair_indices):
                    sorted_tuple = tuple(sorted(term))
                    if sorted_tuple not in existing_terms:
                        existing_terms.add(sorted_tuple)
                        unique_terms.append(term)

                # Warn the users that we have made change to the interactions list
                if len(unique_terms) != len(pair_indices):
                    warn("Detected duplicate interaction terms: removing duplicate interaction terms")
                    pair_indices = unique_terms
                    self.interactions = pair_indices

                # Retrain interactions for base models
                staged_fit_args_iter = (
                    (estimators[i], X, Off, y, w, X_pair, pair_indices) for i in range(self.outer_bags)
                )

                estimators = provider.parallel(BaseCoreEBM.staged_fit_interactions_parallel, staged_fit_args_iter)
        else:  # pragma: no cover
            raise RuntimeError("Argument 'interaction' has invalid value")

        X = np.ascontiguousarray(X.T)
        if X_pair is not None:
            X_pair = np.ascontiguousarray(X_pair.T)

        if isinstance(self.mains, str) and self.mains == "all":
            main_indices = [[x] for x in range(X.shape[0])]
        elif isinstance(self.mains, list) and all(
                isinstance(x, int) for x in self.mains
        ):
            main_indices = [[x] for x in self.mains]
        else:  # pragma: no cover
            msg = "Argument 'mains' has invalid value (valid values are 'all'|list<int>): {}".format(
                self.mains
            )
            raise RuntimeError(msg)

        self.feature_groups_ = main_indices + pair_indices

        self.bagged_models_ = estimators
        # Merge estimators into one.
        self.additive_terms_ = []
        self.term_standard_deviations_ = []
        for index, _ in enumerate(self.feature_groups_):
            log_odds_tensors = []
            for estimator in estimators:
                log_odds_tensors.append(estimator.model_[index])

            averaged_model = np.average(np.array(log_odds_tensors), axis=0)
            model_errors = np.std(np.array(log_odds_tensors), axis=0)

            self.additive_terms_.append(averaged_model)
            self.term_standard_deviations_.append(model_errors)

        # Get episode indexes for base estimators.
        main_episode_idxs = []
        inter_episode_idxs = []
        for estimator in estimators:
            main_episode_idxs.append(estimator.main_episode_idx_)
            inter_episode_idxs.append(estimator.inter_episode_idx_)

        self.breakpoint_iteration_ = [main_episode_idxs]
        if len(pair_indices) != 0:
            self.breakpoint_iteration_.append(inter_episode_idxs)

        # Extract feature group names and feature group types.
        # TODO PK v.3 don't overwrite feature_names and feature_types.  Create new fields called feature_names_out and
        #             feature_types_out_ or feature_group_names_ and feature_group_types_
        self.feature_names = []
        self.feature_types = []
        for index, feature_indices in enumerate(self.feature_groups_):
            feature_group_name = EBMUtils.gen_feature_group_name(
                feature_indices, self.preprocessor_.col_names_
            )
            feature_group_type = EBMUtils.gen_feature_group_type(
                feature_indices, self.preprocessor_.col_types_
            )
            self.feature_types.append(feature_group_type)
            self.feature_names.append(feature_group_name)

        if n_classes <= 2:
            if isinstance(self, (DPExplainableBoostingClassifier, DPExplainableBoostingRegressor)):
                # DP method of centering graphs can generalize if we log pairwise densities
                # No additional privacy loss from this step
                # self.additive_terms_ and self.preprocessor_.col_bin_counts_ are noisy and published publicly
                self._original_term_means_ = []
                for set_idx in range(len(self.feature_groups_)):
                    score_mean = np.average(self.additive_terms_[set_idx],
                                            weights=self.preprocessor_.col_bin_counts_[set_idx])
                    self.additive_terms_[set_idx] = (
                            self.additive_terms_[set_idx] - score_mean
                    )

                    # Add mean center adjustment back to intercept
                    self.intercept_ += score_mean
                    self._original_term_means_.append(score_mean)
            else:

                # Mean center graphs - only for binary classification and regression
                scores_gen = EBMUtils.scores_by_feature_group(
                    X, X_pair, self.feature_groups_, self.additive_terms_
                )

                print("HELLO")
                score_mean = np.average(Off, weights=w)
                print("BYE!")

                self._original_term_means_ = []

                for set_idx, _, scores in scores_gen:
                    score_mean = np.average(scores, weights=w)
                    #score_mean += np.average(scores, weights=w)

                    self.additive_terms_[set_idx] = (
                            self.additive_terms_[set_idx] - score_mean
                    )

                    # Add mean center adjustment back to intercept
                    self.intercept_ += score_mean
                    self._original_term_means_.append(score_mean)
        else:
            # Postprocess model graphs for multiclass
            # Currently pairwise interactions are unsupported for multiclass-classification.
            binned_predict_proba = lambda x: EBMUtils.classifier_predict_proba(
                x, None, Off, self.feature_groups_, self.additive_terms_, self.intercept_
            )

            postprocessed = multiclass_postprocess(
                X, self.additive_terms_, binned_predict_proba, self.feature_types
            )
            self.additive_terms_ = postprocessed["feature_graphs"]
            self.intercept_ = postprocessed["intercepts"]

        for feature_group_idx, feature_group in enumerate(self.feature_groups_):
            entire_tensor = [slice(None, None, None) for i in range(self.additive_terms_[feature_group_idx].ndim)]
            for dimension_idx, feature_idx in enumerate(feature_group):
                if self.preprocessor_.col_bin_counts_[feature_idx][0] == 0:
                    zero_dimension = entire_tensor.copy()
                    zero_dimension[dimension_idx] = 0
                    self.additive_terms_[feature_group_idx][tuple(zero_dimension)] = 0
                    self.term_standard_deviations_[feature_group_idx][tuple(zero_dimension)] = 0

        # Generate overall importance
        self.feature_importances_ = []
        if isinstance(self, (DPExplainableBoostingClassifier, DPExplainableBoostingRegressor)):
            # DP method of generating feature importances can generalize to non-dp if preprocessors start tracking joint distributions
            for i in range(len(self.feature_groups_)):
                mean_abs_score = np.average(np.abs(self.additive_terms_[i]),
                                            weights=self.preprocessor_.col_bin_counts_[i])
                self.feature_importances_.append(mean_abs_score)
        else:
            scores_gen = EBMUtils.scores_by_feature_group(
                X, X_pair, self.feature_groups_, self.additive_terms_
            )
            for set_idx, _, scores in scores_gen:
                mean_abs_score = np.mean(np.abs(scores))
                self.feature_importances_.append(mean_abs_score)

        # Generate selector
        # TODO PK v.3 shouldn't this be self._global_selector_ ??
        self.global_selector = gen_global_selector(
            X_orig, self.feature_names, self.feature_types, None
        )

        self.has_fitted_ = True
        return self

    # Select pairs from base models
    def _merged_pair_score_fn(self, model_type, X, Off, y, X_pair, feature_groups, model, intercept):

        if model_type == "classification":
            prob = EBMUtils.classifier_predict_proba(
                X, X_pair, Off, feature_groups, model, intercept
            )
            return (
                0 if len(y) == 0 else log_loss(y, prob)
            )  # use logloss to conform consistnetly and for multiclass
        elif model_type == "regression":
            pred = EBMUtils.regressor_predict(
                X, X_pair, Off, feature_groups, model, intercept
            )
            return 0 if len(y) == 0 else mean_squared_error(y, pred)
        else:  # pragma: no cover
            msg = "Unknown model_type: '{}'.".format(model_type)
            raise ValueError(msg)

    def decision_function(self, X, Off):
        """ Predict scores from model before calling the link function.

            Args:
                X: Numpy array for samples.

            Returns:
                The sum of the additive term contributions.
        """
        check_is_fitted(self, "has_fitted_")
        X_orig, _, _, _ = unify_data(X, None, self.feature_names, self.feature_types, missing_data_allowed=True)
        X = self.preprocessor_.transform(X_orig)
        X = np.ascontiguousarray(X.T)

        if self.interactions != 0:
            X_pair = self.pair_preprocessor_.transform(X_orig)
            X_pair = np.ascontiguousarray(X_pair.T)
        else:
            X_pair = None

        decision_scores = EBMUtils.decision_function(
            X, X_pair, Off, self.feature_groups_, self.additive_terms_, self.intercept_
        )

        return decision_scores

    # TODO
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

        check_is_fitted(self, "has_fitted_")

        # Obtain min/max for model scores
        lower_bound = np.inf
        upper_bound = -np.inf
        for feature_group_index, _ in enumerate(self.feature_groups_):
            errors = self.term_standard_deviations_[feature_group_index]
            scores = self.additive_terms_[feature_group_index]

            lower_bound = min(lower_bound, np.min(scores - errors))
            upper_bound = max(upper_bound, np.max(scores + errors))

        bounds = (lower_bound, upper_bound)

        # Add per feature graph
        data_dicts = []
        feature_list = []
        density_list = []
        for feature_group_index, feature_indexes in enumerate(
                self.feature_groups_
        ):
            model_graph = self.additive_terms_[feature_group_index]

            # NOTE: This uses stddev. for bounds, consider issue warnings.
            errors = self.term_standard_deviations_[feature_group_index]

            if len(feature_indexes) == 1:
                # hack. remove the 0th index which is for missing values
                model_graph = model_graph[1:]
                errors = errors[1:]

                bin_labels = self.preprocessor_._get_bin_labels(feature_indexes[0])
                # bin_counts = self.preprocessor_.get_bin_counts(
                #     feature_indexes[0]
                # )
                scores = list(model_graph)
                upper_bounds = list(model_graph + errors)
                lower_bounds = list(model_graph - errors)
                density_dict = {
                    "names": self.preprocessor_._get_hist_edges(feature_indexes[0]),
                    "scores": self.preprocessor_._get_hist_counts(feature_indexes[0]),
                }

                feature_dict = {
                    "type": "univariate",
                    "names": bin_labels,
                    "scores": scores,
                    "scores_range": bounds,
                    "upper_bounds": upper_bounds,
                    "lower_bounds": lower_bounds,
                }
                feature_list.append(feature_dict)
                density_list.append(density_dict)

                data_dict = {
                    "type": "univariate",
                    "names": bin_labels,
                    "scores": model_graph,
                    "scores_range": bounds,
                    "upper_bounds": model_graph + errors,
                    "lower_bounds": model_graph - errors,
                    "density": {
                        "names": self.preprocessor_._get_hist_edges(feature_indexes[0]),
                        "scores": self.preprocessor_._get_hist_counts(
                            feature_indexes[0]
                        ),
                    },
                }
                if is_classifier(self):
                    data_dict["meta"] = {
                        "label_names": self.classes_.tolist()  # Classes should be numpy array, convert to list.
                    }

                data_dicts.append(data_dict)
            elif len(feature_indexes) == 2:
                # hack. remove the 0th index which is for missing values
                model_graph = model_graph[1:, 1:]
                # errors = errors[1:, 1:]  # NOTE: This is commented as it's not used in this branch.

                bin_labels_left = self.pair_preprocessor_._get_bin_labels(feature_indexes[0])
                bin_labels_right = self.pair_preprocessor_._get_bin_labels(feature_indexes[1])

                feature_dict = {
                    "type": "interaction",
                    "left_names": bin_labels_left,
                    "right_names": bin_labels_right,
                    "scores": model_graph,
                    "scores_range": bounds,
                }
                feature_list.append(feature_dict)
                density_list.append({})

                data_dict = {
                    "type": "interaction",
                    "left_names": bin_labels_left,
                    "right_names": bin_labels_right,
                    "scores": model_graph,
                    "scores_range": bounds,
                }
                data_dicts.append(data_dict)
            else:  # pragma: no cover
                raise Exception("Interactions greater than 2 not supported.")

        overall_dict = {
            "type": "univariate",
            "names": self.feature_names,
            "scores": self.feature_importances_,
        }
        internal_obj = {
            "overall": overall_dict,
            "specific": data_dicts,
            "mli": [
                {
                    "explanation_type": "ebm_global",
                    "value": {"feature_list": feature_list},
                },
                {"explanation_type": "density", "value": {"density": density_list}},
            ],
        }

        return EBMExplanation(
            "global",
            internal_obj,
            feature_names=self.feature_names,
            feature_types=self.feature_types,
            name=name,
            selector=self.global_selector,
        )

    # TODO
    def explain_local(self, X, y=None, name=None):
        """ Provides local explanations for provided samples.

        Args:
            X: Numpy array for X to explain.
            y: Numpy vector for y to explain.
            name: User-defined explanation name.

        Returns:
            An explanation object, visualizing feature-value pairs
            for each sample as horizontal bar charts.
        """

        # Produce feature value pairs for each sample.
        # Values are the model graph score per respective feature group.
        if name is None:
            name = gen_name_from_class(self)

        check_is_fitted(self, "has_fitted_")

        X, y, _, _ = unify_data(X, y, self.feature_names, self.feature_types, missing_data_allowed=True)

        # Transform y if classifier
        if is_classifier(self) and y is not None:
            y = np.array([self._class_idx_[el] for el in y])

        samples = self.preprocessor_.transform(X)
        samples = np.ascontiguousarray(samples.T)

        if self.interactions != 0:
            pair_samples = self.pair_preprocessor_.transform(X)
            pair_samples = np.ascontiguousarray(pair_samples.T)
        else:
            pair_samples = None

        scores_gen = EBMUtils.scores_by_feature_group(
            samples, pair_samples, self.feature_groups_, self.additive_terms_
        )

        # TODO PK add a test to see if we handle X.ndim == 1 (or should we throw ValueError)

        n_rows = samples.shape[1]
        data_dicts = []
        intercept = self.intercept_
        if not is_classifier(self) or len(self.classes_) <= 2:
            if isinstance(self.intercept_, np.ndarray) or isinstance(
                    self.intercept_, list
            ):
                intercept = intercept[0]

        for _ in range(n_rows):
            data_dict = {
                "type": "univariate",
                "names": [],
                "scores": [],
                "values": [],
                "extra": {"names": ["Intercept"], "scores": [intercept], "values": [1]},
            }
            if is_classifier(self):
                data_dict["meta"] = {
                    "label_names": self.classes_.tolist()  # Classes should be numpy array, convert to list.
                }
            data_dicts.append(data_dict)

        for set_idx, feature_group, scores in scores_gen:
            for row_idx in range(n_rows):
                feature_name = self.feature_names[set_idx]
                data_dicts[row_idx]["names"].append(feature_name)
                data_dicts[row_idx]["scores"].append(scores[row_idx])
                if len(feature_group) == 1:
                    data_dicts[row_idx]["values"].append(
                        X[row_idx, feature_group[0]]
                    )
                else:
                    data_dicts[row_idx]["values"].append("")

        is_classification = is_classifier(self)
        if is_classification:
            scores = EBMUtils.classifier_predict_proba(
                samples, pair_samples, self.feature_groups_, self.additive_terms_, self.intercept_,
            )
        else:
            scores = EBMUtils.regressor_predict(
                samples, pair_samples, self.feature_groups_, self.additive_terms_, self.intercept_,
            )

        perf_list = []
        perf_dicts = gen_perf_dicts(scores, y, is_classification)
        for row_idx in range(n_rows):
            perf = None if perf_dicts is None else perf_dicts[row_idx]
            perf_list.append(perf)
            data_dicts[row_idx]["perf"] = perf

        selector = gen_local_selector(data_dicts, is_classification=is_classification)

        additive_terms = []
        for feature_group_index, feature_indexes in enumerate(self.feature_groups_):
            if len(feature_indexes) == 1:
                # hack. remove the 0th index which is for missing values
                additive_terms.append(self.additive_terms_[feature_group_index][1:])
            elif len(feature_indexes) == 2:
                # hack. remove the 0th index which is for missing values
                additive_terms.append(self.additive_terms_[feature_group_index][1:, 1:])
            else:
                raise ValueError("only handles 1D/2D")

        internal_obj = {
            "overall": None,
            "specific": data_dicts,
            "mli": [
                {
                    "explanation_type": "ebm_local",
                    "value": {
                        "scores": additive_terms,
                        "intercept": self.intercept_,
                        "perf": perf_list,
                    },
                }
            ],
        }
        internal_obj["mli"].append(
            {
                "explanation_type": "evaluation_dataset",
                "value": {"dataset_x": X, "dataset_y": y},
            }
        )

        return EBMExplanation(
            "local",
            internal_obj,
            feature_names=self.feature_names,
            feature_types=self.feature_types,
            name=name,
            selector=selector,
        )


class OffsetExplainableBoostingClassifier(OffsetBaseEBM, ClassifierMixin, ExplainerMixin):
    # TODO PK v.3 use underscores here like ClassifierMixin._estimator_type?
    available_explanations = ["global", "local"]
    explainer_type = "model"

    def __init__(
        self,
        # Explainer
        feature_names=None,
        feature_types=None,
        # Preprocessor
        max_bins=256,
        max_interaction_bins=32,
        binning="quantile",
        # Stages
        mains="all",
        interactions=10,
        # Ensemble
        outer_bags=8,
        inner_bags=0,
        # Boosting
        learning_rate=0.01,
        validation_size=0.15,
        early_stopping_rounds=50,
        early_stopping_tolerance=1e-4,
        max_rounds=5000,
        # Trees
        min_samples_leaf=2,
        max_leaves=3,
        # Overall
        n_jobs=-2,
        random_state=42,
    ):
        """ Explainable Boosting Classifier. The arguments will change in a future release, watch the changelog.

        Args:
            feature_names: List of feature names.
            feature_types: List of feature types.
            max_bins: Max number of bins per feature for pre-processing stage.
            max_interaction_bins: Max number of bins per feature for pre-processing stage on interaction terms. Only used if interactions is non-zero.
            binning: Method to bin values for pre-processing. Choose "uniform", "quantile" or "quantile_humanized".
            mains: Features to be trained on in main effects stage. Either "all" or a list of feature indexes.
            interactions: Interactions to be trained on.
                Either a list of lists of feature indices, or an integer for number of automatically detected interactions.
                Interactions are forcefully set to 0 for multiclass problems.
            outer_bags: Number of outer bags.
            inner_bags: Number of inner bags.
            learning_rate: Learning rate for boosting.
            validation_size: Validation set size for boosting.
            early_stopping_rounds: Number of rounds of no improvement to trigger early stopping.
            early_stopping_tolerance: Tolerance that dictates the smallest delta required to be considered an improvement.
            max_rounds: Number of rounds for boosting.
            min_samples_leaf: Minimum number of cases for tree splits used in boosting.
            max_leaves: Maximum leaf nodes used in boosting.
            n_jobs: Number of jobs to run in parallel.
            random_state: Random state.
        """
        super(OffsetExplainableBoostingClassifier, self).__init__(
            # Explainer
            feature_names=feature_names,
            feature_types=feature_types,
            # Preprocessor
            max_bins=max_bins,
            max_interaction_bins=max_interaction_bins,
            binning=binning,
            # Stages
            mains=mains,
            interactions=interactions,
            # Ensemble
            outer_bags=outer_bags,
            inner_bags=inner_bags,
            # Boosting
            learning_rate=learning_rate,
            validation_size=validation_size,
            early_stopping_rounds=early_stopping_rounds,
            early_stopping_tolerance=early_stopping_tolerance,
            max_rounds=max_rounds,
            # Trees
            min_samples_leaf=min_samples_leaf,
            max_leaves=max_leaves,
            # Overall
            n_jobs=n_jobs,
            random_state=random_state,
        )

    # TODO: Throw ValueError like scikit for 1d instead of 2d arrays
    def predict_proba(self, X, Off):
        """ Probability estimates on provided samples.

        Args:
            X: Numpy array for samples.

        Returns:
            Probability estimate of sample for each class.
        """
        check_is_fitted(self, "has_fitted_")
        X_orig, _, _, _ = unify_data(X, None, self.feature_names, self.feature_types, missing_data_allowed=False)
        X = self.preprocessor_.transform(X_orig)
        X = np.ascontiguousarray(X.T)

        if self.interactions != 0:
            X_pair = self.pair_preprocessor_.transform(X_orig)
            X_pair = np.ascontiguousarray(X_pair.T)
        else:
            X_pair = None

        # TODO PK add a test to see if we handle X.ndim == 1 (or should we throw ValueError)

        prob = EBMUtils.classifier_predict_proba(
            X, X_pair, Off, self.feature_groups_, self.additive_terms_, self.intercept_
        )
        return prob

    def predict(self, X, Off):
        """ Predicts on provided samples.

        Args:
            X: Numpy array for samples.

        Returns:
            Predicted class label per sample.
        """
        check_is_fitted(self, "has_fitted_")
        X_orig, _, _, _ = unify_data(X, None, self.feature_names, self.feature_types, missing_data_allowed=True)
        X = self.preprocessor_.transform(X_orig)
        X = np.ascontiguousarray(X.T)

        if self.interactions != 0:
            X_pair = self.pair_preprocessor_.transform(X_orig)
            X_pair = np.ascontiguousarray(X_pair.T)
        else:
            X_pair = None

        # TODO PK add a test to see if we handle X.ndim == 1 (or should we throw ValueError)

        return EBMUtils.classifier_predict(
            X,
            X_pair,
            Off,
            self.feature_groups_,
            self.additive_terms_,
            self.intercept_,
            self.classes_,
        )

    def predict_and_contrib(self, X, Off, output='probabilities'):
        """Predicts on provided samples, returning predictions and explanations for each sample.

        Args:
            X: Numpy array for samples.
            output: Prediction type to output (i.e. one of 'probabilities', 'logits', 'labels')

        Returns:
            Predictions and local explanations for each sample.
        """

        allowed_outputs = ['probabilities', 'logits', 'labels']
        if output not in allowed_outputs:
            msg = "Argument 'output' has invalid value.  Got '{}', expected one of "
            + repr(allowed_outputs)
            raise ValueError(msg.format(output))

        check_is_fitted(self, "has_fitted_")
        X_orig, _, _, _ = unify_data(
            X, None, self.feature_names, self.feature_types, missing_data_allowed=True
        )
        X = self.preprocessor_.transform(X_orig)
        X = np.ascontiguousarray(X.T)

        if self.interactions != 0:
            X_pair = self.pair_preprocessor_.transform(X_orig)
            X_pair = np.ascontiguousarray(X_pair.T)
        else:
            X_pair = None

        return EBMUtils.classifier_predict_and_contrib(
            X,
            X_pair,
            Off,
            self.feature_groups_,
            self.additive_terms_,
            self.intercept_,
            self.classes_,
            output)


# Add offset to ExplainableBoostingRegressor
class OffsetExplainableBoostingRegressor(OffsetBaseEBM, RegressorMixin, ExplainerMixin):

    # TODO PK v.3 use underscores here like RegressorMixin._estimator_type?
    available_explanations = ["global", "local"]
    explainer_type = "model"

    """ Public facing EBM regressor."""

    def __init__(
        self,
        # Explainer
        feature_names=None,
        feature_types=None,
        # Preprocessor
        max_bins=256,
        max_interaction_bins=32,
        binning="quantile",
        # Stages
        mains="all",
        interactions=10,
        # Ensemble
        outer_bags=8,
        inner_bags=0,
        # Boosting
        learning_rate=0.01,
        validation_size=0.15,
        early_stopping_rounds=50,
        early_stopping_tolerance=1e-4,
        max_rounds=5000,
        # Trees
        min_samples_leaf=2,
        max_leaves=3,
        # Overall
        n_jobs=-2,
        random_state=42,
    ):
        """ Explainable Boosting Regressor. The arguments will change in a future release, watch the changelog.

        Args:
            feature_names: List of feature names.
            feature_types: List of feature types.
            max_bins: Max number of bins per feature for pre-processing stage on main effects.
            max_interaction_bins: Max number of bins per feature for pre-processing stage on interaction terms. Only used if interactions is non-zero.
            binning: Method to bin values for pre-processing. Choose "uniform", "quantile", or "quantile_humanized".
            mains: Features to be trained on in main effects stage. Either "all" or a list of feature indexes.
            interactions: Interactions to be trained on.
                Either a list of lists of feature indices, or an integer for number of automatically detected interactions.
            outer_bags: Number of outer bags.
            inner_bags: Number of inner bags.
            learning_rate: Learning rate for boosting.
            validation_size: Validation set size for boosting.
            early_stopping_rounds: Number of rounds of no improvement to trigger early stopping.
            early_stopping_tolerance: Tolerance that dictates the smallest delta required to be considered an improvement.
            max_rounds: Number of rounds for boosting.
            min_samples_leaf: Minimum number of cases for tree splits used in boosting.
            max_leaves: Maximum leaf nodes used in boosting.
            n_jobs: Number of jobs to run in parallel.
            random_state: Random state.
        """
        super(OffsetExplainableBoostingRegressor, self).__init__(
            # Explainer
            feature_names=feature_names,
            feature_types=feature_types,
            # Preprocessor
            max_bins=max_bins,
            max_interaction_bins=max_interaction_bins,
            binning=binning,
            # Stages
            mains=mains,
            interactions=interactions,
            # Ensemble
            outer_bags=outer_bags,
            inner_bags=inner_bags,
            # Boosting
            learning_rate=learning_rate,
            validation_size=validation_size,
            early_stopping_rounds=early_stopping_rounds,
            early_stopping_tolerance=early_stopping_tolerance,
            max_rounds=max_rounds,
            # Trees
            min_samples_leaf=min_samples_leaf,
            max_leaves=max_leaves,
            # Overall
            n_jobs=n_jobs,
            random_state=random_state,
        )

    def predict(self, X, Off):
        """ Predicts on provided samples.

        Args:
            X: Numpy array for samples.

        Returns:
            Predicted class label per sample.
        """
        check_is_fitted(self, "has_fitted_")
        X_orig, _, _, _ = unify_data(X, None, self.feature_names, self.feature_types, missing_data_allowed=True)
        X = self.preprocessor_.transform(X_orig)
        X = np.ascontiguousarray(X.T)

        if self.interactions != 0:
            X_pair = self.pair_preprocessor_.transform(X_orig)
            X_pair = np.ascontiguousarray(X_pair.T)
        else:
            X_pair = None

        return EBMUtils.regressor_predict(
            X, X_pair, Off, self.feature_groups_, self.additive_terms_, self.intercept_
        )


    def predict_and_contrib(self, X, Off):
        """Predicts on provided samples, returning predictions and explanations for each sample.

        Args:
            X: Numpy array for samples.

        Returns:
            Predictions and local explanations for each sample.
        """

        check_is_fitted(self, "has_fitted_")
        X_orig, _, _, _ = unify_data(
            X, None, self.feature_names, self.feature_types, missing_data_allowed=True
        )
        X = self.preprocessor_.transform(X_orig)
        X = np.ascontiguousarray(X.T)

        if self.interactions != 0:
            X_pair = self.pair_preprocessor_.transform(X_orig)
            X_pair = np.ascontiguousarray(X_pair.T)
        else:
            X_pair = None

        return EBMUtils.regressor_predict_and_contrib(
            X, X_pair, Off, self.feature_groups_, self.additive_terms_, self.intercept_
        )


# Replace with Offset counterparts in locally
EBMUtils = OffsetEBMUtils
BaseEBM = OffsetBaseEBM
BaseCoreEBM = OffsetBaseCoreEBM
ExplainableBoostingClassifier = OffsetExplainableBoostingClassifier




def main():

    # Problem setting
    np.random.seed(2023)
    NB_SAMPLES = 1000
    NB_FEATURES = 2
    INTERACTIONS = 0
    validate_classification = False
    validate_regression = True

    # Generate data
    X = np.random.random((NB_SAMPLES, NB_FEATURES))
    Off = np.random.random(NB_SAMPLES)

    if bool(INTERACTIONS):
        y_C = np.array([int(sum(x) + o > (NB_FEATURES + 1) / 2.0) for x, o in zip(X, Off)], dtype=np.float64)
        y_R = np.array([sum(x) + x[1] * o + x[-3] * x[-2] + o for x, o in zip(X, Off)], dtype=np.float64)
    else:
        #y_R = np.array([5.0 * (x > 0.5) + o for x, o in zip(X, Off)], dtype=np.float64)
        y_C = np.array([int(sum(x) + o > (NB_FEATURES + 1) / 2.0) for x, o in zip(X, Off)], dtype=np.float64)
        y_R = np.array([sum(x) + o for x, o in zip(X, Off)], dtype=np.float64)

    # Split 80% train 20% test
    train_idx = np.random.choice(NB_SAMPLES, int(NB_SAMPLES * 0.8))
    test_idx = [idx for idx in range(NB_SAMPLES) if idx not in train_idx]
    X_train, X_test = X[train_idx, :], X[test_idx, :]
    Off_train, Off_test = Off[train_idx], Off[test_idx]
    X_off_train, X_off_test = np.column_stack([X_train, Off_train]), np.column_stack([X_test, Off_test])
    y_C_train, y_C_test = y_C[train_idx], y_C[test_idx]
    y_R_train, y_R_test = y_R[train_idx], y_R[test_idx]

    if validate_classification:
        test_acc = {}

        # A: Fit base EBM and using offset as a feature
        model_A = ExplainableBoostingClassifier(interactions=INTERACTIONS)
        model_A.fit(X=X_off_train, y=y_C_train, sample_weight=None)
        y_hat_A = model_A.predict(X_off_test)
        test_acc['A'] = np.equal(y_hat_A, y_C_test).mean()

        # B: Use model A to score Offset f_A(off), the use that as the offset
        offset_score_train = model_A.predict_and_contrib(X_off_train)[1][:, -1]
        offset_score_test = model_A.predict_and_contrib(X_off_test)[1][:, -1]
        model_B = OffsetExplainableBoostingClassifier(interactions=INTERACTIONS)
        model_B.fit(X=X_train, Off=offset_score_train, y=y_C_train, sample_weight=None)
        y_hat_B = model_B.predict(X_test, offset_score_test)
        test_acc['B'] = np.equal(y_hat_B, y_C_test).mean()
        print(model_B.intercept_)

        # C: Don't score offset using model A, but use it directly
        model_C = OffsetExplainableBoostingClassifier(interactions=INTERACTIONS)
        model_C.fit(X=X_train, Off=Off_train, y=y_C_train, sample_weight=None)
        y_hat_C = model_C.predict(X_test, Off_test)
        test_acc['C'] = np.equal(y_hat_C, y_C_test).mean()

        print("\n Classification Results (Test Accuracy, interaction =", bool(INTERACTIONS), ")...")
        print("y_true = int(x_1 + ... + x_", NB_FEATURES, "\b + off > ", (NB_FEATURES+1.0)/2, ")")
        print("Test MSE:", test_acc['A'], "Approach A: Model=BaseEBM, X=(x,off), Y=y")
        print("Test MSE:", test_acc['B'], "Approach B: Model=OffsetEBM, X=x, Off=f_A(off), Y=y")
        #print("Test MSE:", test_acc['C'], "Approach C: Model=OffsetEBM, X=x, Off=off, Y=y")

    if validate_regression:
        test_mse = {}
        intercept = {}
        pred_1 = {}

        # Offset model, X=x, Off=off, Y=y
        model = OffsetExplainableBoostingRegressor(interactions=INTERACTIONS)
        model.fit(X=X_train, Off=Off_train, y=y_R_train, sample_weight=None)
        y_hat_R = model.predict(X_test, Off_test)
        test_mse['A'] = np.mean((y_hat_R - y_R_test) ** 2)
        intercept['A'] = model.intercept_

        # Offset model, X=x, Off=None, Y=(y-off)
        model = OffsetExplainableBoostingRegressor(interactions=INTERACTIONS)
        model.fit(X=X_train, Off=Off_train * 0.0, y=y_R_train - Off_train, sample_weight=None)
        y_hat_R = model.predict(X_test, Off_test * 0.0)
        test_mse['B'] = np.mean((y_hat_R + Off_test - y_R_test) ** 2)
        intercept['B'] = model.intercept_

        # Offset model, X=(x, off), Off=None, Y=y
        model = OffsetExplainableBoostingRegressor(interactions=INTERACTIONS)
        model.fit(X=X_off_train, Off=Off_train * 0, y=y_R_train, sample_weight=None)
        y_hat_R = model.predict(X_off_test, Off_test * 0)
        test_mse['C'] = np.mean((y_hat_R - y_R_test) ** 2)
        intercept['C'] = model.intercept_

        # Base model, X=x, Y=(y-off)
        model = ExplainableBoostingRegressor(interactions=INTERACTIONS)
        model.fit(X=X_train, y=y_R_train - Off_train, sample_weight=None)
        y_hat_R = model.predict(X_test)
        test_mse['D'] = np.mean((y_hat_R + Off_test - y_R_test) ** 2)
        intercept['D'] = model.intercept_

        # Base model, X=(x,off), Y=y
        model = ExplainableBoostingRegressor(interactions=INTERACTIONS)
        model.fit(X=X_off_train, y=y_R_train, sample_weight=None)
        y_hat_R = model.predict(X_off_test)
        test_mse['E'] = np.mean((y_hat_R - y_R_test) ** 2)
        intercept['E'] = model.intercept_

        if bool(INTERACTIONS):
            # Offset model, X=(x, off), Off=None, Y=y (A3)
            model = OffsetExplainableBoostingRegressor(interactions=INTERACTIONS)
            model.fit(X=X_off_train, Off=Off_train, y=y_R_train, sample_weight=None)
            y_hat_R = model.predict(X_off_test, Off_test)
            test_mse['F'] = np.mean((y_hat_R - y_R_test) ** 2)

        print("\nRegression Results (Test MSE, interaction =", bool(INTERACTIONS), ")...")
        print("y_true = x1 + x2 + x3 + x4 + x5 + off + (x1*off) + (x3*x4)")
        print("Test MSE:", test_mse['A'], "Approach A: Model=OffsetEBM, X=x, Off=off, Y=y")
        print("Test MSE:", test_mse['B'], "Approach B: Model=OffsetEBM, X=x, Off=None, Y=(y-off)")
        print("Test MSE:", test_mse['C'], "Approach C: Model=OffsetEBM, X=(x, off), Off=None, Y=y")
        print("Test MSE:", test_mse['D'], "Approach D: Model=BaseEBM, X=x, Y=(y-off)")
        print("Test MSE:", test_mse['E'], "Approach E: Model=BaseEBM, X=(x,off), Y=y")
        print(intercept)

        if bool(INTERACTIONS):
            print("Test MSE:", test_mse['F'], "Approach F: Model=OffsetEBM, X=(x,off), Off=off, Y=y")


if __name__ == "__main__":
    main()
