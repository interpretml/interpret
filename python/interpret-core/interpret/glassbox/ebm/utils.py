# Copyright (c) 2019 Microsoft Corporation

# Distributed under the MIT software license
# TODO: Test EBMUtils

from math import ceil
from .internal import Native, Booster, InteractionDetector

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


# TODO: Clean up
class EBMUtils:
    
    @staticmethod
    def weighted_std(a, axis, weights):

        average = np.average(a, axis , weights)
        
        variance = np.average((a - average)**2, axis , weights)

        return np.sqrt(variance)

    @staticmethod
    def merge_models(models):
        """ Merging multiple EBM models trained on the same dataset.
        Args:
            models: List of EBM models to be merged.
        Returns:
            An EBM model with averaged mean and standard deviation of input models.
        """

        if len(models) < 2:  # pragma: no cover
            raise Exception("at least two models are required to merge.")

        # many features are invalid. preprocessor_ and pair_preprocessor_ are cloned form the first model.
        ebm = copy.deepcopy(models[0]) 
        

        ebm.additive_terms_ =[]
        ebm.term_standard_deviations_ = []
        ebm.bagged_models_=[]
        ebm.pair_preprocessor_ = None       
      
         
        if not all([  model.preprocessor_.col_types_ == ebm.preprocessor_.col_types_ for model in models]):  # pragma: no cover
            raise Exception("All models should have the same types of features.")
       
        if not all([  model.preprocessor_.col_bin_edges_.keys() == ebm.preprocessor_.col_bin_edges_.keys() for model in models]):  # pragma: no cover
                    raise Exception("All models should have the same types of numeric features.")


        if not all([  model.preprocessor_.col_mapping_.keys() == ebm.preprocessor_.col_mapping_.keys() for model in models]):  # pragma: no cover
                    raise Exception("All models should have the same types of categorical features.")

        if is_classifier(ebm):  # pragma: no cover
                if not all([is_classifier(model) for model in models]):
                    raise Exception("All models should be the same type.")
        else:  # pragma: no cover
                # Check if ebm is a regressor
                if any([is_classifier(model) for model in models]):
                    raise Exception("All models should be the same type.")

        main_feature_len = len(ebm.preprocessor_.feature_names)

        ebm.feature_groups_ = ebm.feature_groups_[:main_feature_len] 
        ebm.feature_names = ebm.feature_names[:main_feature_len] 
        ebm.feature_types = ebm.feature_types[:main_feature_len]

        ebm.global_selector = ebm.global_selector.iloc[:main_feature_len]
        ebm.interactions = 0
        
        warnings.warn("Interaction features are not supported.")
        
       
        ebm.additive_terms_ = []
        ebm.term_standard_deviations_ = []
        
        bagged_models = []
        for x in models:
            bagged_models.extend(x.bagged_models_) #             
        ebm.bagged_models_ = bagged_models
 
        for index, feature_group in enumerate(ebm.feature_groups_):           

            # Exluding interaction features 
            if len(feature_group) != 1:                
                continue

            log_odds_tensors = []
            bin_weights = []
            
            # numeric features
            if index in ebm.preprocessor_.col_bin_edges_.keys():           
                        
                # merging all bin edges for the current feature across all models.        
                merged_bin_edges = sorted(set().union(*[ set(model.preprocessor_.col_bin_edges_[index]) for model in models]))
                
                ebm.preprocessor_.col_bin_edges_[index] = np.array(merged_bin_edges)
                
            
                estimator_idx=0
                for model in models:            
                    # Merging the bin edges for different models for each feature group
                    bin_edges = model.preprocessor_.col_bin_edges_[index]
                    bin_counts = model.preprocessor_.col_bin_counts_[index]

                    #bin_idx contains the index of each new merged bin edges against the existing bin edges
                    bin_idx = np.searchsorted(bin_edges, merged_bin_edges + [np.inf])   
                    # the first element of adjusted bin indexes is used for weighted average of misssing values.
                    adj_bin_idx = [0] + ( [ x+1 for x in bin_idx ])
                    
                    # new_bin_count_ =  [ bin_counts[x] for x in adj_bin_idx ]  
                                        
                    # All the estimators of one ebm model share the same bin edges
                    for estimator in model.bagged_models_: 

                        mvalues = estimator.model_[index]      
                        
                        # expanding the prediction values to cover all the new merged bin edges                                              
                        new_model_ = [ mvalues[x] for x in adj_bin_idx ]

                        # updating the new EBM model estimator predictions with the new extended predictions
                        ebm.bagged_models_[estimator_idx].model_[index] = new_model_
                        
                        # bin counts are used as weights to calculate weighted average of new merged bins.                        
                        weights =[ bin_counts[x] for x in adj_bin_idx ]
                        
                        log_odds_tensors.append(new_model_)
                        bin_weights.append( weights)
                        
                        estimator_idx +=1

                                
            else:
                # Categorical features
                merged_col_mapping = sorted(set().union(*[ set(model.preprocessor_.col_mapping_[index]) for model in models]))
            
                ebm.preprocessor_.col_mapping_[index] = dict( (key, idx +1) for idx, key in enumerate(merged_col_mapping))
                
                for model in models: 
                                        
                    bin_counts = model.preprocessor_.col_bin_counts_[index]
                    
                    # mask contains the category values for common categories and None for missing ones for each categorical feature
                    mask = [ model.preprocessor_.col_mapping_[index].get(col, None ) for col in merged_col_mapping]

                    for estimator in model.bagged_models_:

                        mvalues = estimator.model_[index]                           
                        new_model_ = [mvalues[0]] + [ mvalues[i] if i else 0.0 for i in mask]
                        
                        weights = [bin_counts[0]] + [ bin_counts[i] if i else 0.0 for i in mask ]
                    
                        log_odds_tensors.append(new_model_)
                        bin_weights.append( weights)
            
                       
            # Adjusting zero weight values to one to avoid sum-to-zero error for weighted average
            if all([ w[0]==0 for w in bin_weights ])  :
                 for bw in bin_weights:
                     bw[0] = 1
            
            ebm.preprocessor_.col_bin_counts_[index] = np.round(np.average(bin_weights, axis=0))
                   
            averaged_model = np.average(log_odds_tensors, axis=0 , weights=bin_weights )
            model_errors = EBMUtils.weighted_std(np.array(log_odds_tensors), axis=0, weights= np.array(bin_weights) )

            ebm.additive_terms_.append(averaged_model)
            ebm.term_standard_deviations_.append(model_errors)           
            
        
        ebm.feature_importances_ = []        
        for i in range(len(ebm.feature_groups_)):

            mean_abs_score = np.average(np.abs(ebm.additive_terms_[i]), weights=ebm.preprocessor_.col_bin_counts_[i])

            ebm.feature_importances_.append(mean_abs_score)
           
        return ebm
    
    @staticmethod
    def normalize_initial_random_seed(seed):  # pragma: no cover
        # Some languages do not support 64-bit values.  Other languages do not support unsigned integers.
        # Almost all languages support signed 32-bit integers, so we standardize on that for our 
        # random number seed values.  If the caller passes us a number that doesn't fit into a 
        # 32-bit signed integer, we convert it.  This conversion doesn't need to generate completely 
        # uniform results provided they are reasonably uniform, since this is just the seed.
        # 
        # We use a simple conversion because we use the same method in multiple languages, 
        # and we need to keep the results identical between them, so simplicity is key.
        # 
        # The result of the modulo operator is not standardized accross languages for 
        # negative numbers, so take the negative before the modulo if the number is negative.
        # https://torstencurdt.com/tech/posts/modulo-of-negative-numbers

        if 2147483647 <= seed:
            return seed % 2147483647
        if seed <= -2147483647:
            return -((-seed) % 2147483647)
        return seed

    # NOTE: Interval / cut conversions are future work. Not registered for code coverage.
    @staticmethod
    def convert_to_intervals(cuts):  # pragma: no cover
        cuts = np.array(cuts, dtype=np.float64)

        if np.isnan(cuts).any():
            raise Exception("cuts cannot contain nan")

        if np.isinf(cuts).any():
            raise Exception("cuts cannot contain infinity")

        smaller = np.insert(cuts, 0, -np.inf)
        larger = np.append(cuts, np.inf)
        intervals = list(zip(smaller, larger))

        if any(x[1] <= x[0] for x in intervals):
            raise Exception("cuts must contain increasing values")

        return intervals

    @staticmethod
    def convert_to_cuts(intervals):  # pragma: no cover
        if len(intervals) == 0:
            raise Exception("intervals must have at least one interval")

        if any(len(x) != 2 for x in intervals):
            raise Exception("intervals must be a list of tuples")

        if intervals[0][0] != -np.inf:
            raise Exception("intervals must start from -inf")

        if intervals[-1][-1] != np.inf:
            raise Exception("intervals must end with inf")

        cuts = [x[0] for x in intervals[1:]]
        cuts_verify = [x[1] for x in intervals[:-1]]

        if np.isnan(cuts).any():
            raise Exception("intervals cannot contain NaN")

        if any(x[0] != x[1] for x in zip(cuts, cuts_verify)):
            raise Exception("intervals must contain adjacent sections")

        if any(higher <= lower for lower, higher in zip(cuts, cuts[1:])):
            raise Exception("intervals must contain increasing sections")

        return cuts

    @staticmethod
    def ebm_train_test_split(
        X, y, w, test_size, random_state, is_classification, is_train=True
    ):
        # all test/train splits should be done with this function to ensure that
        # if we re-generate the train/test splits that they are generated exactly
        # the same as before

        if (X.shape[0] != len(y) or X.shape[0] != len(w)):
            raise Exception("Data, labels and weights should have the same number of rows.")

        sampling_result = None

        if test_size == 0:
            X_train, y_train, w_train = X, y, w
            X_val = np.empty(shape=(0, X.shape[1]), dtype=X.dtype)
            y_val = np.empty(shape=(0,), dtype=y.dtype)
            w_val = np.empty(shape=(0,), dtype=w.dtype)

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

        if not is_train:
            X_train, y_train = None, None

        # TODO PK doing a fortran re-ordering here (and an extra copy) isn't the most efficient way
        #         push the re-ordering right to our first call to fit(..) AND stripe convert
        #         groups of rows at once and they process them in fortran order after that
        # change to Fortran ordering on our data, which is more efficient in terms of memory accesses
        # AND our C code expects it in that ordering
        if X_train is not None:
            X_train = np.ascontiguousarray(X_train.T)

        X_val = np.ascontiguousarray(X_val.T)

        return X_train, X_val, y_train, y_val, w_train, w_val

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
    def decision_function(X, X_pair, feature_groups, model, intercept):
        if X.ndim == 1:
            X = X.reshape(X.shape[0], 1)

        # Initialize empty vector for predictions
        if isinstance(intercept, numbers.Number) or len(intercept) == 1:
            score_vector = np.empty(X.shape[1])
        else:
            score_vector = np.empty((X.shape[1], len(intercept)))

        np.copyto(score_vector, intercept)

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
    def decision_function_and_explain(X, X_pair, feature_groups, model, intercept):
        if X.ndim == 1:
            X = X.reshape(X.shape[0], 1)

        # Initialize empty vector for predictions and explanations
        if isinstance(intercept, numbers.Number) or len(intercept) == 1:
            score_vector = np.empty(X.shape[1])
        else:
            score_vector = np.empty((X.shape[1], len(intercept)))

        np.copyto(score_vector, intercept)

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
    def classifier_predict_proba(X, X_pair, feature_groups, model, intercept):
        log_odds_vector = EBMUtils.decision_function(
            X, X_pair, feature_groups, model, intercept
        )

        # Handle binary classification case -- softmax only works with 0s appended
        if log_odds_vector.ndim == 1:
            log_odds_vector = np.c_[np.zeros(log_odds_vector.shape), log_odds_vector]

        return softmax(log_odds_vector)

    @staticmethod
    def classifier_predict(X, X_pair, feature_groups, model, intercept, classes):
        log_odds_vector = EBMUtils.decision_function(
            X, X_pair, feature_groups, model, intercept
        )
        if log_odds_vector.ndim == 1:
            log_odds_vector = np.c_[np.zeros(log_odds_vector.shape), log_odds_vector]

        return classes[np.argmax(log_odds_vector, axis=1)]

    @staticmethod
    def classifier_predict_and_contrib(X, X_pair, feature_groups, model, intercept, classes, output='probabilities'):
        scores_vector, explanations = EBMUtils.decision_function_and_explain(
            X,
            X_pair,
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
    def regressor_predict(X, X_pair, feature_groups, model, intercept):
        scores = EBMUtils.decision_function(X, X_pair, feature_groups, model, intercept)
        return scores

    @staticmethod
    def regressor_predict_and_contrib(X, X_pair, feature_groups, model, intercept):
        scores, explanations = EBMUtils.decision_function_and_explain(
            X,
            X_pair,
            feature_groups,
            model,
            intercept
        )
        return scores, explanations

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

                    if noise_scale: # Differentially private updates
                        splits = booster.get_model_update_splits()[0]

                        model_update_tensor = booster.get_model_update_expanded()
                        noisy_update_tensor = model_update_tensor.copy()

                        splits_iter = [0] + list(splits + 1) + [len(model_update_tensor)] # Make splits iteration friendly
                        # Loop through all random splits and add noise before updating
                        for f, s in zip(splits_iter[:-1], splits_iter[1:]):
                            if s == 1: 
                                continue # Skip cuts that fall on 0th (missing value) bin -- missing values not supported in DP

                            noise = np.random.normal(0.0, noise_scale)
                            noisy_update_tensor[f:s] = model_update_tensor[f:s] + noise

                            # Native code will be returning sums of residuals in slices, not averages.
                            # Compute noisy average by dividing noisy sum by noisy histogram counts
                            instance_count = np.sum(bin_counts[feature_group_index][f:s])
                            noisy_update_tensor[f:s] = noisy_update_tensor[f:s] / instance_count

                        noisy_update_tensor = noisy_update_tensor * -1 # Invert gradients before updates
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


class DPUtils:

    @staticmethod
    def calc_classic_noise_multi(total_queries, target_epsilon, delta, sensitivity):
        variance = (8*total_queries*sensitivity**2 * np.log(np.exp(1) + target_epsilon / delta)) / target_epsilon ** 2
        return np.sqrt(variance)

    @staticmethod
    def calc_gdp_noise_multi(total_queries, target_epsilon, delta):
        ''' GDP analysis following Algorithm 2 in: https://arxiv.org/abs/2106.09680. 
        '''
        def f(mu, eps, delta):
            return DPUtils.delta_eps_mu(eps, mu) - delta

        final_mu = brentq(lambda x: f(x, target_epsilon, delta), 1e-5, 1000)
        sigma = np.sqrt(total_queries) / final_mu
        return sigma

    # General calculations, largely borrowed from tensorflow/privacy and presented in https://arxiv.org/abs/1911.11607
    @staticmethod
    def delta_eps_mu(eps, mu):
        ''' Code adapted from: https://github.com/tensorflow/privacy/blob/master/tensorflow_privacy/privacy/analysis/gdp_accountant.py#L44
        '''
        return norm.cdf(-eps/mu + mu/2) - np.exp(eps) * norm.cdf(-eps/mu - mu/2)

    @staticmethod
    def eps_from_mu(mu, delta):
        ''' Code adapted from: https://github.com/tensorflow/privacy/blob/master/tensorflow_privacy/privacy/analysis/gdp_accountant.py#L50
        '''
        def f(x):
            return DPUtils.delta_eps_mu(x, mu)-delta    
        return root_scalar(f, bracket=[0, 500], method='brentq').root

    @staticmethod
    def private_numeric_binning(col_data, noise_scale, max_bins, min_val, max_val):
        uniform_counts, uniform_edges = np.histogram(col_data, bins=max_bins*2, range=(min_val, max_val))
        noisy_counts = uniform_counts + np.random.normal(0, noise_scale, size=uniform_counts.shape[0])
        
        # Postprocess to ensure realistic bin values (integers, min=1)
        noisy_counts = np.clip(np.round(noisy_counts), 1, None)

        # Greedily collapse bins until they meet or exceed target_count threshold
        target_count = col_data.shape[0] / max_bins
        bin_counts, bin_cuts = [0], [uniform_edges[0]]
        curr_count = 0
        for index, right_edge in enumerate(uniform_edges[1:]):
            curr_count += noisy_counts[index]
            if curr_count >= target_count:
                bin_cuts.append(right_edge)
                bin_counts.append(curr_count)
                curr_count = 0

        # Ignore min/max value as part of cut definition
        bin_cuts = np.array(bin_cuts)[1:-1] 

        # All leftover datapoints get collapsed into final bin
        bin_counts[-1] += curr_count

        return bin_cuts, bin_counts

    @staticmethod
    def private_categorical_binning(col_data, noise_scale, max_bins):
        # Initialize estimate
        col_data = col_data.astype('U')
        uniq_vals, counts = np.unique(col_data, return_counts=True)

        counts = counts + np.random.normal(0, noise_scale, size=counts.shape[0])
        counts = np.clip(np.round(counts), 1, None) # Postprocessing: Clip and round counts to be realistic (positive ints)

        # Collapse bins until target_size is achieved.
        target_count = col_data.shape[0] / max_bins
        small_bins = np.where(counts < target_count)[0]
        if len(small_bins) > 0:
            other_count = np.sum(counts[small_bins])
            mask = np.ones(counts.shape, dtype=bool)
            mask[small_bins] = False

            # Collapse all small bins into "DPOther"
            uniq_vals = np.append(uniq_vals[mask], "DPOther")
            counts = np.append(counts[mask], other_count)

            # # If "DPOther" bin is too small, absorb 1 more bin (guaranteed above threshold)
            if other_count < target_count:                               
                collapse_bin = np.argmin(counts[:-1])
                mask = np.ones(counts.shape, dtype=bool)
                mask[collapse_bin] = False

                # Pack data into the final "DPOther" bin
                counts[-1] += counts[collapse_bin]

                # Delete absorbed bin
                uniq_vals = uniq_vals[mask]
                counts = counts[mask]

        return uniq_vals, counts

    @staticmethod
    def build_privacy_schema(X, y=None):
        privacy_schema = {}
        for index in range(X.shape[1]):
            min_val, max_val = (np.min(X[:, index]), np.max(X[:, index]))
            if isinstance(min_val, numbers.Number) and isinstance(max_val, numbers.Number):
                privacy_schema[index] = (min_val, max_val)

        if y is not None:
            privacy_schema['target'] = (np.min(y), np.max(y))

        return privacy_schema

    @staticmethod
    def validate_eps_delta(eps, delta):
        if eps is None or eps <= 0 or delta is None or delta <= 0:
            raise ValueError(f"Epsilon: '{eps}' and delta: '{delta}' must be set to positive numbers")

    @staticmethod
    def validate_DP_EBM(ebm):
        fixed_params = dict(
            max_interaction_bins=None,
            interactions=0,
            inner_bags=0,
            early_stopping_rounds=-1,
            early_stopping_tolerance=-1,
        )

        error_params = []
        for param in fixed_params:
            if getattr(ebm, param) != fixed_params[param]:
                error_params.append(param)

        if len(error_params) > 0:
            raise ValueError(f"The following parameters: {error_params} cannot be changed when training with differential privacy.")
