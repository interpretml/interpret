# Copyright (c) 2019 Microsoft Corporation

# Distributed under the MIT software license
# TODO: Test EBMUtils

# from scipy.special import expit
from sklearn.utils.extmath import softmax
from sklearn.model_selection import train_test_split
from sklearn.base import is_classifier
import numbers
import numpy as np
import warnings
import copy


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

        if len(models) < 2:
            raise Exception("at least two models are required to merge.")
            return

        # many features are invalid. preprocessor_ and pair_preprocessor_ are cloned form the first model.
        ebm = copy.deepcopy(models[0]) 
        

        ebm.additive_terms_ =[]
        ebm.term_standard_deviations_ = []
        ebm.bagged_models_=[]
        ebm.pair_preprocessor_ = None       
      
         
        if not all([  model.preprocessor_.col_types_ == ebm.preprocessor_.col_types_ for model in models]): # pragma: no cover
            raise Exception("All models should have the same types of features.")
       
        if not all([  model.preprocessor_.col_bin_edges_.keys() == ebm.preprocessor_.col_bin_edges_.keys() for model in models]): # pragma: no cover
                    raise Exception("All models should have the same types of numeric features.")


        if not all([  model.preprocessor_.col_mapping_.keys() == ebm.preprocessor_.col_mapping_.keys() for model in models]): # pragma: no cover
                    raise Exception("All models should have the same types of categorical features.")

        if is_classifier(ebm): # pragma: no cover
                if not all([is_classifier(model) for model in models]):
                    raise Exception("All models should be the same type.")
        else: # pragma: no cover
                #if ebm is a regressor
                if any([is_classifier(model) for model in models]):
                    raise Exception("All models should be the same type.")

        new_feature_groups = []
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
                model_bin_counts = []
                for model in models:            
                    # Merging the bin edges for different models for each feature group
                    bin_edges = model.preprocessor_.col_bin_edges_[index]
                    bin_counts = model.preprocessor_.col_bin_counts_[index]

                    #bin_idx contains the index of each new merged bin edges against the existing bin edges
                    bin_idx = np.searchsorted(bin_edges, merged_bin_edges + [np.inf])   
                    # the first element of adjusted bin indexes is used for weighted average of misssing values.
                    adj_bin_idx = [0] + ( [ x+1 for x in bin_idx ])
                    
                    # new_bin_count_ =  [ bin_counts[x] for x in adj_bin_idx ]  
                    
                    # model_bin_counts.append(new_bin_count_) 
                                        
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
        X, y, test_size, random_state, is_classification, is_train=True
    ):
        # all test/train splits should be done with this function to ensure that
        # if we re-generate the train/test splits that they are generated exactly
        # the same as before
        if test_size == 0:
            X_train, y_train = X, y
            X_val = np.empty(shape=(0, X.shape[1]), dtype=X.dtype)
            y_val = np.empty(shape=(0,), dtype=y.dtype)
        elif test_size > 0:
            # Adapt test size if too small relative to number of classes
            if is_classification:
                y_uniq = len(set(y))
                n_test_samples = test_size if test_size >= 1 else len(y) * test_size
                if n_test_samples < y_uniq:  # pragma: no cover
                    warnings.warn(
                        "Too few samples per class, adapting test size to guarantee 1 sample per class."
                    )
                    test_size = y_uniq

            # PaulK NOTE: sklearn train_test_split doesn't accept negative random_states
            # we can remove the conversion to just positive values when we transition to C++
            X_train, X_val, y_train, y_val = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=(random_state - (-2147483648)) if random_state < 0 else random_state,
                stratify=y if is_classification else None,
            )
        else:  # pragma: no cover
            raise Exception("test_size must be between 0 and 1.")

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

        return X_train, X_val, y_train, y_val

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
    def regressor_predict(X, X_pair, feature_groups, model, intercept):
        scores = EBMUtils.decision_function(X, X_pair, feature_groups, model, intercept)
        return scores

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
