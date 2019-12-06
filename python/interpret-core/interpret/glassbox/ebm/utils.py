# Copyright (c) 2019 Microsoft Corporation

# Distributed under the MIT software license
# TODO: Test EBMUtils

# from scipy.special import expit
from sklearn.utils.extmath import softmax
from sklearn.model_selection import train_test_split
import numbers
import numpy as np


import logging

log = logging.getLogger(__name__)


# TODO: Clean up
class EBMUtils:
    @staticmethod
    def get_count_scores_c(n_classes):
        # this should reflect how the C code represents scores
        return 1 if n_classes <= 2 else n_classes

    @staticmethod
    def ebm_train_test_split(X, y, test_size, random_state, is_classification, is_train=True):
        # TODO PK Implement the following for memory efficiency and speed of initialization:
        #   - NOTE: FOR RawArray ->  import multiprocessing ++ from multiprocessing import RawArray ++ RawArray(ct.c_ubyte, memory_size) ++ ct.POINTER(ct.c_ubyte)
        #   - OBSERVATION: We want sparse feature support in our booster since we don't need to access 
        #                  memory if there are long segments with a single value
        #   - OBSERVATION: Sorting a dataset with sparse features will lead to unpredictably sized final memory sizes, 
        #                  since more clumped data will be more compressed
        #   - OBSERVATION: for interactions, from a CPU access point of view, we want all of our features to have the 
        #                  same # of bits so that we can have one loop compare any tuple of features.  
        #                  We therefore do NOT want sparse feature support when looking for interactions
        #   - OBSERVATION: sorting will be easier for non-sparse data, and we'll want non-sparse data for interactions anyways, 
        #                  so we should only do sparseness for our boosting dataset allocation
        #   - OBSERVATION: without sparse memory in the initial shared memory object, we can calculate the size without seeing the data.
        #                  Even if we had sorted sparse features, we'd only find out the memory size after the sort, 
        #                  so we'd want dynamically allocated memory during the sort
        #   - OBSERVATION: for boosting, we can compress memory to the right size per feature_combination, 
        #                  but for interactions, we want to compress all features by the same amount
        #                  (all features use the same number of bits) so that we can compare any two/three/etc 
        #                  features and loop at the same points for each
        # STEPS:
        #   - We receive the data from the user in the cache inefficient format X[instances, features]
        #   - Do preprocessing so that we know how many bins each feature has 
        #     (we might want to process X[instances, features] in chunks, like below to do this)
        #   - call into C to get back the exact size of the memory object that we need in order to store all the data.  
        #     We can do this because we won't store any of the data at this point as sparse
        #   - Allocate the buffer in python using RawArray (RawArray will be shared with other processes as read only data)
        #   - Divide the features into M chunks of N features.  Let's choose M to be 32, so that we don't increase memory usage by more than 3%
        #   - Loop over M:
        #     - Take N features and all the instances from the original X and transpose them into X_partial[features_N, instances]
        #     - Loop over N:
        #       - take 1 feature and pass it into C for bit compression (don't use sparse coding here) into the RawArray
        #   - NOTE: this transposes the matrix twice (once for preprocessing and once for adding to C), 
        #     but this is expected to be a small amount of time compared to training, and we care more about memory size at this point
        #   - Call a C function which will finalize the dataset (this function will accept the target array).  
        #     - The C function will create an index array and add this index to the dataset (it will be shared)
        #     - sort the index array by the target first, then the features with the highest counts of the mode value
        #     - sort the underlying data by the index array
        #   - Now the memory is read only from now on, and shareable.  Include a reverse index in the data for reconstructing the
        #     original order inside the data structure.  
        #   - No pointers in the data structure, just offsets (for sharing cross process)!
        #   - Start each child processes, and pass them our shared memory structure 
        #     (it will be mapped into each process address space, but not copied)
        #   - each child calls a train/validation splitter provided by our C that fills a numpy array of bools
        #     We do this in C instead of using the sklearn train_test_split because sklearn would require us to first split sequential indexes, 
        #     possibly sort them (if order in not guaranteed), then convert to bools in a caching inefficient way, 
        #     whereas in C we can do a single pass without any memory array inputs (using just a random number generator) 
        #     and we can make the outputs consistent across languages.
        #   - with the RawArray complete data PLUS the train/validation bool list we can generate either interaction datasets OR boosting dataset as needed.
        #     We can reduce our memory footprint, by never having both an interaction AND boosting dataset in memory at the same time.
        #   - first generate the mains train/validation boosting datasets, then create the interaction sets, then create the pair boosting datasets
        #   - FOR BOOSTING:
        #     - Pass the train/validation bool list into C AND the RawArray AND the feature_combination definitions.
        #     - C takes the bool list, then uses the mapping indexes in the RawArray dataset to reverse the bool index into our internal C sorted order.
        #       This way we only need to do a cache inefficient reordering once per entire dataset, and it's on a bool array (compressed to bits?)
        #     - C will do a first pass to determine how much memory it will need (sparse features are variable sized) [we have all the data to do this!]
        #     - C will allocate the memory for the boosting dataset
        #     - C will do a second pass to fill the boosting data structure and return that to python
        #     - After re-ordering the bool lists to the original feature order, we process each feature using the bool to do a non-branching if statements to select whether each instance for that feature goes into the train or validation set, and handling increments
        #   - FOR INTERACTIONS:
        #     - pass into C the train/validation sets and the RawArray
        #     - C will compute the amount of memory needed, and allocate that (our interaction data is NOT sparse, so we can compute)
        #     - turn the RawArray data into the new memory non-compressed format for interactions, and return it to python

        # all test/train splits should be done with this function to ensure that
        # if we re-generate the train/test splits that they are generated exactly
        # the same as before
        if test_size > 0:
            if is_train:
                X_train, X_val, y_train, y_val = train_test_split(
                    X,
                    y,
                    test_size=test_size,
                    random_state=random_state,
                    stratify=y if is_classification else None,
                )
            else:
                X_train = None
                y_train = None
                _, X_val, _, y_val = train_test_split(
                    X,
                    y,
                    test_size=test_size,
                    random_state=random_state,
                    stratify=y if is_classification else None,
                )
        elif test_size == 0:
            if is_train:
                X_train = X
                y_train = y
            else:
                X_train = None
                y_train = None

            X_val = np.empty(shape=(0, X.shape[1]), dtype=X.dtype)
            y_val = np.empty(shape=(0), dtype=y.dtype)
        else:  # pragma: no cover
            raise Exception("test_size must be between 0 and 1.")

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
    def gen_features(col_types, col_n_bins):
        # Create Python form of features
        # Undocumented.
        features = [None] * len(col_types)
        for col_idx, _ in enumerate(features):
            features[col_idx] = {
                # NOTE: Ordinal only handled at native, override.
                # 'type': col_types[col_idx],
                "type": "continuous",
                # NOTE: Missing not implemented at native, always set to false.
                "has_missing": False,
                "n_bins": col_n_bins[col_idx],
            }
        return features

    @staticmethod
    def gen_feature_combinations(feature_indices):
        feature_combinations = [None] * len(feature_indices)
        for i, indices in enumerate(feature_indices):
            # TODO PK v.2 remove n_attributes (this is the only place it is used, but it's public)
            # TODO PK v.2 rename all instances of "attributes" -> "features"
            feature_combination = {"n_attributes": len(indices), "attributes": indices}
            feature_combinations[i] = feature_combination
        return feature_combinations

    @staticmethod
    def scores_by_feature_combination(
        X, feature_combinations, model
    ):
        for set_idx, feature_combination in enumerate(feature_combinations):
            tensor = model[set_idx]

            # Get the current column(s) to process
            feature_idxs = feature_combination["attributes"]
            feature_idxs = list(reversed(feature_idxs))
            sliced_X = X[feature_idxs, :]
            scores = tensor[tuple(sliced_X)]

            yield set_idx, feature_combination, scores

    @staticmethod
    def decision_function(
        X, feature_combinations, model, intercept
    ):
        if X.ndim == 1:
            X = X.reshape(X.shape[0], 1)

        # Initialize empty vector for predictions
        if isinstance(intercept, numbers.Number) or len(intercept) == 1:
            score_vector = np.empty(X.shape[1])
        else:
            score_vector = np.empty((X.shape[1], len(intercept)))

        np.copyto(score_vector, intercept)

        scores_gen = EBMUtils.scores_by_feature_combination(
            X, feature_combinations, model
        )
        for _, _, scores in scores_gen:
            score_vector += scores

        if not np.all(np.isfinite(score_vector)):  # pragma: no cover
            msg = "Non-finite values present in log odds vector."
            log.error(msg)
            raise Exception(msg)

        return score_vector

    @staticmethod
    def classifier_predict_proba(X, feature_combinations, model, intercept):
        log_odds_vector = EBMUtils.decision_function(
            X,
            feature_combinations,
            model,
            intercept
        )

        # Handle binary classification case -- softmax only works with 0s appended
        if log_odds_vector.ndim == 1:
            log_odds_vector = np.c_[np.zeros(log_odds_vector.shape), log_odds_vector]

        return softmax(log_odds_vector)

    @staticmethod
    def classifier_predict(X, feature_combinations, model, intercept, classes):
        log_odds_vector = EBMUtils.decision_function(
            X,
            feature_combinations,
            model,
            intercept
        )
        if log_odds_vector.ndim == 1:
            log_odds_vector = np.c_[np.zeros(log_odds_vector.shape), log_odds_vector]

        return classes[np.argmax(log_odds_vector, axis=1)]

    @staticmethod
    def regressor_predict(X, feature_combinations, model, intercept):
        scores = EBMUtils.decision_function(
            X,
            feature_combinations,
            model,
            intercept
        )
        return scores

    @staticmethod
    def gen_feature_name(feature_idxs, col_names):
        feature_name = []
        for feature_index in feature_idxs:
            col_name = col_names[feature_index]
            feature_name.append("feature_" + str(col_name) if isinstance(col_name, int) else str(col_name))
        feature_name = " x ".join(feature_name)
        return feature_name

    @staticmethod
    def gen_feature_type(feature_idxs, col_types):
        if len(feature_idxs) == 1:
            return col_types[feature_idxs[0]]
        else:
            # TODO PK we should consider changing the feature type to the same " x " separator
            # style as gen_feature_name, for human understanability
            return "pairwise"
