import numpy as np
import pandas as pd

from copy import deepcopy
from json import dump, load

def get_model_data(ebm):
    """
    Get the model data for GAM Changer.
    Args:
        ebm: Trained EBM model. ExplainableBoostingClassifier or
            ExplainableBoostingRegressor object.
    Returns:
        A Python dictionary of model data
    """

    # Main model info on each feature
    features = []

    # Track the encoding of categorical feature levels
    labelEncoder = {}

    # Track the score range
    score_range = [np.inf, -np.inf]

    for i in range(len(ebm.feature_names)):
        cur_feature = {}
        cur_feature['name'] = ebm.feature_names[i]
        cur_feature['type'] = ebm.feature_types[i]
        cur_feature['importance'] = ebm.feature_importances_[i]

        # Handle interaction term differently from cont/cat
        if cur_feature['type'] == 'interaction':
            cur_id = ebm.feature_groups_[i]
            cur_feature['id'] = list(cur_id)

            # Info for each individual feature
            cur_feature['name1'] = ebm.feature_names[cur_id[0]]
            cur_feature['name2'] = ebm.feature_names[cur_id[1]]

            cur_feature['type1'] = ebm.feature_types[cur_id[0]]
            cur_feature['type2'] = ebm.feature_types[cur_id[1]]

            # Skip the first item from both dimensions
            cur_feature['additive'] = np.round(ebm.additive_terms_[i], 4)\
                [1:, 1:].tolist()
            cur_feature['error'] = np.round(ebm.term_standard_deviations_[i], 4)\
                [1:, 1:].tolist()

            # Get the bin label info
            cur_feature['binLabel1'] = ebm.pair_preprocessor_._get_bin_labels(cur_id[0])
            cur_feature['binLabel2'] = ebm.pair_preprocessor_._get_bin_labels(cur_id[1])

            # Encode categorical levels as integers
            if cur_feature['type1'] == 'categorical':
                level_str_to_int = ebm.pair_preprocessor_.col_mapping_[
                    cur_id[0]]
                cur_feature['binLabel1'] = list(map(lambda x: level_str_to_int[x],
                                                    cur_feature['binLabel1']))

            if cur_feature['type2'] == 'categorical':
                level_str_to_int = ebm.pair_preprocessor_.col_mapping_[
                    cur_id[1]]
                cur_feature['binLabel2'] = list(map(lambda x: level_str_to_int[x],
                                                    cur_feature['binLabel2']))

            # Get density info
            if cur_feature['type1'] == 'categorical':
                level_str_to_int = ebm.pair_preprocessor_.col_mapping_[cur_id[0]]
                cur_feature['histEdge1'] = ebm.preprocessor_._get_hist_edges(cur_id[0])
                cur_feature['histEdge1'] = list(map(lambda x: level_str_to_int[x],
                                                    cur_feature['histEdge1']))
            else:
                cur_feature['histEdge1'] = np.round(
                    ebm.preprocessor_._get_hist_edges(cur_id[0]), 4
                ).tolist()
            cur_feature['histCount1'] = np.round(
                ebm.preprocessor_._get_hist_counts(cur_id[0]), 4
            ).tolist()

            if cur_feature['type2'] == 'categorical':
                level_str_to_int = ebm.pair_preprocessor_.col_mapping_[cur_id[1]]
                cur_feature['histEdge2'] = ebm.preprocessor_._get_hist_edges(cur_id[1])
                cur_feature['histEdge2'] = list(map(lambda x: level_str_to_int[x],
                                                    cur_feature['histEdge2']))
            else:
                cur_feature['histEdge2'] = np.round(
                    ebm.preprocessor_._get_hist_edges(cur_id[1]), 4
                ).tolist()
            cur_feature['histCount2'] = np.round(
                ebm.preprocessor_._get_hist_counts(cur_id[1]), 4
            ).tolist()

        else:
            # Skip the first item (reserved for missing value)
            cur_feature['additive'] = np.round(ebm.additive_terms_[i], 4).tolist()[1:]
            cur_feature['error'] = np.round(ebm.term_standard_deviations_[i], 4).tolist()[1:]
            cur_feature['id'] = ebm.feature_groups_[i]
            cur_id = ebm.feature_groups_[i][0]
            cur_feature['count'] = ebm.preprocessor_.col_bin_counts_[cur_id].tolist()[1:]

            # Track the global score range
            score_range[0] = min(score_range[0],
                np.min(ebm.additive_terms_[i] - ebm.term_standard_deviations_[i]))
            score_range[1] = max(score_range[1],
                np.max(ebm.additive_terms_[i] + ebm.term_standard_deviations_[i]))

            # Add the binning information for continuous features
            if cur_feature['type'] == 'continuous':
                # Add the bin information
                cur_feature['binEdge'] = np.round(
                    ebm.preprocessor_._get_bin_labels(cur_id), 4
                ).tolist()

                # Add the hist information
                cur_feature['histEdge'] = np.round(
                    ebm.preprocessor_._get_hist_edges(cur_id), 4
                ).tolist()
                cur_feature['histCount'] = np.round(
                    ebm.preprocessor_._get_hist_counts(cur_id), 4
                ).tolist()

            elif cur_feature['type'] == 'categorical':
                # Get the level value mapping
                level_str_to_int = ebm.preprocessor_.col_mapping_[cur_id]
                cur_feature['binLabel'] = list(map(lambda x: level_str_to_int[x],
                                               ebm.preprocessor_._get_bin_labels(cur_id)))

                # Add the hist information
                # For categorical data, the edges are strings
                cur_feature['histEdge'] = list(map(lambda x: level_str_to_int[x],
                                               ebm.preprocessor_._get_hist_edges(cur_id)))

                cur_feature['histCount'] = np.round(
                    ebm.preprocessor_._get_hist_counts(cur_id), 4
                ).tolist()

                # Add the label encoding information
                labelEncoder[cur_feature['name']] = {i: s for s, i in level_str_to_int.items()}

        features.append(cur_feature)

    score_range = list(map(lambda x: round(x, 4), score_range))

    data = {
        'intercept': ebm.intercept_[0] if hasattr(ebm, 'classes_') else ebm.intercept_,
        'isClassifier': hasattr(ebm, 'classes_'),
        'features': features,
        'labelEncoder': labelEncoder,
        'scoreRange': score_range
    }
    
    return data


def get_sample_data(ebm, x_test, y_test):
    """
    Get the sample data for GAM Changer.
    Args:
        ebm: Trained EBM model. ExplainableBoostingClassifier or
            ExplainableBoostingRegressor object.
        x_test: Sample features. 2D np.ndarray or pd.DataFrame with dimension [n, k]:
            n samples and k features.
        x_test: Sample labels. 1D np.ndarray or pd.Series with size = n samples.
    Returns:
        A Python dictionary of sample data.
    """

    assert(isinstance(x_test, (pd.DataFrame, np.ndarray)))
    assert(isinstance(y_test, (pd.Series, np.ndarray)))

    feature_names = []
    feature_types = []

    # Sample data does not record interaction features
    for i in range(len(ebm.feature_names)):
        if (ebm.feature_types[i] != 'interaction'):
            feature_names.append(ebm.feature_names[i])
            feature_types.append(ebm.feature_types[i])

    # Transform the dataframe to object array
    x_test_copy = deepcopy(x_test)
    y_test_copy = deepcopy(y_test)

    if isinstance(x_test, pd.DataFrame):
        x_test_copy = x_test.to_numpy()

    if isinstance(y_test, pd.Series):
        y_test_copy = y_test.to_numpy()

    # Encode the categorical variables as integers
    for i in range(len(feature_types)):
        if (feature_types[i] == 'categorical'):
            level_str_to_int = ebm.preprocessor_.col_mapping_[i]

            def get_level_int(x):

                if str(x) in level_str_to_int:
                    return level_str_to_int[str(x)]
                else:
                    # Current sample has an unseen level, we label it as max
                    # level + 1
                    return max(level_str_to_int.values()) + 1

            x_test_copy[:, i] = list(
                map(lambda x: get_level_int(x), x_test_copy[:, i]))

    sample_data = {
        'featureNames': feature_names,
        'featureTypes': feature_types,
        'samples': x_test_copy.tolist(),
        'labels': y_test_copy.tolist()
    }

    return sample_data


def overwrite_bin_definition(ebm, index_id, new_bins, new_scores):
    """
    Overwrite the bin definitions and scores for continuous variables.
    
    Args:
        ebm: EBM object
        index_id: Feature's index id in the ebm object
        new_bins: New bin definition
        new_score: New bin scores
        
    In python, to overwrite the bins, we want to overwrite pair
    `edge[:] with score[2:]` and pair `col_min_ with score [1]`.

    In GAM Changer and EBM.JS, stored bins are `python_label[:-1]` and `python_score[1:]`

    To map GAM Changer and EBM.JS's `newBins`, `newScores` back to Python:

    ```
    newBins[0] => col_min_
    newBins[1:] => col_bin_edges_

    newScores[:] => additive_terms_[1:]
    ```
    
    We also want to update the standard deviation information:
    
    Case 1: Bin definition has not changed:
        We zero out the SDs of bins that have been modified
    
    Case 2: Bin definition has changed (even just a subset):
        We zero out all the SDs of bins
        
    In Python, SDs share the same index as scores.
    """

    assert(len(new_bins) == len(new_scores))

    # Check if GAM Changer has changed the bin definition
    binDefChanged = False

    if len(new_bins) - 1 != len(ebm.preprocessor_.col_bin_edges_[index_id]):
        binDefChanged = True

    else:
        for i in range(1, len(new_bins)):
            if new_bins[i] != round(ebm.preprocessor_.col_bin_edges_[index_id][i - 1], 4):
                binDefChanged = True
                break

    # Update the SDs
    if binDefChanged:
        ebm.term_standard_deviations_[index_id] = np.zeros(len(new_scores) + 1)
    else:
        # Itereate through the scores to zero out SDs of modified bins
        for i in range(1, len(ebm.additive_terms_[index_id])):
            if round(ebm.additive_terms_[index_id][i], 4) != new_scores[i - 1]:
                ebm.term_standard_deviations_[index_id][i] = 0

    # Overwrite the scores
    ebm.additive_terms_[index_id] = np.array(
        [ebm.additive_terms_[index_id][0]] + new_scores
    ).astype(np.float64)

    # Overwrite the bin edges

    # GAM Changer won't change the edge for col_min_, because it
    # will always be one of the end points in any interpolations
    # So we don't really need to change col_min_, change here for testing purpose
    ebm.preprocessor_.col_min_[index_id] = new_bins[0]
    ebm.preprocessor_.col_bin_edges_[index_id] = np.array(
        new_bins[1:]).astype(np.float64)



def get_edited_model(ebm, gamchanger_export):
    """
    Return a copy of ebm that is modified based on the edits from GAM Changer.

    Args:
        ebm: EBM object
        gamchanger_export: Python dictionary: loaded from the GAM Changer
            export (*.gamchanger)

    Returns:
        An edited deep copy of ebm object.
    """

    ebm_copy = deepcopy(ebm)

    history = gamchanger_export['historyList']

    # Mapping from feature name to feature type
    feature_name_to_type = dict(zip(ebm_copy.feature_names, ebm_copy.feature_types))

    # Keep track which feature has been updated in ebm_copy
    updated_features = set()

    # We iterate through the history list from the newest edit to the oldes edit
    # For each modified feature, we overwrite the bin definitions/scores on an EBM
    # copy using the latest edit info on that feature.
    # Note that GAM Changer can only change the bin definitions of continuous features

    for i in range(len(history) - 1, -1, -1):
        cur_history = history[i]

        # Original edit does not change the graph
        if cur_history['type'] == 'original':
            continue

        cur_name = cur_history['featureName']
        cur_index = ebm_copy.feature_names.index(cur_name)

        # If we have already updated EBM on this feature, skip earlier edits
        if cur_name in updated_features:
            continue

        if feature_name_to_type[cur_name] == 'continuous':
            # Collect bin edges and scores
            bin_data = cur_history['state']['pointData']
            bin_edges, bin_scores = [], []

            # bin_data is a linked list, bin_data[0] is gauranteed to be the start
            # point of all bins
            cur_bin = bin_data['0']

            while cur_bin['rightPointID']:
                bin_edges.append(cur_bin['x'])
                bin_scores.append(cur_bin['y'])
                cur_bin = bin_data[str(cur_bin['rightPointID'])]

            # Handle the last bin
            bin_edges.append(cur_bin['x'])
            bin_scores.append(cur_bin['y'])

            assert(len(bin_edges) == len(bin_data))

            # Overwrite EBM bin defintions/additive terms with bin_edges and bin_scores
            overwrite_bin_definition(ebm_copy, cur_index, bin_edges, bin_scores)
            updated_features.add(cur_name)

        elif feature_name_to_type[cur_name] == 'categorical':
            pass
        elif feature_name_to_type[cur_name] == 'interaction':
            pass
        else:
            raise ValueError('Encounter unknown feature type {}'.format(feature_name_to_type[cur_name]))

    return ebm_copy
