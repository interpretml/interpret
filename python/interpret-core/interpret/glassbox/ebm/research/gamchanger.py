import numpy as np
import pandas as pd

from copy import deepcopy
from json import dump

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

            # Get density info
            if ebm.preprocessor_._get_hist_edges(cur_id[0])[0].dtype.type is np.str_:
                cur_feature['histEdge1'] = ebm.preprocessor_._get_hist_edges(cur_id[0])
            else:
                cur_feature['histEdge1'] = np.round(
                    ebm.preprocessor_._get_hist_edges(cur_id[0]), 4
                ).tolist()
            cur_feature['histCount1'] = np.round(
                ebm.preprocessor_._get_hist_counts(cur_id[0]), 4
            ).tolist()

            if ebm.preprocessor_._get_hist_edges(cur_id[1])[0].dtype.type is np.str_:
                cur_feature['histEdge2'] = ebm.preprocessor_._get_hist_edges(cur_id[1])
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
        'intercept': ebm.intercept_[0],
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
                    return max(level_str_to_int.values())

            x_test_copy[:, i] = list(
                map(lambda x: get_level_int(x), x_test_copy[:, i]))

    sample_data = {
        'featureNames': feature_names,
        'featureTypes': feature_types,
        'samples': x_test_copy.tolist(),
        'labels': y_test_copy.tolist()
    }

    return sample_data
