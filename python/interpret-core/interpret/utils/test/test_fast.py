from more_itertools import pairwise
import pytest
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression

from ...test.utils import synthetic_regression
from ..fast import fast, _get_scores

@pytest.fixture(scope="module")
def regression_data():
    data = synthetic_regression()
    return data["full"]["X"], data["full"]["y"]

def test_get_model_scores_if_provided(regression_data):
    X, y = regression_data
    init_scores = np.random.rand(X.shape[0])

    lr = LinearRegression()
    lr.fit(X, y)

    # squeeze() removes axes of length one
    predictions = lr.predict(X).squeeze()
    scores = _get_scores(X, init_scores, lr)

    assert np.array_equal(scores, predictions)

def test_get_init_scores_if_no_model(regression_data):
    X, _ = regression_data

    init_scores = np.random.rand(X.shape[0])
    scores = _get_scores(X, init_scores, init_model=None)
    assert np.array_equal(scores, init_scores)

    init_scores = None
    scores = _get_scores(X, init_scores, init_model=None)
    assert scores is None

def test_kwargs(regression_data):
    X, y = regression_data
    init_scores = np.random.rand(X.shape[0])

    lr = LinearRegression()
    lr.fit(X, y)

    ranked_pairs_dict = fast(X, y, is_classification=False, init_model=lr)
    # 4 features
    assert 6 == len(ranked_pairs_dict)

    ranked_pairs_dict = fast(X, y, is_classification=False, init_scores=init_scores)
    assert 6 == len(ranked_pairs_dict)

def test_missing_is_classification(regression_data):
    X, y = regression_data

    with pytest.raises(ValueError):
        fast(X, y, is_classification=None)

def test_inconsistent_is_classification(regression_data):
    X, y = regression_data

    lr = LinearRegression()

    with pytest.raises(ValueError):
        fast(X, y, init_model=lr, is_classification=True)

def test_inconsistent_X_and_y(regression_data):
    X, y = regression_data
    y = y.head(10)

    with pytest.raises(ValueError):
        fast(X, y, is_classification=False)

def test_inconsistent_sample_weigth(regression_data):
    X, y = regression_data
    sample_weight = np.random.rand(X.shape[0] - 10)

    with pytest.raises(ValueError):
        fast(X, y, is_classification=False, sample_weight=sample_weight)

def test_sample_weigth(regression_data):
    X, y = regression_data
    sample_weight = np.random.rand(X.shape[0])

    ranked_pairs_dict = fast(X, y, is_classification=False, sample_weight=sample_weight)
    # 4 features
    assert 6 == len(ranked_pairs_dict)

def test_feature_names_and_types(regression_data):
    X, y = regression_data
    
    # Original feature names "A", "B", "C", "D"
    feature_names = ["FtA", "FtB", "FtC", "FtD"]
    feature_types = ["continuous", "continuous", "continuous", "continuous"]

    ranked_pairs_dict = fast(X, y, is_classification=False, feature_names=feature_names, feature_types=feature_types)
    assert 6 == len(ranked_pairs_dict)

def test_max_bins_and_binning_options(regression_data):
    X, y = regression_data
    
    max_interaction_bins = 64
    binning = "uniform"

    ranked_pairs_dict = fast(X, y, is_classification=False, max_interaction_bins=max_interaction_bins, binning=binning)
    assert 6 == len(ranked_pairs_dict)

    binning = "rounded_quantile"

    ranked_pairs_dict = fast(X, y, is_classification=False, max_interaction_bins=max_interaction_bins, binning=binning)
    assert 6 == len(ranked_pairs_dict)

def test_min_samples_leaf(regression_data):
    X, y = regression_data

    min_samples_leaf = 4

    ranked_pairs_dict = fast(X, y, is_classification=False, min_samples_leaf=min_samples_leaf)
    assert 6 == len(ranked_pairs_dict)

def test_num_output_interactions(regression_data):
    X, y = regression_data

    ranked_pairs_dict = fast(X, y, is_classification=False, num_output_interactions=3)
    assert 3 == len(ranked_pairs_dict)

    ranked_pairs_dict = fast(X, y, is_classification=False, num_output_interactions=100)
    assert 6 == len(ranked_pairs_dict)

    ranked_pairs_dict = fast(X, y, is_classification=False, num_output_interactions=0)
    assert 6 == len(ranked_pairs_dict)

    ranked_pairs_dict = fast(X, y, is_classification=False, num_output_interactions=-2)
    assert 6 == len(ranked_pairs_dict)

def test_output_dict(regression_data):
    X, y = regression_data

    ranked_pairs_dict = fast(X, y, is_classification=False)
    assert 6 == len(ranked_pairs_dict)
    for key, value in ranked_pairs_dict.items():
        assert isinstance(key, tuple)
        assert isinstance(value, float)

def test_regression_task():
    from sklearn.datasets import load_diabetes
    diabetes_data = load_diabetes(return_X_y=True)
    X = diabetes_data[0]
    y = diabetes_data[1]

    ranked_strengths = fast(X, y, is_classification=False)

    # 10 features
    assert 45 == len(ranked_strengths)

def test_classification_task():
    from sklearn.datasets import load_breast_cancer
    breast_cancer_data = load_breast_cancer(return_X_y=True)
    X = breast_cancer_data[0]
    y = breast_cancer_data[1]

    ranked_strengths = fast(X, y, is_classification=True)

    # 30 features
    assert 435 == len(ranked_strengths)

def test_nulticlass_task():
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=100, n_features=10, n_informative=3, n_classes=3, random_state=2022)

    ranked_strengths = fast(X, y, is_classification=True)

    assert 45 == len(ranked_strengths)

#
# Manual tests
#

@pytest.fixture(scope="module")
def manual_test_regression_data():
    from sklearn.datasets import load_diabetes
    diabetes_data = load_diabetes(return_X_y=True)
    return diabetes_data[0], diabetes_data[1]

@pytest.fixture(scope="module")
def manual_test_classification_data():
    df = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
        header=None)
    df.columns = [
        "Age", "WorkClass", "fnlwgt", "Education", "EducationNum",
        "MaritalStatus", "Occupation", "Relationship", "Race", "Gender",
        "CapitalGain", "CapitalLoss", "HoursPerWeek", "NativeCountry", "Income"
    ]
    train_cols = df.columns[0:-1]
    label = df.columns[-1]
    X_df = df[train_cols]
    y_df = df[label]

    return X_df, y_df

def test_case_1(manual_test_regression_data):
    X, y = manual_test_regression_data

    from ...glassbox import ExplainableBoostingRegressor
    ebm = ExplainableBoostingRegressor(random_state=2022)
    #ebm = ExplainableBoostingRegressor(outer_bags=1, random_state=2022)
    # Take notes on interactions found
    ebm.fit(X, y)

    ranked_strengths = fast(X, y, is_classification=False)

    # 10 features
    assert 45 == len(ranked_strengths)

def test_case_1_classification(manual_test_classification_data):
    X, y = manual_test_classification_data

    from ...glassbox import ExplainableBoostingClassifier
    ebm = ExplainableBoostingClassifier(random_state=2022)
    #ebm = ExplainableBoostingRegressor(outer_bags=1, random_state=2022)
    # Take notes on interactions found
    ebm.fit(X, y)

    ranked_strengths = fast(X, y, is_classification=True)

    # 14 features
    assert 91 == len(ranked_strengths)

def test_case_2(manual_test_regression_data):
    X, y = manual_test_regression_data

    from ...glassbox import ExplainableBoostingRegressor
    ebm = ExplainableBoostingRegressor(random_state=2022, interactions=0)
    ebm.fit(X, y)

    ranked_strengths = fast(X, y, is_classification=False, init_model=ebm)
    # 10 features
    assert 45 == len(ranked_strengths)

    ranked_strengths = fast(X, y, is_classification=False)

def test_case_3(manual_test_regression_data):
    X, y = manual_test_regression_data

    from ...glassbox import ExplainableBoostingRegressor
    ebm = ExplainableBoostingRegressor(random_state=2022)
    # Take notes on interactions found
    ebm.fit(X, y)

    ebm_2 = ExplainableBoostingRegressor(random_state=2022, interactions=0)
    ebm_2.fit(X, y)

    ranked_strengths = fast(X, y, is_classification=False, init_model=ebm_2)
    # 10 features
    assert 45 == len(ranked_strengths)

def test_case_4(manual_test_regression_data):
    X, y = manual_test_regression_data

    from ...glassbox import ExplainableBoostingRegressor
    ebm = ExplainableBoostingRegressor(random_state=2022, learning_rate=0.0000001, max_rounds=1)
    # Take notes on interactions found
    ebm.fit(X, y)

    ranked_strengths = fast(X, y, is_classification=False)
    # 10 features
    assert 45 == len(ranked_strengths)

@pytest.fixture(scope="module")
def modified_diabetes_dataset():
    """
    Fit an EBM with 10 interactions for the diabetes dataset
    Multiply intereactions' additive terms by a scalar to make them stronger
    Return the original X and the ebm predictions as a new y
    """
    from sklearn.datasets import load_diabetes
    diabetes_data = load_diabetes(return_X_y=True)
    X = diabetes_data[0]
    y = diabetes_data[1]

    from ...glassbox import ExplainableBoostingRegressor
    ebm = ExplainableBoostingRegressor(random_state=2022)
    ebm.fit(X, y)

    for additive_term in ebm.additive_terms_[-10:]:
        additive_term *= 100.0

    return X, ebm.predict(X)

def test_case_5(modified_diabetes_dataset):
    X, y = modified_diabetes_dataset

    ranked_strengths = fast(X, y, is_classification=False)
    # 10 features
    assert 45 == len(ranked_strengths)


# TODO Not all x's are used in the function, should we remove it from X?
# TODO How's x generated? Which distribution?
def synthetic_data_function(values):
    """
    values[0] - x1
    values[1] - x2
    ...
    """
    y = np.pi**(values[0]*values[1]) * np.sqrt(2*values[2]) \
        - np.arcsin(values[3]) \
        + np.log10(values[2] + values[4]) \
        - (values[8]/values[9] * np.sqrt(values[6]/values[7])) \
        - values[1]*values[6]
    
    return y

@pytest.fixture(scope="module")
def synthetic_regression_data():
    num_variables = 10
    num_samples = 10000

    X_list = []
    y_list = []
    for sample in range(num_samples):
        features = np.random.rand(num_variables)
        X_list.append(features)
        y_list.append( synthetic_data_function(features) )

    X = np.array(X_list)
    y = np.array(y_list)

    return X, y

def test_case_6(synthetic_regression_data):
    X, y = synthetic_regression_data

    from ...glassbox import ExplainableBoostingRegressor
    ebm = ExplainableBoostingRegressor(random_state=2022)
    # Take notes on interactions found
    ebm.fit(X, y)

    ranked_strengths = fast(X, y, is_classification=False)
    # 10 features
    assert 45 == len(ranked_strengths)


# Set the internal EBM flag to pure when runnign FAST internally
# Run FAST with the original data
# Compare
def test_case_7(manual_test_regression_data):
    X, y = manual_test_regression_data

    from ...glassbox import ExplainableBoostingRegressor
    ebm = ExplainableBoostingRegressor(random_state=2022)
    # Take notes on interactions found
    ebm.fit(X, y)

    ranked_strengths = fast(X, y, is_classification=False)

    # 10 features
    assert 45 == len(ranked_strengths)

# Don't boos mains (low learning rate and only 1 round)
# Set the internal EBM flag to pure when runnign FAST internally
# Run FAST with the original data
# Compare
def test_case_8(manual_test_regression_data):
    X, y = manual_test_regression_data

    from ...glassbox import ExplainableBoostingRegressor
    ebm = ExplainableBoostingRegressor(random_state=2022, learning_rate=0.0000001, max_rounds=1)
    # Take notes on interactions found
    ebm.fit(X, y)

    ranked_strengths = fast(X, y, is_classification=False)

    # 10 features
    assert 45 == len(ranked_strengths)


'''
How to replicate C++ tests?
'''