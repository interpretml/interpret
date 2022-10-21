import pytest
import numpy as np
from math import isclose

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.dummy import DummyClassifier

from ...test.utils import synthetic_regression, synthetic_classification, synthetic_multiclass
from ...utils import measure_interactions

@pytest.fixture(scope="module")
def regression_data():
    data = synthetic_regression()
    return data["full"]["X"], data["full"]["y"]

def test_init_regression_model(regression_data):
    X, y = regression_data
    init_scores = np.random.rand(X.shape[0])

    lr = LinearRegression()
    lr.fit(X, y)

    ranked_pairs = measure_interactions(X, y, init_score=lr)
    # 4 features
    assert 6 == len(ranked_pairs)

    ranked_pairs = measure_interactions(X, y, init_score=init_scores)
    assert 6 == len(ranked_pairs)

def test_init_binary_model():
    data = synthetic_classification()
    X = data["full"]["X"]
    y = np.array(data["full"]["y"]).reshape(-1)

    init_scores = np.random.rand(X.shape[0])

    lr = LogisticRegression(solver='lbfgs')
    lr.fit(X, y)

    ranked_pairs = measure_interactions(X, y, init_score=lr)
    # 4 features
    assert 6 == len(ranked_pairs)

    ranked_pairs = measure_interactions(X, y, init_score=init_scores)
    assert 6 == len(ranked_pairs)

def test_init_multiclass_model():
    data = synthetic_multiclass()
    X = data["full"]["X"]
    y = data["full"]["y"]

    classes, y = np.unique(y, return_inverse=True)
    n_classes = len(classes)

    lr = LogisticRegression(solver='lbfgs', multi_class='auto')
    lr.fit(X, y)

    ranked_pairs = measure_interactions(X, y, init_score=lr)
    # 4 features
    assert 6 == len(ranked_pairs)

def test_init_multiclass_scores():
    data = synthetic_multiclass()
    X = data["full"]["X"]
    y = data["full"]["y"]

    classes, y = np.unique(y, return_inverse=True)
    n_classes = len(classes)

    init_scores = np.random.rand(X.shape[0], n_classes)

    ranked_pairs = measure_interactions(X, y, init_score=init_scores)
    assert 6 == len(ranked_pairs)

def test_init_binary_dummy_model():
    data = synthetic_classification()
    X = data["full"]["X"]
    y = data["full"]["y"]

    lr = DummyClassifier() # has no decision_function only predict_proba
    lr.fit(X, y)

    ranked_pairs = measure_interactions(X, y, init_score=lr)
    # 4 features
    assert 6 == len(ranked_pairs)

def test_init_binary_dummy_scores():
    data = synthetic_classification()
    X = data["full"]["X"]
    y = data["full"]["y"]

    init_scores = np.random.rand(X.shape[0])

    ranked_pairs = measure_interactions(X, y, init_score=init_scores)
    assert 6 == len(ranked_pairs)

def test_init_multiclass_dummy_model():
    data = synthetic_multiclass()
    X = data["full"]["X"]
    y = data["full"]["y"]

    classes, y = np.unique(y, return_inverse=True)
    n_classes = len(classes)

    lr = DummyClassifier() # has no decision_function only predict_proba
    lr.fit(X, y)

    ranked_pairs = measure_interactions(X, y, init_score=lr)
    # 4 features
    assert 6 == len(ranked_pairs)

def test_init_multiclass_dummy_scores():
    data = synthetic_multiclass()
    X = data["full"]["X"]
    y = data["full"]["y"]

    classes, y = np.unique(y, return_inverse=True)
    n_classes = len(classes)

    init_scores = np.random.rand(X.shape[0], n_classes)

    ranked_pairs = measure_interactions(X, y, init_score=init_scores)
    assert 6 == len(ranked_pairs)

def test_inconsistent_objective(regression_data):
    X, y = regression_data

    lr = LinearRegression()

    with pytest.raises(ValueError):
        measure_interactions(X, y, init_score=lr, objective='classification')

def test_inconsistent_X_and_y(regression_data):
    X, y = regression_data
    y = y.head(10)

    with pytest.raises(ValueError):
        measure_interactions(X, y)

def test_inconsistent_sample_weigth(regression_data):
    X, y = regression_data
    sample_weight = np.random.rand(X.shape[0] - 10)

    with pytest.raises(ValueError):
        measure_interactions(X, y, sample_weight=sample_weight)

def test_sample_weigth(regression_data):
    X, y = regression_data
    sample_weight = np.random.rand(X.shape[0])

    ranked_pairs = measure_interactions(X, y, sample_weight=sample_weight)
    # 4 features
    assert 6 == len(ranked_pairs)

def test_feature_names_and_types(regression_data):
    X, y = regression_data

    # Original feature names "A", "B", "C", "D"
    feature_names = ["FtA", "FtB", "FtC", "FtD"]
    feature_types = ["continuous", "continuous", "continuous", "continuous"]

    ranked_pairs = measure_interactions(X, y, feature_names=feature_names, feature_types=feature_types)
    assert 6 == len(ranked_pairs)

def test_max_bins_and_binning_options(regression_data):
    X, y = regression_data

    max_interaction_bins = 64
    binning = "uniform"

    ranked_pairs = measure_interactions(X, y, max_interaction_bins=max_interaction_bins, binning=binning)
    assert 6 == len(ranked_pairs)

    binning = "rounded_quantile"

    ranked_pairs = measure_interactions(X, y, max_interaction_bins=max_interaction_bins, binning=binning)
    assert 6 == len(ranked_pairs)

def test_min_samples_leaf(regression_data):
    X, y = regression_data

    min_samples_leaf = 4

    ranked_pairs = measure_interactions(X, y, min_samples_leaf=min_samples_leaf)
    assert 6 == len(ranked_pairs)

def test_num_output_interactions(regression_data):
    X, y = regression_data

    ranked_pairs = measure_interactions(X, y, interactions=3)
    assert 3 == len(ranked_pairs)

    ranked_pairs = measure_interactions(X, y, interactions=100)
    assert 6 == len(ranked_pairs)

    ranked_pairs = measure_interactions(X, y, interactions=0)
    assert 6 == len(ranked_pairs)

    ranked_pairs = measure_interactions(X, y, interactions=-2)
    assert 6 == len(ranked_pairs)

def test_output_list(regression_data):
    X, y = regression_data

    ranked_pairs_list = measure_interactions(X, y)
    assert type(ranked_pairs_list) is list
    assert 6 == len(ranked_pairs_list)
    for key, value in ranked_pairs_list:
        assert isinstance(key, tuple)
        assert isinstance(value, float)

def test_specific_results(regression_data):
    X, y = regression_data

    baseline = dict(measure_interactions(X, y))
    assert 6 == len(baseline)

    specific = dict(measure_interactions(X, y, interactions=[(2, 1), (2, 0)]))
    assert 2 == len(specific)

    assert isclose(specific[(2, 1)], baseline[(1, 2)])
    assert isclose(specific[(2, 0)], baseline[(0, 2)])
    
def test_regression_task():
    from sklearn.datasets import load_diabetes
    diabetes_data = load_diabetes(return_X_y=True)
    X = diabetes_data[0]
    y = diabetes_data[1]

    ranked_strengths = measure_interactions(X, y)

    # 10 features
    assert 45 == len(ranked_strengths)

def test_classification_task():
    from sklearn.datasets import load_breast_cancer
    breast_cancer_data = load_breast_cancer(return_X_y=True)
    X = breast_cancer_data[0]
    y = breast_cancer_data[1]

    ranked_strengths = measure_interactions(X, y)

    # 30 features
    assert 435 == len(ranked_strengths)

def test_nulticlass_task():
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=100, n_features=10, n_informative=3, n_classes=3, random_state=2022)

    ranked_strengths = measure_interactions(X, y)

    assert 45 == len(ranked_strengths)

def test_impure_interaction_is_zero():
    X = [
        ["A", "A"],
        ["A", "B"],
        ["B", "A"],
        ["B", "B"],
    ]
    y = [
        3.0 + 11.0,
        3.0 + 7.0,
        5.0 + 11.0,
        5.0 + 7.0
    ]
    sample_weight = [
        24.25,
        21.5,
        8.125,
        11.625
    ]

    ranked_strengths = dict(measure_interactions(X, y, min_samples_leaf=1, sample_weight=sample_weight, objective='regression'))
    assert ranked_strengths[(0, 1)] == 0.0

def test_added_impure_contribution_is_zero():
    X = [
        ["A", "A"],
        ["A", "B"],
        ["B", "A"],
        ["B", "B"],
    ]
    y = [
        -16.0,
        2.0,
        32.0,
        -8.0
    ]
    sample_weight = [
        2.5,
        20,
        1.25,
        5
    ]

    ranked_strengths_pure_int = dict(measure_interactions(X, y, min_samples_leaf=1, sample_weight=sample_weight))

    y = [
        -16.0 + 3.0 + 11.0,
        2.0 + 3.0 + 7.0,
        32.0 +5.0 + 11.0,
        -8.0 + 5.0 + 7.0
    ]

    ranked_strengths_impure = dict(measure_interactions(X, y, min_samples_leaf=1, sample_weight=sample_weight))
    assert ranked_strengths_pure_int[(0, 1)] == ranked_strengths_impure[(0, 1)]
