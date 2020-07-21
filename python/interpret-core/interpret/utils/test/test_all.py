from ..all import gen_perf_dicts

import numpy as np


def test_gen_perf_dicts_regression():
    y = np.array([0, 0.5])
    scores = np.array([0.9, 0.1])
    expected_predicted = np.array([0.9, 0.1])
    expected_actual_score = np.array([0, 0.5])
    expected_predicted_score = np.array([0.9, 0.1])

    records = gen_perf_dicts(scores, y, False)
    for i, di in enumerate(records):
        assert di["actual"] == y[i]
        assert di["predicted"] == expected_predicted[i]
        assert di["actual_score"] == expected_actual_score[i]
        assert di["predicted_score"] == expected_predicted_score[i]


def test_gen_perf_dicts_classification():
    y = np.array([0, 2])
    scores = np.array([[0.9, 0.06, 0.04], [0.1, 0.5, 0.4],])
    expected_predicted = np.array([0, 1])
    expected_actual_score = np.array([0.9, 0.4])
    expected_predicted_score = np.array([0.9, 0.5])

    records = gen_perf_dicts(scores, y, True)
    for i, di in enumerate(records):
        assert di["actual"] == y[i]
        assert di["predicted"] == expected_predicted[i]
        assert di["actual_score"] == expected_actual_score[i]
        assert di["predicted_score"] == expected_predicted_score[i]
