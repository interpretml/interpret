# from interpret.test.utils import synthetic_regression, synthetic_classification
# from interpret.blackbox import PermutationImportance
# from interpret.blackbox.permutationimportance import VALID_SKLEARN_METRICS
#
# from sklearn import svm
# import sklearn.metrics
#
#
# def test_classifier():
#     data = synthetic_classification()
#     X = data["full"]["X"]
#     y = data["full"]["y"]
#     clf = svm.SVC(gamma=0.001, C=100., probability=True, random_state=777)
#     clf.fit(X, y)
#     metric = X.columns[0]
#
#     explainer = PermutationImportance(clf.predict, X, y)
#     explainer.explain_global()
#
# def test_regressor():
#     data = synthetic_regression()
#     X = data["full"]["X"]
#     y = data["full"]["y"]
#     clf = svm.SVR(gamma=0.001, C=100.)
#     clf.fit(X, y)
#     metric = X.columns[0]
#
#     explainer = PermutationImportance(clf.predict, X, y)
#     explainer.explain_global()
#
# def test_regressor_f1_score():
#     data = synthetic_regression()
#     X = data["full"]["X"]
#     y = data["full"]["y"]
#     clf = svm.SVR(gamma=0.001, C=100.)
#     clf.fit(X, y)
#     metric = X.columns[0]
#
#     explainer = PermutationImportance(clf.predict, X, y, metric="mean_absolute_error")
#     explainer.explain_global()
#
# def test_metric_to_func_returns_func():
#     for metric in VALID_SKLEARN_METRICS:
#         metric_func = getattr(sklearn.metrics, metric)
#         assert callable(metric_func)
