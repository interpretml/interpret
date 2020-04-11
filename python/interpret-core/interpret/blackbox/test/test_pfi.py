from interpret.test.utils import synthetic_regression, synthetic_classification
from interpret.blackbox.pfi import PermutationImportanceClassification
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn import svm


def test_classifier():
    data = synthetic_classification()
    X = data["full"]["X"]
    y = data["full"]["y"]
    clf = svm.SVC(gamma=0.001, C=100., probability=True, random_state=777)
    clf.fit(X, y)
    metric = X.columns[0]

    explainer = PermutationImportanceClassification(clf.predict, metric)
    explanation = explainer.explain_global(X, y)
    import pdb; pdb.set_trace()
