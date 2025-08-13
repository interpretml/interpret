# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

import numpy as np

from ..api.templates import FeatureValueExplanation
from ..utils._clean_simple import clean_dimensions, typify_classification
from ..utils._clean_x import preclean_X
from ..utils._unify_data import unify_data
from ..utils._unify_predict import determine_classes, unify_predict_fn
from ._explanation import gen_local_selector, gen_name_from_class, gen_perf_dicts


def shap_explain_local(explainer, X, y, name, is_treeshap, **kwargs):
    import shap

    if name is None:
        name = gen_name_from_class(explainer)

    n_samples = None
    if y is not None:
        y = clean_dimensions(y, "y")
        if y.ndim != 1:
            msg = "y must be 1 dimensional"
            raise ValueError(msg)
        n_samples = len(y)

    feature_names = (
        explainer.feature_names
        if explainer.feature_names_in_ is None
        else explainer.feature_names_in_
    )
    feature_types = (
        explainer.feature_types
        if explainer.feature_types_in_ is None
        else explainer.feature_types_in_
    )

    X, n_samples = preclean_X(X, feature_names, feature_types, n_samples)

    predict_fn, n_classes, classes = determine_classes(explainer.model, X, n_samples)
    if n_classes >= 3:
        msg = "multiclass SHAP not supported"
        raise Exception(msg)
    predict_fn = unify_predict_fn(predict_fn, X, 1 if n_classes == 2 else -1)

    X, feature_names, feature_types = unify_data(
        X, n_samples, feature_names, feature_types, False, 0
    )

    # SHAP does not support string categoricals, and np.object_ is slower,
    # so convert to np.float64 until we implement some automatic categorical handling
    X = X.astype(np.float64, order="C", copy=False)

    if y is not None:
        # Use n_classes >= 0 to determine if this is classification
        # n_classes >= 0 indicates classification, n_classes < 0 indicates regression
        if n_classes >= 0:
            y = typify_classification(y)
        else:
            y = y.astype(np.float64, copy=False)

    if is_treeshap and n_classes == 2:
        all_shap_values = explainer.shap_.shap_values(X, **kwargs)
        if shap.__version__ >= "0.45.0":
            all_shap_values = all_shap_values[..., 1]
        else:
            all_shap_values = all_shap_values[1]
        expected_value = explainer.shap_.expected_value[1]
    else:
        all_shap_values = explainer.shap_.shap_values(X, **kwargs)
        expected_value = explainer.shap_.expected_value

    predictions = predict_fn(X)

    data_dicts = []
    scores_list = all_shap_values
    perf_list = []
    # Convert model classes to strings the same way as y values for consistency
    model_is_classifier = n_classes >= 0
    if model_is_classifier and classes is not None:
        # Apply the same typify_classification to classes as we do to y
        classes = typify_classification(classes)
    
    perf_dicts = gen_perf_dicts(predictions, y, model_is_classifier, classes)
    for i, instance in enumerate(X):
        shap_values = all_shap_values[i]
        perf_dict_obj = None if perf_dicts is None else perf_dicts[i]

        perf_list.append(perf_dict_obj)

        data_dict = {
            "type": "univariate",
            "names": feature_names,
            "perf": perf_dict_obj,
            "scores": shap_values,
            "values": instance,
            "extra": {
                "names": ["Base Value"],
                "scores": [expected_value],
                "values": [1],
            },
        }
        data_dicts.append(data_dict)

    internal_obj = {
        "overall": None,
        "specific": data_dicts,
        "mli": [
            {
                "explanation_type": "local_feature_importance",
                "value": {
                    "scores": scores_list,
                    "intercept": expected_value,
                    "perf": perf_list,
                },
            }
        ],
    }
    internal_obj["mli"].append(
        {
            "explanation_type": "evaluation_dataset",
            "value": {"dataset_x": X, "dataset_y": y},
        }
    )
    selector = gen_local_selector(data_dicts, is_classification=model_is_classifier)

    return FeatureValueExplanation(
        "local",
        internal_obj,
        feature_names=feature_names,
        feature_types=feature_types,
        name=name,
        selector=selector,
    )
