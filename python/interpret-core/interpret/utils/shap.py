# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

from ..api.templates import FeatureValueExplanation
from . import gen_name_from_class, gen_perf_dicts, gen_local_selector

import numpy as np
from ..utils._binning import (
    determine_min_cols,
    clean_X,
    unify_data2,
    clean_dimensions,
    typify_classification,
)


def shap_explain_local(explainer, X, y, name, is_take_only_second, **kwargs):
    if name is None:
        name = gen_name_from_class(explainer)

    n_samples = None
    if y is not None:
        y = clean_dimensions(y, "y")
        if y.ndim != 1:
            raise ValueError("y must be 1 dimensional")
        n_samples = len(y)

        if 0 <= explainer.n_classes:
            y = typify_classification(y)
        else:
            y = y.astype(np.float64, copy=False)

    min_cols = determine_min_cols(explainer.feature_names, explainer.feature_types)
    X, n_samples = clean_X(X, min_cols, n_samples)

    X, _, _ = unify_data2(
        X, n_samples, explainer.feature_names, explainer.feature_types, False, 0
    )

    if is_take_only_second:
        all_shap_values = explainer.shap.shap_values(X, **kwargs)[1]
        expected_value = explainer.shap.expected_value[1]
    else:
        all_shap_values = explainer.shap.shap_values(X, **kwargs)
        expected_value = explainer.shap.expected_value

    predictions = explainer.predict_fn(X)

    data_dicts = []
    scores_list = all_shap_values
    perf_list = []
    perf_dicts = gen_perf_dicts(predictions, y, False)
    for i, instance in enumerate(X):
        shap_values = all_shap_values[i]
        perf_dict_obj = None if perf_dicts is None else perf_dicts[i]

        perf_list.append(perf_dict_obj)

        data_dict = {
            "type": "univariate",
            "names": explainer.feature_names,
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
    selector = gen_local_selector(data_dicts, is_classification=False)

    return FeatureValueExplanation(
        "local",
        internal_obj,
        feature_names=explainer.feature_names,
        feature_types=explainer.feature_types,
        name=name,
        selector=selector,
    )
