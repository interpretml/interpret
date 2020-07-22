# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

from ..api.templates import FeatureValueExplanation
from . import gen_name_from_class, unify_data, gen_perf_dicts, gen_local_selector


def shap_explain_local(explainer, X, y=None, name=None, is_classification=False, check_additivity=True):
    if name is None:
        name = gen_name_from_class(explainer)
    X, y, _, _ = unify_data(X, y, explainer.feature_names, explainer.feature_types)

    if is_classification:
        all_shap_values = explainer.shap.shap_values(X, check_additivity=check_additivity)[1]
        expected_value = explainer.shap.expected_value[1]
    else:
        all_shap_values = explainer.shap.shap_values(X, check_additivity=check_additivity)
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
