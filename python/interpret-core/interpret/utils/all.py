# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

from itertools import count

import numpy as np
import pandas as pd

import logging

log = logging.getLogger(__name__)


def gen_perf_dicts(scores, y, is_classification, classes=None):
    # TODO: rename from scores to something else: predicted & make predicted best_predicted
    #       or perhaps make it predicted but then add a predicted_proba just for classification
    if is_classification:
        if classes is not None:
            invert_classes = dict(zip(classes, count(0)))
        if scores.ndim == 1:
            scores = np.vstack([1 - scores, scores]).T

        predicted = np.argmax(scores, axis=1)
    else:
        predicted = scores

    records = []
    for i in range(len(predicted)):
        di = {}
        di["is_classification"] = is_classification
        di["actual"] = np.nan if y is None else y[i]

        if is_classification:
            di["predicted"] = predicted[i] if classes is None else classes[predicted[i]]
            actual_prob = np.nan
            if y is not None:
                if classes is None:
                    # if y is a legal integer then we should assume it's an index
                    inv_index = y[i]
                    try:
                        inv_index = int(inv_index)
                        if inv_index < scores.shape[1]:
                            actual_prob = scores[i, inv_index]
                    except ValueError:
                        pass
                else:
                    inv_index = invert_classes.get(y[i], -1)
                    actual_prob = 0 if inv_index < 0 else scores[i, inv_index]
            di["actual_score"] = actual_prob
            di["predicted_score"] = scores[i, predicted[i]]

            # TODO: The UI currently expects an index in di["predicted"] and di["actual"]
            #       and then it uses the classes to map to the original strings, so it
            #       works in all cases EXCEPT if di["actual"] is something new like
            #       y[0] = "NEVER_SEEN_BEFORE".  In that case the value is not in the classes
            #       array and therefore is not preserved.  If we change di["predicted"] and
            #       di["actual"] to hold the actual value then we could display
            #       "NEVER_SEEN_BEFORE" in the "actual" value field
            #       FOR NOW WE'RE MAPPING IT BACK TO INDEXES SO THAT THE UI WORKS, BUT CHANGE THIS
            # START SECTION TO BE REMOVED
            di["predicted"] = predicted[i]
            di["actual"] = np.nan
            if y is not None:
                if classes is None:
                    inv_index = y[i]
                    try:
                        inv_index = int(inv_index)
                        if inv_index < scores.shape[1]:
                            di["actual"] = inv_index
                    except ValueError:
                        pass
                else:
                    di["actual"] = invert_classes.get(y[i], np.nan)
            # TODO: END SECTION TO BE REMOVED
        else:
            di["predicted"] = predicted[i]
            di["actual_score"] = np.nan if y is None else y[i]
            di["predicted_score"] = scores[i]

        records.append(di)

    return records


def gen_global_selector(
    n_samples,
    n_features,
    term_names,
    term_types,
    unique_val_counts,
    zero_val_counts,
    importance_scores,
    round=3,
):
    records = []
    for term_idx in range(len(term_names)):
        record = {}
        record["Name"] = term_names[term_idx]
        record["Type"] = _legacy_type(term_types[term_idx])

        if term_idx < n_features:
            record["# Unique"] = (
                np.nan if unique_val_counts is None else unique_val_counts[term_idx]
            )
            if n_samples is None or zero_val_counts is None:
                record["% Non-zero"] = np.nan
            else:
                record["% Non-zero"] = (
                    n_samples - zero_val_counts[term_idx]
                ) / n_samples

            # if importance_scores is None:
            #     record["Importance"] = np.nan
            # else:
            #     record["Importance"] = importance_scores[term_idx]
        else:
            record["# Unique"] = np.nan
            record["% Non-zero"] = np.nan
            # record["Importance"] = np.nan

        records.append(record)

    # columns = ["Name", "Type", "# Unique", "% Non-zero", "Importance"]
    columns = ["Name", "Type", "# Unique", "% Non-zero"]
    df = pd.DataFrame.from_records(records, columns=columns)
    if round is not None:
        return df.round(round)
    else:  # pragma: no cover
        return df


def gen_local_selector(data_dicts, round=3, is_classification=True):
    records = []

    for data_dict in data_dicts:
        perf_dict = data_dict["perf"]
        record = {}
        record["PrScore"] = perf_dict["predicted_score"]
        record["AcScore"] = perf_dict["actual_score"]

        record["Predicted"] = perf_dict["predicted"]
        record["Actual"] = perf_dict["actual"]

        record["Resid"] = record["AcScore"] - record["PrScore"]
        record["AbsResid"] = abs(record["Resid"])

        records.append(record)

    if is_classification:
        columns = ["Actual", "Predicted", "PrScore", "AcScore", "Resid", "AbsResid"]
    else:
        columns = ["Actual", "Predicted", "Resid", "AbsResid"]

    df = pd.DataFrame.from_records(records, columns=columns)
    if round is not None:
        return df.round(round)
    else:  # pragma: no cover
        return df


def gen_name_from_class(obj):
    """Generates a name for a given class.

    Args:
        obj: An object.

    Returns:
        A generated name as a string that uses
        class name and a static counter.
    """
    class_name = obj.__class__.__name__
    if class_name not in gen_name_from_class.cache:
        gen_name_from_class.cache[class_name] = count(0)
    identifier = next(gen_name_from_class.cache[class_name])

    return str(obj.__class__.__name__) + "_" + str(identifier)


gen_name_from_class.cache = {}


def _legacy_type(feature_type):
    # TODO: someday get rid of this when we've propagated nominal and ordinal to the UI
    return (
        "categorical"
        if feature_type == "nominal" or feature_type == "ordinal"
        else feature_type
    )
