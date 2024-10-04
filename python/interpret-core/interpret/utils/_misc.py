# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license
import logging
from itertools import count

import numpy as np

_log = logging.getLogger(__name__)


def clean_index(index, n_items, names, param_name, attribute_name):
    if isinstance(index, str):
        if names is None:
            msg = f"{param_name} cannot be used to index by name since {attribute_name} has been removed."
            _log.error(msg)
            raise ValueError(msg)
        try:
            index = names.index(index)
        except:
            msg = f'{attribute_name} does not contain "{index}".'
            _log.error(msg)
            raise ValueError(msg)
    else:
        if isinstance(index, int):
            pass
        elif isinstance(index, float):
            if index.is_integer():
                index = int(index)
            else:
                msg = f"{param_name} is {index}, which is not an integer."
                _log.error(msg)
                raise ValueError(msg)
        else:
            msg = f"{param_name} must be an integer index or string name."
            _log.error(msg)
            raise ValueError(msg)

        if index < 0 or n_items <= index:
            msg = f"{param_name} index {index} out of bounds."
            _log.error(msg)
            raise ValueError(msg)
    return index


def clean_indexes(indexes, n_items, names, param_name, attribute_name):
    if names is not None:
        names = dict(zip(names, count()))
    n_bools = 0
    n_indexes = 0
    result = set()
    for i, v in enumerate(indexes):
        n_indexes += 1
        if isinstance(v, bool) or isinstance(v, np.bool_):
            n_bools += 1
            if v:
                v = i
        elif isinstance(v, str):
            if names is None:
                msg = f"{param_name} cannot be indexed by name since {attribute_name} has been removed."
                _log.error(msg)
                raise ValueError(msg)
            try:
                v = names[v]
            except:
                msg = f'{attribute_name} does not contain "{v}".'
                _log.error(msg)
                raise ValueError(msg)
        else:
            if isinstance(v, int):
                pass
            elif isinstance(v, float):
                if v.is_integer():
                    v = int(v)
                else:
                    msg = f"{param_name} contains {v}, which is not an integer."
                    _log.error(msg)
                    raise ValueError(msg)
            else:
                msg = f"{param_name} must contain integer indexes or string names or booleans."
                _log.error(msg)
                raise ValueError(msg)

            if v < 0 or n_items <= v:
                msg = f"{param_name} index {v} out of bounds."
                _log.error(msg)
                raise ValueError(msg)

        result.add(v)

    if n_bools != 0:
        if n_indexes != n_bools:
            msg = f"If {param_name} contains booleans, they must all be booleans."
            _log.error(msg)
            raise ValueError(msg)
        if n_items != n_bools:
            msg = f"If {param_name} contains booleans, it must be the same length as in the EBM."
            _log.error(msg)
            raise ValueError(msg)

    return result
