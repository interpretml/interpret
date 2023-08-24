# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license


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
