# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

from collections.abc import Iterable
import sys
from ._misc import safe_isinstance
import numpy as np
from typing import Any


def total_bytes(obj: Any) -> int:
    n_bytes = 0
    items = [obj]

    seen_ids = set()
    while items:
        item = items.pop()

        obj_id = id(item)
        if obj_id in seen_ids:
            continue
        seen_ids.add(obj_id)

        if safe_isinstance(item, "pandas.DataFrame"):
            n_bytes += item.memory_usage().sum()
            # pandas only includes the pointer to the object but not the object
            for col in item.select_dtypes(include=["object"]):
                for val in item[col]:
                    try:
                        n_bytes += sys.getsizeof(val)
                    except Exception:
                        pass
                    if isinstance(val, Iterable) and not isinstance(val, (str, bytes)):
                        try:
                            items.extend(val)
                        except Exception:
                            pass
        elif safe_isinstance(item, "pandas.Series"):
            n_bytes += item.memory_usage()
            if item.dtype == "O":
                for val in item:
                    try:
                        n_bytes += sys.getsizeof(val)
                    except Exception:
                        pass
                    if isinstance(val, Iterable) and not isinstance(val, (str, bytes)):
                        try:
                            items.extend(val)
                        except Exception:
                            pass
        elif isinstance(item, np.ndarray):
            n_bytes += item.nbytes
            if item.dtype == "O":
                items.extend(item.flat)
        elif safe_isinstance(item, "scipy.sparse.spmatrix") or safe_isinstance(
            item, "scipy.sparse.sparray"
        ):
            n_bytes += item.data.nbytes + item.indptr.nbytes + item.indices.nbytes
        elif isinstance(item, dict):
            try:
                n_bytes += sys.getsizeof(item)
            except Exception:
                pass
            items.extend(item.values())
            items.extend(item.keys())
        else:
            try:
                n_bytes += sys.getsizeof(item)
            except Exception:
                pass
            if isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
                try:
                    items.extend(item)
                except Exception:
                    pass

    return int(n_bytes)
