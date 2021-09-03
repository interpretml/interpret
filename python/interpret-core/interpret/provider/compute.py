# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

from joblib import Parallel, delayed
import gc
from abc import ABC, abstractmethod


class ComputeProvider(ABC):
    @abstractmethod
    def parallel(self, compute_fn, compute_args_iter):
        pass  # pragma: no cover


class JobLibProvider(ComputeProvider):
    def __init__(self, n_jobs=-1):
        self.n_jobs = n_jobs

    def parallel(self, compute_fn, compute_args_iter):
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(compute_fn)(*args) for args in compute_args_iter
        )
        # NOTE: Force gc, as Python does not free native memory easy.
        gc.collect()
        return results


# NOTE: Not implemented yet
class AzureMLProvider(ComputeProvider):
    def parallel(self, compute_fn, compute_args_iter):
        raise NotImplementedError
