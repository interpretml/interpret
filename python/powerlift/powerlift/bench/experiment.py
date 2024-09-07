"""Experiment classes for benchmarking.

The experiment and its associate classes.
"""

import pandas as pd
import numpy as np
from typing import Dict, Iterable
from typing import Type, TypeVar
from typing import Union, Optional, List
from dataclasses import dataclass
from numbers import Number
import time

from powerlift.bench.store import Store, MIMETYPE_DF, MIMETYPE_SERIES


@dataclass(frozen=True)
class Wheel:
    """Represents a wheel asset that is used by experiment objects."""

    experiment_id: int
    name: str
    embedded: bytes


@dataclass(frozen=True)
class Measure:
    """Represents a measure emitted by a trial or task.

    This is what gets logged by the user in trial runs. Also used by tasks to record statistics (i.e. number of classes for a binary classification task).
    """

    name: str
    description: str
    type: type
    lower_is_better: bool
    values: pd.DataFrame


@dataclass(frozen=True)
class Task:
    """Represents a problem and its associated data. For instance, binary classification on the adult dataset."""

    store: Store

    id: Optional[int]
    name: str
    problem: str
    origin: str
    n_samples: int
    n_features: int
    n_classes: int
    meta: Dict[str, object]

    def data(self) -> List[object]:
        """Returns assets.

        Returns:
            List[object]: List of assets retrieved.
        """
        from powerlift.bench.store import BytesParser

        x, y = self.store.get_assets(self.id)
        x = BytesParser.deserialize(MIMETYPE_DF, x)
        y = np.array(BytesParser.deserialize(MIMETYPE_SERIES, y))

        return (x, y)


@dataclass(frozen=True)
class Experiment:
    """Represents an experiment with its trials."""

    id: Optional[int]
    name: str
    description: str
    shell_install: str
    pip_install: str
    script: str
    trial_fn: str
    wheels: List[Wheel]
    trials: List


class Trial:
    def __init__(
        self,
        _id: Optional[int],
        store: Store,
        task: Task,
        method: str,
        replicate_num: int,
        meta: Dict[str, object],
    ):
        """Represents a single trial within an experiment.

        Args:
            _id (Optional[int]): ID of trial or None.
            store (Store): Store to persist measures.
            task (Task): Task of trial.
            method (str): Method of trial.
            replicate_num (int): Replicate number of trial (when a trial is repeated many times).
            meta (Dict[str, object]): Metadata associated with the trial.
        """
        self._id = _id
        self._store = store
        self._task = task
        self._method = method
        self._replicate_num = replicate_num
        self._meta = meta
        self._measure_counts = {}

    def log(
        self,
        name: str,
        value: Union[Number, str, dict],
    ):
        """Records a measure for the trial. This should be called in the trial run function.

        Args:
            name (str): Name of measure.
            value (Union[Number, str, dict]): Value of measure.
        """

        seq_num = self._measure_counts[name] = self._measure_counts.get(name, -1) + 1
        self._store.add_measure(
            self._id,
            name,
            value,
            seq_num,
        )

    @property
    def store(self):
        return self._store

    @property
    def task(self):
        return self._task

    @property
    def method(self):
        return self._method

    @property
    def replicate_num(self):
        return self._replicate_num

    @property
    def meta(self):
        return self._meta.copy()

    @property
    def id(self):
        return self._id
