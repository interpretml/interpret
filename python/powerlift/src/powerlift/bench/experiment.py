"""Experiment classes for benchmarking.

The experiment and its associate classes.
"""

import pandas as pd
from typing import Dict, Iterable
from typing import Type, TypeVar
from typing import Union, Optional, List
from dataclasses import dataclass
from numbers import Number

from powerlift.bench.store import Store


@dataclass(frozen=True)
class Asset:
    """Represents an asset (including mimetype) that is produced or used by trial and task objects."""

    id: Optional[int]
    name: str
    description: str
    version: str
    is_embedded: bool
    embedded: Optional[bytes]
    uri: bool
    mimetype: str


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

    id: Optional[int]
    name: str
    description: str
    version: str
    problem: str
    origin: str
    config: dict
    assets: List[Asset]
    measures: Dict[str, Measure]

    def measure(self, name: str) -> pd.DataFrame:
        """Returns a named measure as a dataframe with sequence number, timestamp and value for entries.

        Args:
            name (str): Name of measure.

        Returns:
            pd.DataFrame: A dataframe with values associated with named measure.
        """
        return self.measures[name].values

    def scalar_measure(self, name: str) -> Union[Number, str, dict]:
        """Returns a named measure as a scalar value.

        Args:
            name (str): Name of measure.

        Returns:
            Union[Number, str, dict]: Scalar value associated with named measure.
        """
        return self.measures[name].values["val"].iloc[0]

    def data(self, aliases: Iterable[str]) -> List[object]:
        """Returns assets as a list of objects that maps to the aliases provided.

        Args:
            aliases (Iterable[str]): Aliases of assets to be retrieved.

        Returns:
            List[object]: List of objects retrieved by aliases.
        """
        from powerlift.bench.store import BytesParser

        outputs = []
        alias_map = self.config["aliases"]
        name_to_asset = {asset.name: asset for asset in self.assets}
        for alias in aliases:
            name = alias_map[alias]
            asset = name_to_asset[name]
            parsed = BytesParser.deserialize(asset.mimetype, asset.embedded)
            outputs.append(parsed)
        return outputs


T = TypeVar("T", bound="Method")


@dataclass(frozen=True)
class Method:
    """Represents a method that is ran for a given trial. I.e. support vector machine for a binary classification task."""

    id: Optional[int]
    name: str
    description: str
    version: str
    params: dict
    env: dict

    @classmethod
    def from_name(cls: Type[T], name: str) -> T:
        """Produces Method object from name.

        Args:
            name (str): Name of method to be created.

        Returns:
            Method: Created method.
        """
        return cls(None, name, f"Method: {name}", "0.0.1", {}, {})


@dataclass(frozen=True)
class Experiment:
    """Represents an experiment with its trials."""

    id: Optional[int]
    name: str
    description: str
    trials: List


class Trial:
    def __init__(
        self,
        _id: Optional[int],
        store: Store,
        task: Task,
        method: Method,
        replicate_num: int,
        meta: dict,
        input_assets: List[Asset],
    ):
        """Represents a single trial within an experiment.

        Args:
            _id (Optional[int]): ID of trial or None.
            store (Store): Store to persist measures.
            task (Task): Task of trial.
            method (Method): Method of trial.
            replicate_num (int): Replicate number of trial (when a trial is repeated many times).
            meta (dict): Metadata associated with the trial.
            input_assets (List[Asset]): Input assets that are available on trial run.
        """
        self._id = _id
        self._store = store
        self._task = task
        self._method = method
        self._replicate_num = replicate_num
        self._meta = meta
        self._input_assets = input_assets

    def log(
        self,
        name: str,
        value: Union[Number, str, dict],
        description: Optional[str] = None,
        type_: Union[None, Type, str] = None,
        lower_is_better: bool = True,
    ):
        """Records a measure for the trial. This should be called in the trial run function.

        Args:
            name (str): Name of measure.
            value (Union[Number, str, dict]): Value of measure.
            description (Optional[str], optional): Description of measure. Defaults to None.
            type_ (Union[None, Type, str], optional): Type of measure. Defaults to None.
            lower_is_better (bool, optional): Whether the measure is considered better at a lower value. Defaults to True.
        """
        self._store.add_measure(
            self._id, type(self), name, value, description, type_, lower_is_better
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
        return self._meta

    @property
    def input_assets(self):
        return self._input_assets

    @property
    def id(self):
        return self._id
