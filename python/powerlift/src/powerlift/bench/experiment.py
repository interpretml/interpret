"""Experiment classes for benchmarking.

The experiment and its associate classes are the main interface to work with benchmarking.
"""

import random
from types import FunctionType
import pandas as pd
from typing import Dict, Iterable
from typing import Type, TypeVar
from powerlift.bench.store import Store
from typing import Union, Optional, List
from dataclasses import dataclass
from numbers import Number

from powerlift.executors.base import Executor
from powerlift.executors import LocalMachine


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


class Experiment:
    """Represents an experiment with its trials and a store."""

    def __init__(
        self,
        store_or_uri: Union[str, Store],
        name: str = None,
        description: str = None,
        _id: int = None,
    ):
        """Initializes - autogenerates fields if args aren't set.

        Args:
            store_or_uri (Union[str, Store]): Store or database uri.
            name (str, optional): Name of experiment. Defaults to None and autogenerates.
            description (str, optional): Description of experiment. Defaults to None and autogenerates.
            _id (int, optional): ID of experiment. If provided, will associated with an existing experiment with same name. Defaults to None and creates a new experiment.
        """
        if name is None:
            name = f"#{random.randint(0, 9999)}"
        if description is None:
            description = f"Experiment: {name}"
        if isinstance(store_or_uri, str):
            self._store = Store(store_or_uri)
        else:
            self._store = store_or_uri
        self._name = name
        self._description = description
        self._id = _id
        self._trials = []

    @property
    def store(self) -> Store:
        return self._store

    @property
    def id(self) -> int:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def trials(self) -> List:
        return self._trials

    def run(
        self,
        trial_run_fn: FunctionType,
        trial_gen_fn: FunctionType,
        timeout: int = None,
        n_replicates: int = 1,
        executor: Executor = None,
    ) -> Executor:
        """Runs the experiment. Typically done asynchronously.

        Args:
            trial_run_fn (FunctionType): This function is run for each trial (locally or remote). Make sure all imports and objects needed are defined within.
            trial_gen_fn (FunctionType): This function determines which trials are generated from the tasks found in store. Best to see documentation examples for usage.
            timeout (int, optional): Timeout in seconds for each trial run. Defaults to None.
            n_replicates (int, optional): Number of times to repeat a trial run. Defaults to 1.
            executor (Executor, optional): Executor responsible for running trials (local machine, remote docker container, etc). Defaults to None and will select local machine.

        Raises:
            TypeError: Invalid arguments may raise this.

        Returns:
            Executor: Either executor provided to call, or local machine executor.
        """
        # Create experiment if needed
        if self._id is None:
            self._id, _ = self._store.get_or_create_experiment(
                self._name, self._description
            )

        # Create trials
        trial_params = []
        for task in self._store.iter_tasks():
            generated_trials = trial_gen_fn(task)
            for generated_trial in generated_trials:
                if isinstance(generated_trial, tuple):
                    # Response: (method, meta)
                    method = generated_trial[0]
                    meta = generated_trial[1]
                else:
                    # Response: method
                    method = generated_trial
                    meta = {}
                if isinstance(method, str):
                    method = Method.from_name(method)
                elif isinstance(method, Method):
                    pass
                else:
                    raise TypeError(f"Cannot handle method type: {type(method)}")
                method_id, _ = self._store.get_or_create_method(
                    method.name,
                    method.description,
                    method.version,
                    method.params,
                    method.env,
                )
                method = Method(
                    method_id,
                    method.name,
                    method.description,
                    method.version,
                    method.params,
                    method.env,
                )

                for replicate_num in range(n_replicates):
                    trial_param = {
                        "experiment_id": self.id,
                        "task_id": task.id,
                        "method_id": method.id,
                        "replicate_num": replicate_num,
                        "meta": meta,
                    }
                    trial_params.append(trial_param)
                    self._trials.append(
                        Trial(None, self, task, method, replicate_num, meta, [])
                    )

        # Save to store
        trial_ids = self._store.create_trials(trial_params)
        for _id, trial in zip(trial_ids, self._trials):
            trial._id = _id

        # Run trials
        if executor is None:
            executor = LocalMachine(self._store)
        executor.submit(trial_run_fn, self._trials, timeout=timeout)
        return executor


class Trial:
    def __init__(
        self,
        _id: Optional[int],
        experiment: Experiment,
        task: Task,
        method: Method,
        replicate_num: int,
        meta: dict,
        input_assets: List[Asset],
    ):
        """Represents a single trial within an experiment.

        Args:
            _id (Optional[int]): ID of trial or None.
            experiment (Experiment): Experiment of trial.
            task (Task): Task of trial.
            method (Method): Method of trial.
            replicate_num (int): Replicate number of trial (when a trial is repeated many times).
            meta (dict): Metadata associated with the trial.
            input_assets (List[Asset]): Input assets that are available on trial run.
        """
        self._id = _id
        self._experiment = experiment
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
        self._experiment.store.add_measure(
            self._id, type(self), name, value, description, type_, lower_is_better
        )

    @property
    def experiment(self):
        return self._experiment

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
