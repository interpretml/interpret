"""User facing benchmark class."""

from types import FunctionType
from typing import Optional, Union

from powerlift.bench.experiment import Experiment, Method, Trial
from powerlift.bench.store import Store
from powerlift.executors.base import Executor
from powerlift.executors import LocalMachine
import pandas as pd
import random

import os


class Benchmark:
    def __init__(
        self,
        store_or_uri: Optional[Union[str, Store]] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        if store_or_uri is None:
            self._store = Store(
                "sqlite:///" + os.path.join(os.getcwd(), "powerlift.db")
            )
        elif isinstance(store_or_uri, str):
            self._store = Store(store_or_uri)
        elif isinstance(store_or_uri, Store):
            self._store = store_or_uri
        else:  # pragma: no cover
            raise TypeError(f"Incorrect type: {type(store_or_uri)}")

        if name is None:
            name = f"#{random.randint(0, 9999)}"
        if description is None:
            description = f"Experiment: {name}"

        self._name = name
        self._description = description
        self._experiment_id = None
        self._executors = set()

    def run(
        self,
        trial_run_fn: FunctionType,
        trial_gen_fn: FunctionType,
        timeout: Optional[int] = None,
        n_replicates: int = 1,
        executor: Optional[Executor] = None,
    ) -> Executor:
        # Create experiment
        if self._experiment_id is None:
            self._experiment_id, _ = self._store.get_or_create_experiment(
                self._name, self._description
            )

        # Create trials
        trial_params = []
        pending_trials = []
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
                        "experiment_id": self._experiment_id,
                        "task_id": task.id,
                        "method_id": method.id,
                        "replicate_num": replicate_num,
                        "meta": meta,
                    }
                    trial_params.append(trial_param)
                    pending_trials.append(
                        Trial(None, self._store, task, method, replicate_num, meta, [])
                    )

        # Save to store
        trial_ids = self._store.create_trials(trial_params)
        for _id, trial in zip(trial_ids, pending_trials):
            trial._id = _id

        # Run trials
        if executor is None:
            executor = LocalMachine(self._store)
        self._executors.add(executor)
        executor.submit(trial_run_fn, pending_trials, timeout=timeout)
        return executor

    def wait_until_complete(self):
        """Wait for all executors to run and then return."""
        for executor in self._executors:
            executor.join()

    def _experiment(self) -> Optional[Experiment]:
        """Retrieves experiment snapshot that contains trial and assets.

        This method is kept private due to an unstable API.

        Returns:
            Experiment (Optional[Experiment]): Experiment snapshot.
        """
        self._experiment_id = self._store.get_experiment(self._name)
        if self._experiment_id is None:
            return None

        self._experiment_id, _ = self._store.get_or_create_experiment(
            self._name, self._description
        )
        trials = list(self._store.iter_experiment_trials(self._experiment_id))
        experiment = Experiment(
            self._experiment_id, self._name, self._description, trials
        )
        return experiment

    def status(self) -> Optional[pd.DataFrame]:
        """Retrieves all trial's status and associated information.

        Returns:
            Trial statuses (Optional[pandas.DataFrame]): Experiment's trials' status.
        """
        self._experiment_id = self._store.get_experiment(self._name)
        if self._experiment_id is None:
            return None

        records = list(self._store.iter_status(self._experiment_id))
        return pd.DataFrame.from_records(records)

    def results(self) -> Optional[pd.DataFrame]:
        """Retrieves trial measures of an experiment in long form.

        Returns:
            Results (pandas.DataFrame): Measures of an experiment in long form.
        """
        self._experiment_id = self._store.get_experiment(self._name)
        if self._experiment_id is None:
            return None

        records = list(self._store.iter_results(self._experiment_id))
        return pd.DataFrame.from_records(records)

    def available_tasks(self, include_measures=False) -> Optional[pd.DataFrame]:
        """Retrieves available tasks to run a benchmark against.

        Args:
            include_measures (bool): Includes measure columns in long form.

        Returns:
            Results (pandas.DataFrame): Available tasks including their measures.
        """
        records = list(
            self._store.iter_available_tasks(include_measures=include_measures)
        )
        return pd.DataFrame.from_records(records)

    @property
    def name(self):
        return self._name

    @property
    def description(self):
        return self._description
