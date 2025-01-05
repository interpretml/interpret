"""User facing benchmark class."""

import inspect
import os
import pathlib
import random
from types import FunctionType
from typing import Optional, Union

import numpy as np
import pandas as pd

from powerlift.bench.experiment import Experiment
from powerlift.bench.store import Store
from powerlift.db import schema as db
from powerlift.executors.base import Executor


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
            msg = f"Incorrect type: {type(store_or_uri)}"
            raise TypeError(msg)

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
        script_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "run", "__main__.py"
        )
        with open(script_file) as file:
            script_contents = file.read().replace("\r\n", "\n")

        shell_install = None
        if hasattr(executor, "_shell_install"):
            shell_install = executor._shell_install
        if shell_install is None:
            shell_install = ""

        pip_install = None
        if hasattr(executor, "_pip_install"):
            pip_install = executor._pip_install
        if pip_install is None:
            pip_install = ""

        wheels = []
        if hasattr(executor, "_wheel_filepaths"):
            wheel_filepaths = executor._wheel_filepaths
            if wheel_filepaths is not None:
                for wheel_filepath in wheel_filepaths:
                    with open(wheel_filepath, "rb") as f:
                        content = f.read()
                    name = pathlib.Path(wheel_filepath).name
                    wheel = db.Wheel(name=name, embedded=content)
                    wheels.append(wheel)

        trial_fn = inspect.getsource(trial_run_fn)

        tasks = self._store.get_tasks()

        # Do the hardest tasks first so that we can slip
        # the faster ones into the cracks, but do some easy
        # ones first just to verify everything works.
        tasks = sorted(
            tasks,
            reverse=True,
            key=lambda x: (1 if x.n_classes < 3 else x.n_classes)
            * x.n_features
            * x.n_samples,
        )

        # Create experiment
        if self._experiment_id is None:
            experiment_id = self._store.create_experiment(
                self._name,
                self._description,
                shell_install,
                pip_install,
                script_contents,
                trial_fn,
                wheels,
            )

        # Create trials
        trials = []
        task_trials = []
        for task in tasks:
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
                for replicate_num in range(n_replicates):
                    trial_param = {
                        "experiment_id": experiment_id,
                        "task_id": task.id,
                        "method": method,
                        "replicate_num": replicate_num,
                        "meta": meta,
                    }
                    task_trials.append(trial_param)
            random.shuffle(task_trials)
            trials.extend(task_trials)
            task_trials = []

        # put some of the easy ones first so that we find problems quickly
        trials = np.array(trials, dtype=object)
        n_fastest = int(len(trials) * 0.25)
        n_true = int(n_fastest * 0.25)
        take = [True] * n_true + [False] * (n_fastest - n_true)
        random.shuffle(take)
        take = np.array([False] * (len(trials) - n_fastest) + take, dtype=bool)
        trials = np.concatenate([trials[take][::-1], trials[~take]])

        # Save to store
        self._store.create_trials(trials)
        self._experiment_id = experiment_id

        # Run trials
        if executor is None:
            from powerlift.executors import LocalMachine

            executor = LocalMachine(self._store)
        self._executors.add(executor)
        executor.submit(experiment_id, timeout=timeout)
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

        self._experiment_id, shell_install, pip_install, script, _ = (
            self._store.get_or_create_experiment(self._name, self._description)
        )
        trials = list(self._store.iter_experiment_trials(self._experiment_id))
        return Experiment(
            self._experiment_id,
            self._name,
            self._description,
            shell_install,
            pip_install,
            script,
            trials,
        )

    def status(self) -> Optional[pd.DataFrame]:
        """Retrieves all trial's status and associated information.

        Returns:
            Trial statuses (Optional[pandas.DataFrame]): Experiment's trials' status.
        """
        df = self._store.get_status(self._name)
        return df.sort_values(by=["task", "method", "meta", "replicate_num"])

    def results(self) -> Optional[pd.DataFrame]:
        """Retrieves trial measures of an experiment in long form.

        Returns:
            Results (pandas.DataFrame): Measures of an experiment in long form.
        """

        df = self._store.get_results(self._name)

        # sometimes logs are written twice when a runner attempts a DB transaction
        # and the result is commited in the DB, but the response to the runner fails
        # so remove the duplicates here
        idx = df.groupby(
            ["task", "method", "meta", "replicate_num", "name", "seq_num"]
        )["id"].idxmin()
        df = df.loc[idx]
        df = df.drop(["id"], axis=1)
        return df.sort_values(
            by=["task", "method", "meta", "replicate_num", "name", "seq_num"]
        )

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
