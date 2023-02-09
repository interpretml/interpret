""" Runs all trials on local machine that is running powerlift.

This currently uses multiprocessing pools to handle parallelism.
"""

from powerlift.bench.store import Store
from powerlift.executors.base import Executor
from multiprocessing import Pool
from typing import Iterable, List
from powerlift.executors.base import handle_err


class LocalMachine(Executor):
    def __init__(
        self,
        store: Store,
        n_cpus: int = None,
        raise_exception: bool = False,
        wheel_filepaths: List[str] = None,
    ):
        """Runs trial runs on the local machine.

        Args:
            store (Store): Store that houses trials.
            n_cpus (int, optional): Max number of cpus to run on.. Defaults to None.
            raise_exception (bool, optional): Raise exception on failure. Defaults to False.
            wheel_filepaths (List[str], optional): List of wheel filepaths to install on ACI trial run. Defaults to None.
        """
        if n_cpus != 1:
            self._pool = Pool(processes=n_cpus)
        else:
            self._pool = None

        self._trial_id_to_result = {}
        self._store = store
        self._n_cpus = n_cpus
        self._raise_exception = raise_exception
        self._wheel_filepaths = wheel_filepaths

    def __del__(self):
        if self._pool is not None:
            self._pool.close()

    def submit(self, trial_run_fn, trials: Iterable, timeout=None):
        from powerlift.run import __main__ as runner

        self._store.add_trial_run_fn(
            [x.id for x in trials], trial_run_fn, self._wheel_filepaths
        )
        for trial in trials:
            if self._pool is None:
                try:
                    res = runner.run_trials([trial.id], self._store.uri, timeout, False)
                    self._trial_id_to_result[trial.id] = res
                except Exception as e:
                    self._trial_id_to_result[trial.id] = e
                    if self._raise_exception:
                        raise e
            else:
                self._trial_id_to_result[trial.id] = self._pool.apply_async(
                    runner.run_trials,
                    ([trial.id], self._store.uri, timeout, self._raise_exception),
                    error_callback=handle_err,
                )

    def join(self):
        if self._pool is not None:
            for _, result in self._trial_id_to_result.items():
                result.get()

    def cancel(self):
        if self._pool is not None:
            self._pool.terminate()

    @property
    def n_cpus(self):
        return self._n_cpus

    @property
    def store(self):
        return self._store

    @property
    def raise_exception(self):
        return self._raise_exception

    @property
    def wheel_filepaths(self):
        return self._wheel_filepaths
