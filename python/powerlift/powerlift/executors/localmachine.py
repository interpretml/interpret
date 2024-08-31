""" Runs all trials on local machine that is running powerlift.

This currently uses multiprocessing pools to handle parallelism.
"""

from powerlift.bench.store import Store
from powerlift.executors.base import Executor
from multiprocessing import Pool
from typing import List
from powerlift.executors.base import handle_err
import multiprocessing


class LocalMachine(Executor):
    def __init__(
        self,
        store: Store,
        n_cpus: int = None,
        debug_mode: bool = False,
        wheel_filepaths: List[str] = None,
        raise_exception: bool = False,
    ):
        """Runs trial runs on the local machine.

        Args:
            store (Store): Store that houses trials.
            n_cpus (int, optional): Max number of cpus to run on.. Defaults to None.
            debug_mode (bool, optional): Restricts to a single thread and raises exceptions. Good for debugging.
            wheel_filepaths (List[str], optional): List of wheel filepaths to install on trial run. Defaults to None.
            raise_exception (bool, optional): Raise exception on failure.
        """
        if debug_mode:
            self._pool = None
        else:
            self._pool = Pool(processes=n_cpus)

        self._runner_id_to_result = {}
        self._store = store
        self._n_cpus = n_cpus
        self._debug_mode = debug_mode
        self._wheel_filepaths = wheel_filepaths
        self._raise_exception = raise_exception

    def __del__(self):
        if self._pool is not None:
            self._pool.close()

    def submit(self, experiment_id, trials: List, timeout=None):
        from powerlift.run import __main__ as runner

        n_runners = min(
            len(trials),
            multiprocessing.cpu_count() if self._n_cpus is None else self._n_cpus,
        )
        for runner_id in range(n_runners):
            if self._pool is None:
                try:
                    debug_fn = trial_run_fn if self._debug_mode else None
                    res = runner.run_trials(
                        experiment_id,
                        runner_id,
                        self._store.uri,
                        timeout,
                        self._raise_exception or self._debug_mode,
                        debug_fn=debug_fn,
                    )
                    self._runner_id_to_result[runner_id] = res
                except Exception as e:
                    self._runner_id_to_result[runner_id] = e
                    if self._raise_exception or self._debug_mode:
                        raise
            else:
                self._runner_id_to_result[runner_id] = self._pool.apply_async(
                    runner.run_trials,
                    (
                        experiment_id,
                        runner_id,
                        self._store.uri,
                        timeout,
                        self._raise_exception or self._debug_mode,
                    ),
                    error_callback=handle_err,
                )

    def join(self):
        results = []
        if self._pool is not None:
            for _, result in self._runner_id_to_result.items():
                res = result.get()
                results.append(res)
        return results

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
    def debug_mode(self):
        return self._debug_mode

    @property
    def wheel_filepaths(self):
        return self._wheel_filepaths
