""" Runs docker containers locally.

This is handy for testing if dockerfiles work with powerlift before
sending it a remote container service.
"""

from powerlift.bench.store import Store
from powerlift.executors.localmachine import LocalMachine
from powerlift.executors.base import handle_err
from typing import Iterable, List


def _run_docker(trial_ids, db_url, timeout, raise_exception, image):
    import docker

    client = docker.from_env()

    container = client.containers.run(
        image,
        "python -m powerlift.run",
        environment={
            "TRIAL_IDS": ",".join([str(x) for x in trial_ids]),
            "DB_URL": db_url,
            "TIMEOUT": timeout,
            "RAISE_EXCEPTION": raise_exception,
        },
        network_mode="host",
        detach=True,
    )
    exit_code = container.wait()
    container.remove()
    return exit_code


class InsecureDocker(LocalMachine):
    """Runs trials in local docker containers.

    Make sure the machine you run on is fully trusted.
    Environment variables are used to pass the database connection string,
    which means other users on the machine may be able to see it via enumerating
    the processes on the machine and their args.
    """

    def __init__(
        self,
        store: Store,
        image: str = "interpretml/powerlift:0.1.9",
        n_running_containers: int = None,
        wheel_filepaths: List[str] = None,
        docker_db_uri: str = None,
        raise_exception: bool = False
    ):
        """Runs trials in local docker containers.

        Args:
            store (Store): Store that houses trials.
            image (str, optional): Image to execute in container. Defaults to "interpretml/powerlift:0.1.4".
            n_running_containers (int, optional): Max number of containers running simultaneously. Defaults to None.
            wheel_filepaths (List[str], optional): List of wheel filepaths to install on docker trial run. Defaults to None.
            docker_db_uri (str, optional): Database URI for container. Defaults to None.
            raise_exception (bool, optional): Raise exception on failure.
        """
        self._docker_db_uri = docker_db_uri
        self._image = image
        super().__init__(store=store, n_cpus=n_running_containers, raise_exception=raise_exception, wheel_filepaths=wheel_filepaths)

    def submit(self, trial_run_fn, trials: Iterable, timeout=None):
        uri = (
            self._docker_db_uri if self._docker_db_uri is not None else self._store.uri
        )
        self._store.add_trial_run_fn(
            [x.id for x in trials], trial_run_fn, self._wheel_filepaths
        )
        for trial in trials:
            self._trial_id_to_result[trial.id] = self._pool.apply_async(
                _run_docker,
                ([trial.id], uri, timeout, self._raise_exception, self._image),
                error_callback=handle_err,
            )

    @property
    def docker_db_uri(self):
        return self._docker_db_uri

    @property
    def image(self):
        return self._image
