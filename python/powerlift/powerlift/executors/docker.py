"""Runs docker containers locally.

This is handy for testing if dockerfiles work with powerlift before
sending it a remote container service.
"""

import multiprocessing
from typing import List, Optional

from powerlift.bench.store import Store
from powerlift.executors.base import handle_err
from powerlift.executors.localmachine import LocalMachine


def _run_docker(experiment_id, runner_id, db_url, timeout, image):
    import docker

    client = docker.from_env()

    container = client.containers.run(
        image,
        "python -m powerlift.run",
        environment={
            "EXPERIMENT_ID": experiment_id,
            "RUNNER_ID": runner_id,
            "DB_URL": db_url,
            "TIMEOUT": timeout,
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
        image: str = "mcr.microsoft.com/devcontainers/python:latest",
        n_running_containers: Optional[int] = None,
        wheel_filepaths: Optional[List[str]] = None,
        docker_db_uri: Optional[str] = None,
    ):
        """Runs trials in local docker containers.

        Args:
            store (Store): Store that houses trials.
            image (str, optional): Image to execute in container. Defaults to "mcr.microsoft.com/devcontainers/python:latest".
            n_running_containers (int, optional): Max number of containers running simultaneously. Defaults to None.
            wheel_filepaths (List[str], optional): List of wheel filepaths to install on docker trial run. Defaults to None.
            docker_db_uri (str, optional): Database URI for container. Defaults to None.
        """
        self._docker_db_uri = docker_db_uri
        self._image = image
        super().__init__(
            store=store,
            n_cpus=n_running_containers,
            raise_exception=False,
            wheel_filepaths=wheel_filepaths,
        )

    def submit(self, experiment_id, timeout=None):
        uri = (
            self._docker_db_uri if self._docker_db_uri is not None else self._store.uri
        )

        n_runners = (
            multiprocessing.cpu_count() if self._n_cpus is None else self._n_cpus
        )
        for runner_id in range(n_runners):
            self._runner_id_to_result[runner_id] = self._pool.apply_async(
                _run_docker,
                (
                    experiment_id,
                    runner_id,
                    uri,
                    timeout,
                    self._image,
                ),
                error_callback=handle_err,
            )

    @property
    def docker_db_uri(self):
        return self._docker_db_uri

    @property
    def image(self):
        return self._image
