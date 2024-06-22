""" Azure Container Instances runtime.

# See more details here:
https://docs.microsoft.com/en-us/python/api/overview/azure/container-instance?view=azure-python
"""

from powerlift.executors.localmachine import LocalMachine
from powerlift.bench.store import Store
from typing import Iterable, List
from powerlift.executors.base import handle_err
import random


def _wait_for_completed_worker(results):
    import time

    if len(results) == 0:
        return None

    while True:
        for worker_id, result in results.items():
            if result is None or result.done():
                del results[worker_id]
                return worker_id
        time.sleep(1)


def _run(tasks, azure_json, num_cores, mem_size_gb, n_running_containers, delete_group_container_on_complete, batch_id):
    from azure.mgmt.containerinstance.models import (
        ContainerGroup,
        Container,
        ContainerGroupRestartPolicy,
        EnvironmentVariable,
        ResourceRequests,
        ResourceRequirements,
        OperatingSystemTypes,
    )
    from azure.mgmt.containerinstance import ContainerInstanceManagementClient
    from azure.mgmt.resource import ResourceManagementClient
    from azure.identity import ClientSecretCredential

    credential = ClientSecretCredential(
        tenant_id=azure_json["tenant_id"],
        client_id=azure_json["client_id"],
        client_secret=azure_json["client_secret"],
    )
    resource_group_name = azure_json["resource_group"]
    aci_client = ContainerInstanceManagementClient(
        credential, azure_json["subscription_id"]
    )
    res_client = ResourceManagementClient(credential, azure_json["subscription_id"])
    resource_group = res_client.resource_groups.get(resource_group_name)

    # Run until completion.
    container_counter = 0
    n_tasks = len(tasks)
    n_containers = min(n_tasks, n_running_containers)
    results = {x: None for x in range(n_containers)}
    container_group_names = set()
    while len(tasks) != 0:
        params = tasks.pop(0)
        worker_id = _wait_for_completed_worker(results)

        trial_ids, uri, timeout, raise_exception, image = params
        env_vars = [
            EnvironmentVariable(
                name="TRIAL_IDS", value=",".join([str(x) for x in trial_ids])
            ),
            EnvironmentVariable(name="DB_URL", secure_value=uri),
            EnvironmentVariable(name="TIMEOUT", value=timeout),
            EnvironmentVariable(name="RAISE_EXCEPTION", value=raise_exception),
        ]
        container_resource_requests = ResourceRequests(
            cpu=num_cores,
            memory_in_gb=mem_size_gb,
        )
        container_resource_requirements = ResourceRequirements(
            requests=container_resource_requests
        )
        container_name = f"powerlift-container-{container_counter}"
        container_counter += 1
        container = Container(
            name=container_name,
            image=image,
            resources=container_resource_requirements,
            command=["python", "-m", "powerlift.run"],
            environment_variables=env_vars,
        )
        container_group = ContainerGroup(
            location=resource_group.location,
            containers=[container],
            os_type=OperatingSystemTypes.linux,
            restart_policy=ContainerGroupRestartPolicy.never,
        )
        container_group_name = f"powerlift-container-group-{worker_id}-{batch_id}"
        result = aci_client.container_groups.begin_create_or_update(
            resource_group.name, container_group_name, container_group
        )

        container_group_names.add(container_group_name)
        results[worker_id] = result

    # Wait for all container groups to complete
    while _wait_for_completed_worker(results) is not None:
        pass

    # Delete all container groups
    if delete_group_container_on_complete:
        for container_group_name in container_group_names:
            aci_client.container_groups.begin_delete(
                resource_group_name, container_group_name
            )
    return None


class AzureContainerInstance(LocalMachine):
    """Runs trials on Azure Container Instances."""

    def __init__(
        self,
        store: Store,
        azure_tenant_id: str,
        azure_client_id: str,
        azure_client_secret: str,
        subscription_id: str,
        resource_group: str,
        image: str = "interpretml/powerlift:0.1.9",
        n_running_containers: int = 1,
        num_cores: int = 1,
        mem_size_gb: int = 2,
        wheel_filepaths: List[str] = None,
        docker_db_uri: str = None,
        raise_exception: bool = False,
        delete_group_container_on_complete: bool = True
    ):
        """Runs remote execution of trials via Azure Container Instances.

        Args:
            store (Store): Store that houses trials.
            azure_tenant_id (str): Azure tentant ID.
            azure_client_id (str): Azure client ID.
            azure_client_secret (str): Azure client secret.
            subscription_id (str): Azure subscription ID.
            resource_group (str): Azure resource group.
            image (str, optional): Image to execute. Defaults to "interpretml/powerlift:0.0.1".
            n_running_containers (int, optional): Max number of containers to run simultaneously. Defaults to 1.
            num_cores (int, optional): Number of cores per container. Defaults to 1.
            mem_size_gb (int, optional): RAM size in GB per container. Defaults to 2.
            wheel_filepaths (List[str], optional): List of wheel filepaths to install on ACI trial run. Defaults to None.
            docker_db_uri (str, optional): Database URI for container. Defaults to None.
            raise_exception (bool, optional): Raise exception on failure.
            delete_group_container_on_complete (bool, optional): Delete group containers after completion. Defaults to True.
        """
        from multiprocessing import Manager

        self._image = image
        self._n_running_containers = n_running_containers
        self._num_cores = num_cores
        self._mem_size_gb = mem_size_gb
        self._delete_group_container_on_complete = delete_group_container_on_complete

        self._docker_db_uri = docker_db_uri
        self._azure_json = {
            "tenant_id": azure_tenant_id,
            "client_id": azure_client_id,
            "client_secret": azure_client_secret,
            "subscription_id": subscription_id,
            "resource_group": resource_group,
        }
        self._batch_id = random.getrandbits(64)
        super().__init__(store=store, n_cpus=1, raise_exception=raise_exception, wheel_filepaths=wheel_filepaths)

    def delete_credentials(self):
        """Deletes credentials in object for accessing Azure Resources."""
        del self._azure_json

    def submit(self, trial_run_fn, trials: Iterable, timeout=None):
        uri = (
            self._docker_db_uri if self._docker_db_uri is not None else self._store.uri
        )
        tasks = []
        self._store.add_trial_run_fn(
            [x.id for x in trials], trial_run_fn, self._wheel_filepaths
        )
        for trial in trials:
            params = (
                [trial.id],
                uri,
                timeout,
                self._raise_exception,
                self._image,
            )
            tasks.append(params)

        params = (
            tasks,
            self._azure_json,
            self._num_cores,
            self._mem_size_gb,
            self._n_running_containers,
            self._delete_group_container_on_complete,
            self._batch_id,
        )
        self._batch_id = random.getrandbits(64)
        if self._pool is None:
            try:
                res = _run(*params)
                self._trial_id_to_result[0] = res
            except Exception as e:
                self._trial_id_to_result[0] = e
        else:
            self._trial_id_to_result[0] = self._pool.apply_async(
                _run,
                params,
                error_callback=handle_err,
            )
