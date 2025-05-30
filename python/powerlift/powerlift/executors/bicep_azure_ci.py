"""Azure Container Instances runtime created via bicep.

# See more details here:
https://docs.microsoft.com/en-us/python/api/overview/azure/container-instance?view=azure-python
"""

import random
from multiprocessing import Pool
from typing import List, Optional

from powerlift.bench.store import Store
from powerlift.executors.base import Executor
from pathlib import Path


class BicepAzureContainerInstance(Executor):
    """Runs trials on Azure Container Instances via bicep."""

    def __init__(
        self,
        store: Store,
        azure_tenant_id: str,
        subscription_id: str,
        azure_client_id: str,
        credential=None,
        azure_client_secret: Optional[str] = None,
        resource_group: str = "powerlift_rg",
        shell_install: Optional[str] = None,
        pip_install: Optional[str] = None,
        wheel_filepaths: Optional[List[str]] = None,
        n_instances: int = 1,
        num_cores: int = 4,
        mem_size_gb: int = 16,
        # other images available at:
        # https://mcr.microsoft.com/en-us/product/devcontainers/python/tags
        # alternative: "mcr.microsoft.com/devcontainers/python:latest"
        image: str = "mcr.microsoft.com/devcontainers/python:3.12",
        docker_db_uri: Optional[str] = None,
        resource_uris: Optional[List[str]] = None,
        delete_on_complete: bool = True,
    ):
        """Runs remote execution of trials via Azure Container Instances.

        Args:
            store (Store): Store that houses trials.
            azure_tenant_id (str): Azure tentant ID.
            subscription_id (str): Azure subscription ID.
            azure_client_id (str): Azure client ID.
            credential: Azure credential
            azure_client_secret (str): Azure client secret.
            resource_group (str): Azure resource group.
            shell_install (str): apt-get install parameters.
            pip_install (str): pip install parameters.
            wheel_filepaths (List[str], optional): List of wheel filepaths to install on ACI trial run. Defaults to None.
            n_instances (int, optional): Max number of containers to run simultaneously. Defaults to 1.
            num_cores (int, optional): Number of cores per container. Defaults to 1.
            mem_size_gb (int, optional): RAM size in GB per container. Defaults to 2.
            image (str, optional): Image to execute. Defaults to "mcr.microsoft.com/devcontainers/python:3.12".
            docker_db_uri (str, optional): Database URI for container. Defaults to None.
            resource_uris (List[str], optional): Azure resources to grant contributor access permissions to.
            delete_on_complete (bool, optional): Delete group containers after completion. Defaults to True.
        """

        self._image = image
        self._shell_install = shell_install
        self._pip_install = pip_install
        self._n_instances = n_instances
        self._num_cores = num_cores
        self._mem_size_gb = mem_size_gb
        self._delete_on_complete = delete_on_complete

        self._docker_db_uri = docker_db_uri

        self._resource_uris = resource_uris
        self._azure_json = {
            "credential": credential,
            "tenant_id": azure_tenant_id,
            "client_id": azure_client_id,
            "client_secret": azure_client_secret,
            "subscription_id": subscription_id,
            "resource_group": resource_group,
        }
        self._batch_id = random.getrandbits(64)

        self._pool = Pool()
        self._runner_id_to_result = {}
        self._store = store
        self._wheel_filepaths = wheel_filepaths

        self._deployment_poll = None
        self._deployment_name = None

    def __del__(self):
        if self._pool is not None:
            self._pool.close()

    def delete_credentials(self):
        """Deletes credentials in object for accessing Azure Resources."""
        del self._azure_json

    def submit(self, experiment_id, timeout=None):
        from azure.mgmt.resource import ResourceManagementClient
        from azure.mgmt.resource.resources.models import (
            Deployment,
            DeploymentProperties,
        )
        import json
        import uuid

        script_dir = Path(__file__).resolve().parent
        start_script_path = script_dir / "startup.sh"
        template_file_path = script_dir / "aci.json"
        with open(template_file_path, "r") as template_file:
            template_content = json.load(template_file)
        with open(start_script_path, "r") as start_script:
            start_content = start_script.read().replace("\r\n", "\n")

        subscription_id = self._azure_json["subscription_id"]
        resource_group = self._azure_json["resource_group"]
        credential = self._azure_json["credential"]

        uri = (
            self._docker_db_uri if self._docker_db_uri is not None else self._store.uri
        )
        resource_client = ResourceManagementClient(credential, subscription_id)
        parameters = {
            "containerCount": {"value": self._n_instances},
            "batchSize": {"value": 50},
            "containerImage": {"value": self._image},
            "startupScript": {"value": start_content},
            "experimentId": {"value": str(experiment_id)},
            "dbUrl": {"type": "secureString", "value": uri},
            "timeout": {"value": timeout},
            "resourceGroupName": {"value": resource_group},
            "subscriptionId": {"value": subscription_id},
        }

        deployment = Deployment(
            properties=DeploymentProperties(
                template=template_content, mode="Incremental", parameters=parameters
            )
        )
        deployment_name = f"powerlift-{uuid.uuid4()}"

        # Deploy the template
        self._deployment_poll = resource_client.deployments.begin_create_or_update(
            resource_group, deployment_name, deployment
        )
        self._deployment_name = deployment_name

    def join(self):
        if self._deployment_poll is not None:
            self._deployment_poll.wait()

        results = []
        return results

    def cancel(self):
        if self._deployment_poll is not None:
            from azure.mgmt.resource import ResourceManagementClient

            resource_client = ResourceManagementClient(
                self._azure_json["credential"], self._azure_json["subscription_id"]
            )
            cancel_operation = resource_client.deployments.begin_cancel(
                self._azure_json["resource_group"], self._deployment_name
            )
            cancel_operation.wait()

    @property
    def store(self):
        return self._store

    @property
    def wheel_filepaths(self):
        return self._wheel_filepaths
