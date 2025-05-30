"""Azure VM Instances runtime."""

import random
from multiprocessing import Pool
from typing import List, Optional

from powerlift.bench.store import Store
from powerlift.executors.base import Executor, handle_err


class AzureVMInstance(Executor):
    """Runs trials on Azure VMs."""

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
        location: Optional[str] = None,  # we use "canadacentral" or "southcentralus"
        vm_size: str = "Standard_B16s_v2",  # can use "Standard_NC24ads_A100_v4" for GPUs
        image_publisher: str = "canonical",
        image_offer: str = "ubuntu-24_04-lts",
        image_sku: str = "server",
        image_version: str = "latest",
        disk_type: str = "Standard_LRS",  # use "Premium_LRS" for SSDs
        docker_db_uri: Optional[str] = None,
        resource_uris: Optional[List[str]] = None,
        max_undead: int = 1,
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
            location (str, optional): Azure location to create the resources
            vm_size (str): Azure vm_size parameter for the VM
            image_publisher (str): Azure publisher parameter for the VM
            image_offer (str): Azure offer parameter for the VM
            image_sku (str): Azure sku parameter for the VM
            image_version (str): Azure version parameter for the VM
            disk_type (str): Azure disk type parameter for the VM
            docker_db_uri (str, optional): Database URI for container. Defaults to None.
            resource_uris (List[str], optional): Azure resources to grant contributor access permissions to.
            max_undead (int): maximum number of containers that are allowed to be left alive if there is an error during initialization. Higher numbers increase the speed of initialization, but might incur higher cost if any zombies escape.
            delete_on_complete (bool, optional): Delete group containers after completion. Defaults to True.
        """

        self._credential = credential
        self._shell_install = shell_install
        self._pip_install = pip_install
        self._n_instances = n_instances
        self._location = location
        self._vm_size = vm_size
        self._image_publisher = image_publisher
        self._image_offer = image_offer
        self._image_sku = image_sku
        self._image_version = image_version
        self._disk_type = disk_type
        self._docker_db_uri = docker_db_uri
        self._resource_uris = resource_uris
        self._max_undead = max_undead
        self._delete_on_complete = delete_on_complete

        self._azure_json = {
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

    def __del__(self):
        if self._pool is not None:
            self._pool.close()

    def delete_credentials(self):
        """Deletes credentials in object for accessing Azure Resources."""
        del self._azure_json

    def submit(self, experiment_id, timeout=None):
        from powerlift.run import azure_vm as remote_process

        uri = (
            self._docker_db_uri if self._docker_db_uri is not None else self._store.uri
        )

        params = (
            experiment_id,
            self._n_instances,
            uri,
            self._resource_uris,
            timeout,
            self._azure_json,
            self._credential,
            self._location,
            self._vm_size,
            self._image_publisher,
            self._image_offer,
            self._image_sku,
            self._image_version,
            self._disk_type,
            self._max_undead,
            self._delete_on_complete,
            self._batch_id,
        )
        self._batch_id = random.getrandbits(64)
        self._runner_id_to_result[0] = self._pool.apply_async(
            remote_process.run_azure_process,
            params,
            error_callback=handle_err,
        )

    def join(self):
        results = []
        if self._pool is not None:
            for result in self._runner_id_to_result.values():
                res = result.get()
                results.append(res)
        return results

    def cancel(self):
        if self._pool is not None:
            self._pool.terminate()

    @property
    def store(self):
        return self._store

    @property
    def wheel_filepaths(self):
        return self._wheel_filepaths
